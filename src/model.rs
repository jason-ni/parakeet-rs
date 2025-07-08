use anyhow::{Context};
use std::path::Path;
use std::io::{self, BufRead, BufReader};
use std::ops::Add;
use ndarray::{ArrayViewD, ArrayD, IxDyn, arr2, ArrayBase, OwnedRepr, Array2, Ix2, Axis};
use ndarray_stats::QuantileExt;
use crate::errors::ParakeetError;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;

pub struct ParakeetModel {
    encoder: Session,
    decoder: Session,
    joint_pred: Session,
    joint_enc: Session,
    joint_net: Session,
    preprocessor: Session,
}

fn log_softmax(input: &ArrayViewD<f32>, axis: Axis) -> Result<ArrayD<f32>, ParakeetError> {
    let max = input.max().context("Failed to compute max")?.to_owned();
    // 2. Subtract the max from the input for numerical stability.
    //    Broadcasting automatically handles this subtraction.
    let x_minus_max = input - max;

    // 3. Compute the exponentials of the shifted values.
    let exps = x_minus_max.mapv(f32::exp);

    // 4. Sum the exponentials along the axis.
    let sum_of_exps = exps.sum_axis(axis);

    // 5. Take the natural logarithm of the sum.
    //    Again, we must insert an axis to make it broadcastable for the final subtraction.
    let log_sum_of_exps = sum_of_exps.mapv(f32::ln).insert_axis(axis);

    // 6. Final subtraction to get the log_softmax values.
    //    This is the final result: (x_i - m) - log(sum(exp(x_j - m)))
    let log_softmax_values = x_minus_max - &log_sum_of_exps;

    Ok(log_softmax_values)
}

impl ParakeetModel {
    pub fn new<P: AsRef<Path>>(model_dir: P, is_quantized: bool) -> Result<Self, ParakeetError> {
        let encoder_model_name = if is_quantized {
            "encoder.int8.onnx"
        } else {
            "full_encoder.onnx"
        };
        let encoder = Session::builder().unwrap()
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_parallel_execution(true)?
            .with_intra_threads(1)?
            .with_inter_threads(4)?
            .commit_from_file(model_dir.as_ref().join(encoder_model_name))?;

        let decoder = Session::builder().unwrap()
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_parallel_execution(true)?
            .with_intra_threads(1)?
            .with_inter_threads(4)?
            .commit_from_file(model_dir.as_ref().join("decoder.onnx"))?;

        let joint_pred = Session::builder().unwrap()
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_parallel_execution(true)?
            .with_intra_threads(1)?
            .with_inter_threads(4)?
            .commit_from_file(model_dir.as_ref().join("joint.pred.onnx"))?;

        let joint_enc = Session::builder().unwrap()
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_parallel_execution(true)?
            .with_intra_threads(1)?
            .with_inter_threads(4)?
            .commit_from_file(model_dir.as_ref().join("joint.enc.onnx"))?;

        let joint_net = Session::builder().unwrap()
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_parallel_execution(true)?
            .with_intra_threads(1)?
            .with_inter_threads(4)?
            .commit_from_file(model_dir.as_ref().join("joint.joint_net.onnx"))?;

        let preprocessor = Session::builder().unwrap()
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_parallel_execution(true)?
            .with_intra_threads(1)?
            .with_inter_threads(4)?
            .commit_from_file(model_dir.as_ref().join("nemo128.onnx"))?;
        Ok(ParakeetModel {
            encoder,
            decoder,
            joint_pred,
            joint_enc,
            joint_net,
            preprocessor,
        })
    }

    pub fn fbank_features(&self, audio: &ArrayViewD<f32>, audio_len: i64) -> Result<ArrayD<f32>, ParakeetError> {
        let audio_size = ndarray::Array1::from_shape_fn(1, |_| audio_len as i64);
        let inputs = ort::inputs![
            "waveforms" => audio.clone().into_dyn(),
            "waveforms_lens" => audio_size,
        ]?;
        let outputs = self.preprocessor.run(inputs)?;
        let features: ArrayViewD<f32> = outputs.get("features").unwrap().try_extract_tensor()?;
        Ok(features.to_owned())
    }

    pub fn encoder_infer(&self, features: &ArrayD<f32>) -> Result<ArrayD<f32>, ParakeetError> {
        let input_length= features.shape()[2];
        let input_length = ndarray::Array1::from_shape_fn(1, |_| input_length as i64);
        let inputs = ort::inputs![
            "audio_signal" => features.clone().into_dyn(),
            "length" => input_length,
        ]?;
        let outputs = self.encoder.run(inputs)?;
        let encoder_output: ArrayViewD<f32> = outputs.get("outputs").unwrap().try_extract_tensor()?;
        let encoder_output = encoder_output.permuted_axes(IxDyn(&[0, 2, 1]));
        Ok(encoder_output.to_owned())
    }

    pub fn joint_enc_infer(&self, encoder_output: &ArrayD<f32>) -> Result<ArrayD<f32>, ParakeetError> {
        let inputs = ort::inputs![
            "input" => encoder_output.clone().into_dyn(),
        ]?;
        let outputs = self.joint_enc.run(inputs)?;
        let res = outputs.get("output").unwrap().try_extract_tensor()?;
        Ok(res.to_owned())
    }

    pub fn get_init_decoder_state(&self) -> Result<(ArrayD<f32>, ArrayD<f32>), ParakeetError> {
        let state0 = ArrayD::zeros(IxDyn(&[2, 1, 640])); // TODO: hardcoded
        let state1 = ArrayD::zeros(IxDyn(&[2, 1, 640]));
        Ok((state0, state1))
    }

    pub fn decoder_infer(
        &self,
        target: i64,
        state0: &ArrayD<f32>,
        state1: &ArrayD<f32>,
    ) -> Result<(ArrayD<f32>, ArrayD<f32>, ArrayD<f32>), ParakeetError> {
        let target_array: ArrayD<i32> = ArrayD::from_shape_fn(IxDyn(&[1, 1]), |_| target as i32);
        let target_length = ndarray::Array1::from_shape_fn(1, |_| 1i32);
        let inputs = ort::inputs![
            "targets" => target_array.into_dyn(),
            "target_length" => target_length,
            "states.1" => state0.clone().into_dyn(),
            "onnx::Slice_3" => state1.clone().into_dyn(),
        ]?;
        let outputs = self.decoder.run(inputs)?;
        let decoder_output: ArrayViewD<f32> = outputs.get("outputs").unwrap().try_extract_tensor()?;
        let decoder_output = decoder_output.permuted_axes(IxDyn(&[0, 2, 1]));
        let state0_next: ArrayViewD<f32> = outputs.get("states").unwrap().try_extract_tensor()?;
        let state1_next: ArrayViewD<f32> = outputs.get("162").unwrap().try_extract_tensor()?;
        Ok((
            decoder_output.to_owned(),
            state0_next.to_owned(),
            state1_next.to_owned(),
        ))
    }

    pub fn joint_pred_infer(
        &self,
        decoder_output: &ArrayD<f32>,
    ) -> Result<ArrayD<f32>, ParakeetError> {
        let inputs = ort::inputs![
            "onnx::MatMul_0" => decoder_output.clone().into_dyn(),
        ]?;
        let outputs = self.joint_pred.run(inputs)?;
        let res = outputs.get("5").unwrap().try_extract_tensor()?;
        Ok(res.to_owned())
    }

    pub fn joint_net_infer(
        &self,
        f: &ArrayD<f32>,
        g: &ArrayD<f32>,
    ) -> Result<ArrayD<f32>, ParakeetError> {
        let ff = f.to_shape(IxDyn(&[1, 1, 1, 640]))?;
        let gg = g.to_shape(IxDyn(&[1, 1, 1, 640]))?;
        let input: ArrayD<f32> = ff.add(gg).to_owned();
        let inputs = ort::inputs![
            "input.1" => input.clone().into_dyn(),
        ]?;
        let outputs = self.joint_net.run(inputs)?;
        let res = outputs.get("6").unwrap().try_extract_tensor()?;
        let res = log_softmax(&res, Axis(3))?;
        Ok(res)
    }
}

pub struct ParakeetTokenizer {
    tokens: Vec<String>,
}

impl ParakeetTokenizer {
    pub fn new<P: AsRef<Path>>(model_dir: P) -> Result<Self, ParakeetError> {
        let file = std::fs::File::open(model_dir.as_ref().join("tokens.txt"))?;
        let reader = BufReader::new(file);
        let tokens: Vec<String> = reader
            .lines()
            .map(|line| {
                line.unwrap().split_whitespace().next().unwrap().to_string()
            }).collect();
        Ok(ParakeetTokenizer { tokens })
    }

    pub fn blank_id(&self) -> i64 {
        self.tokens.len() as i64 - 1
    }

    pub fn decode(&self, idx: i64) -> &str {
        &self.tokens[idx as usize]
    }
}