use anyhow::{Context};
use std::path::Path;
use std::io::{BufRead, BufReader};
use std::ops::Add;
use ndarray::{ArrayViewD, ArrayD, IxDyn, Axis};
use ndarray_stats::QuantileExt;
use crate::errors::ParakeetError;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
#[cfg(target_os = "macos")]
use ort::execution_providers::CoreMLExecutionProvider;
#[cfg(not(target_os = "macos"))]
use ort::execution_providers::TensorRTExecutionProvider;
use ort::execution_providers::ExecutionProviderDispatch;
use ort::value::{TensorRef, Tensor};

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

#[allow(unused_variables)]
#[cfg(target_os = "macos")]
fn get_platform_provider<P: AsRef<Path>>(model_dir: P) -> ExecutionProviderDispatch {
    CoreMLExecutionProvider::default().build()
}

#[cfg(not(target_os = "macos"))]
fn get_platform_provider<P: AsRef<Path>>(model_dir: P) -> ExecutionProviderDispatch {
    let trt_cache_dir = model_dir.as_ref().join("trt_cache");
    TensorRTExecutionProvider::default()
        .with_engine_cache(true)
        .with_engine_cache_path(trt_cache_dir.to_str().unwrap())
        .build()
}

impl ParakeetModel {
    pub fn new<P: AsRef<Path>>(model_dir: P, is_quantized: bool) -> Result<Self, ParakeetError> {
        let encoder_model_name = if is_quantized {
            "encoder.int8.onnx"
        } else {
            "full_encoder.onnx"
        };

        let provider = get_platform_provider(model_dir.as_ref());

        let encoder = Session::builder().unwrap()
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_execution_providers([provider.clone()])?
            .with_parallel_execution(true)?
            //.with_intra_threads(1)?
            //.with_inter_threads(1)?
            .commit_from_file(model_dir.as_ref().join(encoder_model_name))?;

        let decoder = Session::builder().unwrap()
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_execution_providers([provider.clone()])?
            .with_parallel_execution(true)?
            //.with_intra_threads(1)?
            //.with_inter_threads(1)?
            .commit_from_file(model_dir.as_ref().join("decoder.int8.onnx"))?;

        let joint_pred = Session::builder().unwrap()
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_execution_providers([provider.clone()])?
            .with_parallel_execution(true)?
            //.with_intra_threads(1)?
            //.with_inter_threads(1)?
            .commit_from_file(model_dir.as_ref().join("joint.pred.int8.onnx"))?;

        let joint_enc = Session::builder().unwrap()
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_parallel_execution(true)?
            .with_execution_providers([provider.clone()])?
            //.with_intra_threads(1)?
            //.with_inter_threads(1)?
            .commit_from_file(model_dir.as_ref().join("joint.enc.int8.onnx"))?;

        let joint_net = Session::builder().unwrap()
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_parallel_execution(true)?
            .with_execution_providers([provider.clone()])?
            //.with_intra_threads(1)?
            //.with_inter_threads(1)?
            .commit_from_file(model_dir.as_ref().join("joint.joint_net.int8.onnx"))?;

        let preprocessor = Session::builder().unwrap()
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_execution_providers([provider.clone()])?
            .with_parallel_execution(true)?
            //.with_intra_threads(1)?
            //.with_inter_threads(1)?
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

    pub fn fbank_features(&mut self, audio: &ArrayViewD<f32>, audio_len: i64) -> Result<ArrayD<f32>, ParakeetError> {
        let inputs = ort::inputs![
            "waveforms" => TensorRef::from_array_view(audio.view())?,
            "waveforms_lens" => Tensor::from_array(([1], vec![audio_len].into_boxed_slice()))?,
        ];
        let outputs = self.preprocessor.run(inputs)?;
        let features = outputs.get("features").unwrap().try_extract_array()?;
        Ok(features.to_owned())
    }

    pub fn encoder_infer(&mut self, features: &ArrayViewD<f32>) -> Result<ArrayD<f32>, ParakeetError> {
        let input_length= features.shape()[2] as i64;
        let inputs = ort::inputs![
            "audio_signal" => TensorRef::from_array_view(features.view())?,
            "length" => Tensor::from_array(([1], vec![input_length].into_boxed_slice()))?,
        ];
        let outputs = self.encoder.run(inputs)?;
        let encoder_output = outputs.get("outputs").unwrap().try_extract_array()?;
        let encoder_output = encoder_output.permuted_axes(IxDyn(&[0, 2, 1]));
        Ok(encoder_output.to_owned())
    }

    pub fn joint_enc_infer(&mut self, encoder_output: &ArrayViewD<f32>) -> Result<ArrayD<f32>, ParakeetError> {
        let contiguous_encoder_output = encoder_output.as_standard_layout();
        let inputs = ort::inputs![
            "input" => TensorRef::from_array_view(contiguous_encoder_output.view())?
        ];
        let outputs = self.joint_enc.run(inputs)?;
        let res = outputs.get("output").unwrap().try_extract_array()?;
        Ok(res.to_owned())
    }

    pub fn get_init_decoder_state(&self) -> Result<(ArrayD<f32>, ArrayD<f32>), ParakeetError> {
        let state0 = ArrayD::zeros(IxDyn(&[2, 1, 640])); // TODO: hardcoded
        let state1 = ArrayD::zeros(IxDyn(&[2, 1, 640]));
        Ok((state0, state1))
    }

    pub fn decoder_infer(
        &mut self,
        target: i64,
        state0: &ArrayViewD<f32>,
        state1: &ArrayViewD<f32>,
    ) -> Result<(ArrayD<f32>, ArrayD<f32>, ArrayD<f32>), ParakeetError> {
        let inputs = ort::inputs![
            "targets" => Tensor::from_array(([1, 1], vec![target as i32].into_boxed_slice()))?,
            "target_length" => Tensor::from_array(([1], vec![1i32].into_boxed_slice()))?,
            "states.1" => TensorRef::from_array_view(state0.view())?,
            "onnx::Slice_3" => TensorRef::from_array_view(state1.view())?,
        ];
        let outputs = self.decoder.run(inputs)?;
        let decoder_output = outputs.get("outputs").unwrap().try_extract_array()?;
        let decoder_output = decoder_output.permuted_axes(IxDyn(&[0, 2, 1]));
        let state0_next: ArrayViewD<f32> = outputs.get("states").unwrap().try_extract_array()?;
        let state1_next: ArrayViewD<f32> = outputs.get("162").unwrap().try_extract_array()?;
        Ok((
            decoder_output.to_owned(),
            state0_next.to_owned(),
            state1_next.to_owned(),
        ))
    }

    pub fn joint_pred_infer(
        &mut self,
        decoder_output: &ArrayViewD<f32>,
    ) -> Result<ArrayD<f32>, ParakeetError> {
        let inputs = ort::inputs![
            "onnx::MatMul_0" => TensorRef::from_array_view(decoder_output.view())?,
        ];
        let outputs = self.joint_pred.run(inputs)?;
        let res = outputs.get("5").unwrap().try_extract_array()?;
        Ok(res.to_owned())
    }

    pub fn joint_net_infer(
        &mut self,
        f: &ArrayViewD<f32>,
        g: &ArrayViewD<f32>,
    ) -> Result<ArrayD<f32>, ParakeetError> {
        let ff = f.to_shape(IxDyn(&[1, 1, 1, 640]))?;
        let gg = g.to_shape(IxDyn(&[1, 1, 1, 640]))?;
        let input = ff.add(gg);
        let inputs = ort::inputs![
            "input.1" => TensorRef::from_array_view(input.view())?,
        ];
        let outputs = self.joint_net.run(inputs)?;
        let res = outputs.get("6").unwrap().try_extract_array()?;
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