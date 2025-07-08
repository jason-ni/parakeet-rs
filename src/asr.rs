use anyhow::Context;
use ndarray::{ArrayD, ArrayViewD, IxDyn, s, Axis};
use ndarray_stats::QuantileExt;
use crate::model::{ParakeetModel, ParakeetTokenizer};
use crate::errors::ParakeetError;

pub struct ParakeetASR {
    tokenizer: ParakeetTokenizer,
    model: ParakeetModel,
}

impl ParakeetASR {
    pub fn new(model_path: &str) -> Result<Self, ParakeetError> {
        let tokenizer = ParakeetTokenizer::new(model_path)?;
        let model = ParakeetModel::new(model_path)?;
        Ok(Self { tokenizer, model })
    }

    pub fn infer_buffer(&self, buffer: &[f32]) -> Result<ASRResult, ParakeetError> {
        let shape = vec![1, buffer.len()];
        let audio: ArrayViewD<f32> = ArrayViewD::from_shape(
            IxDyn(&shape),
            buffer,
        )?;

        let features = self.model.fbank_features(&audio, audio.len() as i64)?;

        let blank_id = self.tokenizer.blank_id();

        //let num_duration = 5;
        //let max_symbols = 10;

        let encoder_output = self.model.encoder_infer(&features)?;

        let encoder_output_length = encoder_output.shape()[1];

        let encoder_output_projected = self.model.joint_enc_infer(&encoder_output)?;

        let (mut last_state0, mut last_state1) = self.model.get_init_decoder_state()?;
        let (mut state0, mut state1) = self.model.get_init_decoder_state()?;
        let mut labels = ndarray::Array1::from_shape_fn(1, |_| blank_id);

        let mut time_index = 0usize;
        let mut safe_time_index = 0;
        let mut time_index_current_labels = 0;
        let mut last_timestamps = encoder_output_length - 1;

        let mut active = encoder_output_length > 0;
        let mut advance = false;
        let mut active_prev = false;
        let mut label = blank_id;
        let mut score = 0.0;
        let mut duration = 0;

        let mut asr_result = ASRResult::new();

        while active {
            active_prev = active;

            // state 1: get decoder (prediction network) output
            let (mut decoder_output, state0_next, state1_next) = self.model.decoder_infer(
                label,
                &state0,
                &state1,
            )?;
            decoder_output = self.model.joint_pred_infer(&decoder_output)?;
            let f: ArrayD<f32> = encoder_output_projected.slice(s![0, safe_time_index, ..])
                .to_shape(IxDyn(&[1, 1, 640]))?
                .to_owned();
            let logits = self.model.joint_net_infer(
                &f, &decoder_output)?;
            //log::info!("logits shape: {:?}", logits.shape());
            //log::info!("logits: {}", logits);

            score = logits.slice(s![0, 0, 0, ..1025]).max()?.to_owned();
            label = match logits.slice(s![0, 0, 0, ..1025]).argmax() {
                Ok(label) => label as i64,
                Err(e) => return Err(ParakeetError::from(e)),
            };
            //log::info!("score: {}, label: {}, token: {}, safe_time_index: {}",
            //    score, label, self.tokenizer.decode(label), safe_time_index);

            let jump_duration_index = match logits.slice(s![0, 0, 0, 1025..]).argmax() {
                Ok(jump_duration_index) => jump_duration_index as i64,
                Err(e) => return Err(ParakeetError::from(e)),
            };
            //log::info!("duration: {}", jump_duration_index);
            duration = jump_duration_index as i32;
            time_index_current_labels = time_index;

            if label == blank_id && duration == 0 {
                duration = 1;
            }

            time_index += duration as usize;
            safe_time_index = time_index.min(last_timestamps);
            if time_index >= last_timestamps {
                active = false;
            }
            if active && label == blank_id {
                advance = true;
            }

            while advance {
                time_index_current_labels = time_index;

                //log::info!("safe_time_index: {}", safe_time_index);

                let f: ArrayD<f32> = encoder_output_projected.slice(s![0, safe_time_index, ..])
                    .to_shape(IxDyn(&[1, 1, 640]))?
                    .to_owned();
                let logits = self.model.joint_net_infer(
                    &f, &decoder_output)?;
                //log::info!("advance logits shape: {:?}", logits.shape());
                //log::info!("advance logits: {}", logits);
                score = logits.slice(s![0, 0, 0, ..1025]).max()?.to_owned();
                label = match logits.slice(s![0, 0, 0, ..1025]).argmax() {
                    Ok(label) => label as i64,
                    Err(e) => return Err(ParakeetError::from(e)),
                };
                //log::info!("advance score: {}, label: {}, token: {}",
                //    score, label, self.tokenizer.decode(label));

                let jump_duration_index = match logits.slice(s![0, 0, 0, 1025..]).argmax() {
                    Ok(jump_duration_index) => jump_duration_index as i64,
                    Err(e) => return Err(ParakeetError::from(e)),
                };
                //log::info!("advance jump_duration_index: {}", jump_duration_index);
                duration = jump_duration_index as i32;

                if label == blank_id && duration == 0 {
                    duration = 1;
                }
                time_index += duration as usize;
                safe_time_index = time_index.min(last_timestamps);
                if time_index >= last_timestamps {
                    active = false;
                }
                if active && label == blank_id {
                    advance = true;
                } else {
                    advance = false;
                }
                //log::info!("active: {}, advance: {}, safe_time_index: {}, time_index_current_labels: {}",
                //    active, advance, safe_time_index, time_index_current_labels);
            }
            state0 = state0_next;
            state1 = state1_next;
            if label != blank_id {
                asr_result.add_token_record(
                    self.tokenizer.decode(label).to_string(),
                    duration,
                    time_index_current_labels as i32,
                    score,
                );
            }
        }


        Ok(asr_result)
    }
}

#[derive(Debug, Clone)]
pub struct TokenRecognitionRecord {
    pub token: String,
    pub duration: i32,
    pub time_index: i32,
    pub score: f32,
}

#[derive(Debug, Clone)]
pub struct WordRecognitionRecord {
    pub word: String,
    pub start_time: f32,
    pub end_time: f32,
}

#[derive(Debug, Clone)]
pub struct SegmentRecognitionRecord {
    pub text: String,
    pub start_time: f32,
    pub end_time: f32,
}

const TOKEN_PREFIX: &str = "▁";
const ASR_WINDOW: f32 = 0.08;
const EN_PUNCTUATION: &str = ".,;:!?";
const EN_SENTENCE_END: &str = ".!?";

#[derive(Debug)]
pub struct ASRResult {
    token_records: Vec<TokenRecognitionRecord>,
    word_records: Vec<WordRecognitionRecord>,
    segment_records: Vec<SegmentRecognitionRecord>,
}

impl ASRResult {

    pub fn new() -> Self {
        Self {
            token_records: vec![],
            word_records: vec![],
            segment_records: vec![],
        }
    }

    pub fn add_token_record(&mut self, token: String, duration: i32, time_index: i32, score: f32) {
        self.token_records.push(TokenRecognitionRecord {
            token,
            duration,
            time_index,
            score,
        });
    }

    pub fn to_text(&self) -> String {
        let mut text = String::new();
        for token_record in &self.token_records {
            text.push_str(&token_record.token);
        }
        text.replace(TOKEN_PREFIX, " ").trim().to_string()
    }

    pub fn generate_word_records(&mut self) {
        let mut word_start_time = 0.0;
        let mut word_buffer = String::new();
        let mut word_duration = 0.0;
        for token_record in &self.token_records {
            let is_punctuation = EN_PUNCTUATION.contains(&token_record.token);
            if token_record.token.starts_with(TOKEN_PREFIX) || is_punctuation {
                if!word_buffer.is_empty() {
                    self.word_records.push(WordRecognitionRecord {
                        word: word_buffer.clone(),
                        start_time: word_start_time,
                        end_time: word_start_time + word_duration,
                    });
                    word_buffer.clear();
                    word_duration = 0.0;
                }
                word_buffer.push_str(&token_record.token.replace(TOKEN_PREFIX, " "));
                word_start_time = token_record.time_index as f32 * ASR_WINDOW;
            } else {
                word_buffer.push_str(&token_record.token);
            }
            word_duration += token_record.duration as f32 * ASR_WINDOW;
        }
        if!word_buffer.is_empty() {
            self.word_records.push(WordRecognitionRecord {
                word: word_buffer.clone(),
                start_time: word_start_time,
                end_time: word_start_time + word_duration,
            });
        }
    }

    pub fn get_word_records(&self) -> &Vec<WordRecognitionRecord> {
        &self.word_records
    }

}