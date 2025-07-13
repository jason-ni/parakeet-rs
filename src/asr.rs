use ndarray::{ArrayViewD, IxDyn, s};
use ndarray_stats::QuantileExt;
use crate::model::{ParakeetModel, ParakeetTokenizer};
use crate::errors::ParakeetError;

pub struct ParakeetASR {
    tokenizer: ParakeetTokenizer,
    model: ParakeetModel,
}

impl ParakeetASR {
    pub fn new(model_path: &str, is_quantized: bool, has_cuda: bool) -> Result<Self, ParakeetError> {
        let tokenizer = ParakeetTokenizer::new(model_path)?;
        let model = ParakeetModel::new(model_path, is_quantized, has_cuda)?;
        Ok(Self { tokenizer, model })
    }

    pub fn infer_buffer(&mut self, buffer: &[f32]) -> Result<ASRResult, ParakeetError> {
        let shape = vec![1, buffer.len()];
        let audio: ArrayViewD<f32> = ArrayViewD::from_shape(
            IxDyn(&shape),
            buffer,
        )?;

        let features = self.model.fbank_features(&audio, audio.len() as i64)?;

        let blank_id = self.tokenizer.blank_id();

        //let num_duration = 5;

        let encoder_output = self.model.encoder_infer(&features.view())?;

        let encoder_output_length = encoder_output.shape()[1];

        let encoder_output_projected = self.model.joint_enc_infer(&encoder_output.view())?;

        let (mut state0, mut state1) = self.model.get_init_decoder_state()?;

        let mut time_index = 0usize;
        let mut safe_time_index = 0;
        let mut time_index_current_labels;
        let last_timestamps = encoder_output_length - 1;

        let mut active = encoder_output_length > 0;
        let mut advance = false;
        let mut label = blank_id;
        let mut score;
        let mut duration;

        let mut asr_result = ASRResult::new();

        while active {
            // state 1: get decoder (prediction network) output
            let (mut decoder_output, state0_next, state1_next) = self.model.decoder_infer(
                label,
                &state0.view(),
                &state1.view(),
            )?;
            decoder_output = self.model.joint_pred_infer(&decoder_output.view())?;
            let f_slice = encoder_output_projected.slice(s![0, safe_time_index, ..]);
            let f =  f_slice.to_shape(IxDyn(&[1, 1, 640]))?;
            let logits = self.model.joint_net_infer(
                &f.view(), &decoder_output.view())?;
            //log::trace!("logits shape: {:?}", logits.shape());
            //log::trace!("logits: {}", logits);

            score = logits.slice(s![0, 0, 0, ..1025]).max()?.to_owned();
            label = match logits.slice(s![0, 0, 0, ..1025]).argmax() {
                Ok(label) => label as i64,
                Err(e) => return Err(ParakeetError::from(e)),
            };
            log::trace!("score: {}, label: {}, token: {}, safe_time_index: {}",
                score, label, self.tokenizer.decode(label), safe_time_index);

            let jump_duration_index = match logits.slice(s![0, 0, 0, 1025..]).argmax() {
                Ok(jump_duration_index) => jump_duration_index as i64,
                Err(e) => return Err(ParakeetError::from(e)),
            };
            log::trace!("duration: {}", jump_duration_index);
            duration = jump_duration_index as i32;
            time_index_current_labels = time_index;

            if duration == 0 {
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

                //log::trace!("safe_time_index: {}", safe_time_index);

                let f_slice = encoder_output_projected.slice(s![0, safe_time_index, ..]);
                let f = f_slice.to_shape(IxDyn(&[1, 1, 640]))?;
                let logits = self.model.joint_net_infer(
                    &f.view(), &decoder_output.view())?;
                //log::trace!("advance logits shape: {:?}", logits.shape());
                //log::trace!("advance logits: {}", logits);
                score = logits.slice(s![0, 0, 0, ..1025]).max()?.to_owned();
                label = match logits.slice(s![0, 0, 0, ..1025]).argmax() {
                    Ok(label) => label as i64,
                    Err(e) => return Err(ParakeetError::from(e)),
                };
                log::trace!("advance score: {}, label: {}, token: {}",
                    score, label, self.tokenizer.decode(label));

                let jump_duration_index = match logits.slice(s![0, 0, 0, 1025..]).argmax() {
                    Ok(jump_duration_index) => jump_duration_index as i64,
                    Err(e) => return Err(ParakeetError::from(e)),
                };
                //log::trace!("advance jump_duration_index: {}", jump_duration_index);
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
                //log::trace!("active: {}, advance: {}, safe_time_index: {}, time_index_current_labels: {}",
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

#[derive(Debug)]
pub struct ASRResult {
    token_records: Vec<TokenRecognitionRecord>,
    word_records: Vec<WordRecognitionRecord>,
}

impl ASRResult {

    pub fn new() -> Self {
        Self {
            token_records: vec![],
            word_records: vec![],
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
        let mut in_abbr_state = false;
        let token_len = self.token_records.len();
        for (idx, token_record) in self.token_records.iter().enumerate() {
            let is_punctuation = EN_PUNCTUATION.contains(&token_record.token);
            let is_period = token_record.token == ".";
            let is_starting_token = token_record.token.starts_with(TOKEN_PREFIX);

            if is_period && idx < (token_len - 1) && !self.token_records[idx+1].token.starts_with(TOKEN_PREFIX) {
                in_abbr_state = true;
            }

            //println!("is_starting_token: {}, is_punctuation: {}, is_period: {}, in_abbr_state: {}, token: {}",
            //    is_starting_token, is_punctuation, is_period, in_abbr_state, token_record.token);
            if is_starting_token ||
                (is_punctuation && !(is_period && in_abbr_state))
            {
                if !word_buffer.is_empty() {
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
                in_abbr_state = false;
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