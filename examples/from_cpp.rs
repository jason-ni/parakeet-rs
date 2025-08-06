use std::path::Path;
use parakeet_rs::errors::ParakeetError;
use parakeet_rs::asr::ParakeetASR;

fn main() -> Result<(), ParakeetError> {
    env_logger::init();

    let model_dir = std::env::args().nth(1).expect("Please provide model directory as the first argument");
    let is_quantized = true;
    let has_cuda = false;
    let encoder_output_path = std::env::args().nth(2).expect("Please provide encoder output file path as the second argument");
    let encoder_output = std::fs::read(Path::new(&encoder_output_path))?;
    let f32_encoder_output = unsafe {
        std::slice::from_raw_parts(encoder_output.as_ptr() as *const f32, encoder_output.len() / 4)
    };

    let mut asr = ParakeetASR::new(&model_dir, is_quantized, has_cuda)?;

    let mut res = asr.infer_from_encoder_output(f32_encoder_output)?;
    println!("Result: {}", res.to_text());
    res.generate_word_records();
    for word in res.get_word_records() {
        println!("Word: <{}>, start: {}, end: {}", word.word, word.start_time, word.end_time);
    }

    Ok(())
}