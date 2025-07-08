use std::path::Path;
use parakeet_rs::errors::ParakeetError;
use parakeet_rs::asr::ParakeetASR;
use hound::WavReader;

fn load_audio<P: AsRef<Path>>(audio_path: P) -> Vec<f32> {
    let mut reader = WavReader::open(audio_path).unwrap();
    let spec = reader.spec();
    log::info!("Audio format: {} channels, {} Hz", spec.channels, spec.sample_rate);
    let mut audio = vec![];
    for sample in reader.samples::<i16>() {
        audio.push(sample.unwrap() as f32 / i16::MAX as f32);
    }
    audio
}

fn main() -> Result<(), ParakeetError> {
    env_logger::init();

    let model_dir = "/Users/jason/model/asr/parakeet-tdt-0.6b-v2";

    let asr = ParakeetASR::new(model_dir, true)?;

    let audio_path = Path::new("/Users/jason/prj/parakeet").join("long.wav");
    let audio = load_audio(audio_path);
    let mut res = asr.infer_buffer(&audio)?;
    println!("Result: {}", res.to_text());
    res.generate_word_records();
    for word in res.get_word_records() {
        println!("Word: <{}>, start: {}, end: {}", word.word, word.start_time, word.end_time);
    }

    Ok(())
}