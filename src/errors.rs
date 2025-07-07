use thiserror::Error;

type BoxError = Box<dyn std::error::Error + Send + Sync>;

#[derive(Error, Debug)]
pub enum ParakeetError {
    #[error("General logic error: {0}")]
    AnyError(#[from] anyhow::Error),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("NdArray error: {0}")]
    NdArrayError(#[from] ndarray::ShapeError),
    #[error("Boxed error: {0}")]
    BoxedError(#[from] BoxError),
    #[error("Ort error: {0}")]
    OrtError(#[from] ort::Error),
    #[error("Min max error: {0}")]
    MinMaxError(#[from] ndarray_stats::errors::MinMaxError),
}
