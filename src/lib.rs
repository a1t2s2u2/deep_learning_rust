pub type Float = f64;
pub mod tensor;
pub mod layer;
pub mod loss;
pub mod model;
pub use ndarray::{array, Array2};
pub use tensor::Tensor;
