pub type Float = f32;

use ndarray::Array2;
use rand::Rng;

pub struct Tensor {
    pub data: Array2<Float>,
}

impl Tensor {
    pub fn new(data: Array2<Float>) -> Self {
        Self { data }
    }

    pub fn shape(&self) -> (usize, usize) {
        let s = self.data.shape();
        (s[0], s[1])
    }

    pub fn random(rows: usize, cols: usize) -> Self {
        let mut rng = rand::thread_rng();
        let arr = Array2::from_shape_fn((rows, cols), |_| {
            rng.gen_range(0.0 as Float..1.0 as Float)
        });
        Tensor { data: arr }
    }

    pub fn zeros(rows: usize, cols: usize) -> Self {
        Tensor {
            data: Array2::zeros((rows, cols)),
        }
    }
}
