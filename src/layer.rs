use crate::tensor::{Float, Tensor};

pub trait Layer {
    fn forward(&mut self, input: &Tensor) -> Tensor;
    fn backward(&mut self, input: &Tensor, grad_output: &Tensor) -> Tensor;
}

pub struct Dense {
    pub weights: Tensor,
    pub bias: Tensor,
}

impl Dense {
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        Self {
            weights: Tensor::random(input_dim, output_dim),
            bias: Tensor::zeros(1, output_dim),
        }
    }
}

impl Layer for Dense {
    fn forward(&mut self, input: &Tensor) -> Tensor {
        let z = input.data.dot(&self.weights.data) + &self.bias.data;
        Tensor::new(z)
    }

    fn backward(&mut self, input: &Tensor, grad_output: &Tensor) -> Tensor {
        let grad_weights = input.data.t().dot(&grad_output.data);
        let grad_bias = grad_output.data.sum_axis(ndarray::Axis(0)).insert_axis(ndarray::Axis(0));
        self.weights.grad = Some(grad_weights.clone());
        self.bias.grad = Some(grad_bias.clone());
        let grad_input = grad_output.data.dot(&self.weights.data.t());
        Tensor::new(grad_input)
    }
}

pub struct ReLU;

impl ReLU {
    pub fn new() -> Self {
        Self
    }
}

impl Layer for ReLU {
    fn forward(&mut self, input: &Tensor) -> Tensor {
        let data = input.data.mapv(|x| if x > 0.0 as Float { x } else { 0.0 as Float });
        Tensor::new(data)
    }

    fn backward(&mut self, input: &Tensor, grad_output: &Tensor) -> Tensor {
        let grad = input.data.mapv(|x| if x > 0.0 as Float { 1.0 as Float } else { 0.0 as Float });
        Tensor::new(&grad * &grad_output.data)
    }
}
