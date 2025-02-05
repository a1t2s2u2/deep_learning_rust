use std::any::Any;
use crate::Float;
use crate::tensor::Tensor;

pub trait Layer {
    fn forward(&mut self, input: &Tensor) -> Tensor;
    fn backward(&mut self, input: &Tensor, grad_output: &Tensor) -> Tensor;
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn update_parameters(&mut self, _learning_rate: Float) {}
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
    
    fn update_parameters(&mut self, learning_rate: Float) {
        if let Some(ref grad_w) = self.weights.grad {
            self.weights.data = &self.weights.data - &(grad_w * learning_rate);
        }
        if let Some(ref grad_b) = self.bias.grad {
            self.bias.data = &self.bias.data - &(grad_b * learning_rate);
        }
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
    
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

pub struct Dropout {
    pub prob: Float,
    mask: Option<Tensor>,
}

impl Dropout {
    pub fn new(prob: Float) -> Self {
        Self {
            prob,
            mask: None,
        }
    }
}

impl Layer for Dropout {
    fn forward(&mut self, input: &Tensor) -> Tensor {
        let keep_prob: f64 = 1.0 - self.prob;
        let (rows, cols) = input.data.dim();
        let rand_tensor: Tensor = Tensor::random(rows, cols);
        let mask_data = rand_tensor.data.mapv(|x| {
            if x < keep_prob {
                1.0 / keep_prob
            } else {
                0.0
            }
        });
        self.mask = Some(Tensor::new(mask_data.clone()));
        Tensor::new(&input.data * &mask_data)
    }

    fn backward(&mut self, _input: &Tensor, grad_output: &Tensor) -> Tensor {
        if let Some(ref mask) = self.mask {
            Tensor::new(&grad_output.data * &mask.data)
        } else {
            panic!("Dropout mask not set during forward pass");
        }
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
    
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
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
    
    fn as_any(&self) -> &dyn Any {
        self
    }
    
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

pub struct Sigmoid;

impl Sigmoid {
    pub fn new() -> Self {
        Self
    }
}

impl Layer for Sigmoid {
    fn forward(&mut self, input: &Tensor) -> Tensor {
        let data = input.data.mapv(|x| 1.0 / (1.0 + (-x).exp()));
        Tensor::new(data)
    }

    fn backward(&mut self, input: &Tensor, grad_output: &Tensor) -> Tensor {
        let sigmoid = input.data.mapv(|x| 1.0 / (1.0 + (-x).exp()));
        let grad = sigmoid.mapv(|s| s * (1.0 - s));
        Tensor::new(&grad * &grad_output.data)
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
    
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

pub struct Tanh;

impl Tanh {
    pub fn new() -> Self {
        Self
    }
}

impl Layer for Tanh {
    fn forward(&mut self, input: &Tensor) -> Tensor {
        let data = input.data.mapv(|x| x.tanh());
        Tensor::new(data)
    }

    fn backward(&mut self, input: &Tensor, grad_output: &Tensor) -> Tensor {
        let tanh_val = input.data.mapv(|x| x.tanh());
        let grad = tanh_val.mapv(|t| 1.0 - t * t);
        Tensor::new(&grad * &grad_output.data)
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
    
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

pub struct Softmax;

impl Softmax {
    pub fn new() -> Self {
        Self
    }
}

impl Layer for Softmax {
    fn forward(&mut self, input: &Tensor) -> Tensor {
        // Numerical stability: subtract the row-wise max
        let max_per_row = input.data.map_axis(ndarray::Axis(1), |row| row.fold(std::f64::NEG_INFINITY, |a, &b| a.max(b)));
        let max_broadcast = max_per_row.insert_axis(ndarray::Axis(1));
        let shifted = &input.data - &max_broadcast;
        let exp_data = shifted.mapv(|x| x.exp());
        let sum_per_row = exp_data.sum_axis(ndarray::Axis(1));
        let sum_broadcast = sum_per_row.insert_axis(ndarray::Axis(1));
        let softmax = &exp_data / &sum_broadcast;
        Tensor::new(softmax)
    }

    fn backward(&mut self, input: &Tensor, grad_output: &Tensor) -> Tensor {
        // Compute softmax output first for the given input
        let max_per_row = input.data.map_axis(ndarray::Axis(1), |row| row.fold(std::f64::NEG_INFINITY, |a, &b| a.max(b)));
        let max_broadcast = max_per_row.insert_axis(ndarray::Axis(1));
        let shifted = &input.data - &max_broadcast;
        let exp_data = shifted.mapv(|x| x.exp());
        let sum_per_row = exp_data.sum_axis(ndarray::Axis(1));
        let sum_broadcast = sum_per_row.insert_axis(ndarray::Axis(1));
        let s = &exp_data / &sum_broadcast;

        // For each row, calculate: grad_input = s * (grad_output - sum(s * grad_output))
        let dot = (&s * &grad_output.data).sum_axis(ndarray::Axis(1));
        let dot_broadcast = dot.insert_axis(ndarray::Axis(1));
        let grad_input = &s * (&grad_output.data - &dot_broadcast);
        Tensor::new(grad_input)
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
    
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
