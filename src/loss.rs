use ndarray::Array2;
use crate::Float;
use crate::tensor::Tensor;

/// MSE（平均二乗誤差）損失関数と勾配の計算
/// 損失: loss = 0.5 * sum((output - target)^2)
/// 勾配: grad = output - target
pub fn mse_loss(output: &Tensor, target: &Array2<Float>) -> (Float, Array2<Float>) {
    let diff = &output.data - target;
    let loss = 0.5 * diff.mapv(|x| x.powi(2)).sum();
    (loss, diff)
}
