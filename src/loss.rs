use ndarray::Array2;
use crate::Float;
use crate::tensor::Tensor;

/// MSE（平均二乗誤差）損失関数の計算
/// loss = 0.5 * sum((output - target)^2)
pub fn mse_loss(output: &Tensor, target: &Array2<Float>) -> Float {
    let diff = &output.data - target;
    0.5 * diff.mapv(|x| x.powi(2)).sum()
}

/// MSE損失に対する出力の勾配計算
/// dL/dy = (output - target)
pub fn mse_loss_grad(output: &Tensor, target: &Array2<Float>) -> Array2<Float> {
    &output.data - target
}
