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

/// バイナリクロスエントロピー損失関数と勾配の計算
/// 損失: loss = sum(target * ln(output) + (1 - target) * ln(1 - output)) * -1
/// 勾配: grad = output - target
pub fn binary_cross_entropy_loss(output: &Tensor, target: &Array2<Float>) -> (Float, Array2<Float>) {
    // 数値的安定性のため、log計算時に微小値を加える
    let epsilon = 1e-12;
    let clipped_p = output.data.mapv(|x| x.clamp(epsilon, 1.0 - epsilon));

    // 損失の計算: -sum(target * ln(p) + (1 - target) * ln(1 - p))
    let loss = -target
        .iter()
        .zip(clipped_p.iter())
        .fold(0.0, |acc, (&t, &p)| {
            acc + t * p.ln() + (1.0 - t) * (1.0 - p).ln()
        });

    // 勾配の計算: p - target
    let grad = &output.data - target;

    (loss, grad)
}