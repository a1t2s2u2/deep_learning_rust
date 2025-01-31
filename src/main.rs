use deep_learning_rust::tensor::Tensor;
use deep_learning_rust::layer::{Dense, ReLU, Layer};

fn main() {
    // 入力（例: バッチサイズ5、特徴量次元2）
    let input = Tensor::random(5, 2);

    // Dense層 (入力次元2, 出力次元4)
    let mut dense1 = Dense::new(2, 4);

    // ReLU層
    let mut relu = ReLU::new();

    // フォワードパス
    let out_dense = dense1.forward(&input);
    let out_relu = relu.forward(&out_dense);

    // 出力の表示
    println!("Output from ReLU layer: {:?}", out_relu.data);

    // 逆伝播用にダミーの勾配を用意（出力と同サイズ）
    let grad_output = Tensor::random(out_relu.shape().0, out_relu.shape().1);

    // 逆伝播を実行
    let grad_relu = relu.backward(&out_dense, &grad_output);
    let grad_dense = dense1.backward(&input, &grad_relu);

    println!("Gradient back-propagated to input: {:?}", grad_dense.data);
}
