use ndarray::{array, Array2};
use deep_learning_rust::tensor::Tensor;
use deep_learning_rust::layer::{Layer, Dense};
use deep_learning_rust::loss;

fn main() {
    // ダミー入力テンソルの作成（1行3列の行列）
    let input_data: Array2<f32> = array![[1.0, 2.0, 3.0]];
    let input: Tensor = Tensor::new(input_data);
    
    // ダミー正解出力（1行2列の行列、例として [0.0, 1.0] を目標値とする）
    let target_data: Array2<f32> = array![[0.0, 1.0]];
    
    // Denseレイヤーの作成 (入力次元3, 出力次元2)
    let mut dense: Dense = Dense::new(3, 2);
    
    let learning_rate: f32 = 0.001;
    let epochs = 1000;
    
    for epoch in 0..epochs {
        // 順伝播を実行
        let output = Layer::forward(&mut dense, &input);
        
        // MSE損失の計算
        let loss = loss::mse_loss(&output, &target_data);
        
        // 出力に対する勾配の計算
        let grad_output_data = loss::mse_loss_grad(&output, &target_data);
        let grad_output = Tensor::new(grad_output_data);
        
        // 逆伝播を実行し、勾配を内部に保存（または次層へ伝播）
        let _ = Layer::backward(&mut dense, &input, &grad_output);
        
        // 重みとバイアスの更新
        if let Some(ref grad_w) = dense.weights.grad {
            dense.weights.data = &dense.weights.data - &(grad_w * learning_rate);
        }
        if let Some(ref grad_b) = dense.bias.grad {
            dense.bias.data = &dense.bias.data - &(grad_b * learning_rate);
        }
        
        // 定期的にlossを表示
        if epoch % 100 == 0 {
            println!("Epoch {}: Loss = {:.6}", epoch, loss);
        }
    }
    
    // 最終出力の表示
    let final_output = Layer::forward(&mut dense, &input);
    println!("最終出力 (Denseレイヤー):");
    println!("{:?}", final_output.data);
}
