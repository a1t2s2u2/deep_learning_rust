extern crate deep_learning_rust;
use deep_learning_rust::{array, Array2, Float};
use deep_learning_rust::tensor::Tensor;
use deep_learning_rust::layer::{Layer, Dense};
use deep_learning_rust::loss;

fn main() {
    // 入力値と正解値を定義
    let input_data: Array2<Float> = array![[1.0, 2.0, 3.0]];
    let target_data: Array2<Float> = array![[0.0, 1.0]];
    
    // 入力値をテンソルに変換
    let input: Tensor = Tensor::new(input_data);
    
    // Dense レイヤーの作成 (入力次元3, 出力次元2)
    let mut dense: Dense = Dense::new(3, 2);
    
    let learning_rate: f32 = 0.001;
    let epochs = 1000;
    
    for epoch in 0..epochs {
        // 順伝播を実行
        let output = Layer::forward(&mut dense, &input);
        
        // MSE 損失の計算
        let loss_val = loss::mse_loss(&output, &target_data);
        
        // 出力に対する勾配の計算
        let grad_output_data = loss::mse_loss_grad(&output, &target_data);
        let grad_output = Tensor::new(grad_output_data);
        
        // 逆伝播を実行
        let _ = Layer::backward(&mut dense, &input, &grad_output);
        
        // パラメータ更新
        if let Some(ref grad_w) = dense.weights.grad {
            dense.weights.data = &dense.weights.data - &(grad_w * learning_rate);
        }
        if let Some(ref grad_b) = dense.bias.grad {
            dense.bias.data = &dense.bias.data - &(grad_b * learning_rate);
        }
        
        // 定期的に loss を表示
        if epoch % 100 == 0 {
            println!("Epoch {}: Loss = {:.6}", epoch, loss_val);
        }
    }
    
    // 最終出力の表示
    let final_output = Layer::forward(&mut dense, &input);
    println!("Final output (Dense layer):");
    println!("{:?}", final_output.data);
}
