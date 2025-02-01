extern crate deep_learning_rust;
use deep_learning_rust::{array, Array2, Float};
use deep_learning_rust::tensor::Tensor;
use deep_learning_rust::layer::{Layer, Dense};
use deep_learning_rust::loss;
use deep_learning_rust::model::Model;

fn main() {
    // 入力値と正解値を定義
    let input_data: Array2<Float> = array![[1.0, 2.0, 3.0]];
    let target_data: Array2<Float> = array![[0.0, 1.0]];
    
    // 入力値をテンソルに変換
    let input: Tensor = Tensor::new(input_data);
    
    // モデルの作成と層の追加
    let mut model = Model::new();
    let dense: Dense = Dense::new(3, 2);
    model.add_layer(dense);
    
    let learning_rate: f32 = 0.0005;
    let epochs = 1000;
    
    for epoch in 0..epochs {
        // 順伝播を実行
        let output = model.forward(&input);
        
        // MSE 損失の計算
        let loss_val = loss::mse_loss(&output, &target_data);

        // 逆伝播とパラメータ更新
        let grad_output_data = loss::mse_loss_grad(&output, &target_data);
        let grad_output = Tensor::new(grad_output_data);
        for layer in model.layers.iter_mut() {
            let _ = layer.backward(&input, &grad_output);
            layer.update_parameters(learning_rate);
        }
        
        // 定期的に loss を表示
        if epoch % 100 == 0 {
            println!("Epoch {}: Loss = {:.6}", epoch, loss_val);
        }
    }
    
    // 最終出力の表示
    let final_output = model.forward(&input);
    println!("Final output (Model):");
    println!("{:?}", final_output.data);
}
