extern crate deep_learning_rust;
use deep_learning_rust::{array, Array2, Float};
use deep_learning_rust::tensor::Tensor;
use deep_learning_rust::layer::{Layer, Dense, ReLU};
use deep_learning_rust::loss;
use deep_learning_rust::model::Model;

fn main() {
    // 入力値と正解値を定義
    let input_data: Array2<Float> = array![
        [0.5, -1.2, 3.8],
        [1.0,  0.0, -2.0],
        [2.2, -0.5, 1.5]
    ];
    let target_data: Array2<Float> = array![
        [1.0, 0.5],
        [0.0, 1.0],
        [0.8, 0.3]
    ];
    
    // 入力値をテンソルに変換
    let input: Tensor = Tensor::new(input_data);
    
    // モデルの作成と層の追加（複数の層）
    let mut model = Model::new();
    // 第1層: 入力次元3 -> 隠れ層次元4
    model.add_layer(Dense::new(3, 4));
    // 第2層: 活性化関数ReLU
    model.add_layer(ReLU::new());
    // 第3層: 隠れ層次元4 -> 出力次元2
    model.add_layer(Dense::new(4, 2));
    
    let learning_rate: f32 = 0.001;
    let epochs = 1000;
    
    for epoch in 0..epochs {
        // 順伝播：各層の出力を保持
        let mut activations: Vec<Tensor> = vec![input.clone()];
        for layer in model.layers.iter_mut() {
            let act = layer.forward(activations.last().unwrap());
            activations.push(act);
        }
        let output = activations.last().unwrap();
        
        // MSE 損失の計算
        let loss_val = loss::mse_loss(output, &target_data);
        
        // 逆伝播：最終出力から開始
        let mut grad = Tensor::new(loss::mse_loss_grad(output, &target_data));
        // 各層を逆順にたどって逆伝播を実行
        for (layer, activation) in model.layers.iter_mut().rev().zip(activations.iter().rev().skip(1)) {
            grad = layer.backward(activation, &grad);
            layer.update_parameters(learning_rate);
        }
        
        // 定期的に損失を表示
        if epoch % 100 == 0 {
            println!("Epoch {}: Loss = {:.6}", epoch, loss_val);
        }
    }
    
    // 最終出力の表示（順伝播のみ）
    let mut final_activation = input.clone();
    for layer in model.layers.iter_mut() {
        final_activation = layer.forward(&final_activation);
    }
    println!("Final output (Model):");
    println!("{:?}", final_activation.data);
}
