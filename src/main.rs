extern crate deep_learning_rust;
use deep_learning_rust::{array, Array2, Float};
use deep_learning_rust::tensor::Tensor;
use deep_learning_rust::layer::{Dense, ReLU, Dropout};
use deep_learning_rust::model::Model;
use deep_learning_rust::loss;

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
    
    // テンソルに変換
    let input: Tensor = Tensor::new(input_data);
    
    // モデルの定義
    let mut model = Model::new(loss::binary_cross_entropy_loss);
    model.add_layer(Dense::new(3, 10));
    model.add_layer(Dropout::new(0.15));
    model.add_layer(ReLU::new());
    model.add_layer(Dense::new(10, 5));
    model.add_layer(Dropout::new(0.15));
    model.add_layer(ReLU::new());
    model.add_layer(Dense::new(5, 2));
    
    let learning_rate: Float = 0.0005;
    let epochs = 1000;
    
    for epoch in 0..epochs {
        // 順伝播処理
        let output = model.forward(&input);
        // 損失の計算
        let (loss_val, _) = (model.loss_fn)(&output, &target_data);
        // 逆伝播処理とパラメータ更新
        model.backward(&target_data, learning_rate);
        
        if epoch % 100 == 0 {
            println!("Epoch {}: Loss = {:.6}", epoch, loss_val);
        }
    }
}
