use crate::layer::Layer;
use crate::tensor::Tensor;

pub struct Model {
    pub layers: Vec<Box<dyn Layer>>,
}

impl Model {
    // 新しいモデルを生成する
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
        }
    }

    // モデルに層を追加する
    pub fn add_layer<L: Layer + 'static>(&mut self, layer: L) {
        self.layers.push(Box::new(layer));
    }

    // 順伝播処理（各層を通して処理を進める）
    // 入力は参照で受け取り、内部でcloneして計算します
    pub fn forward(&mut self, input: &Tensor) -> Tensor {
        let mut x: Tensor = input.clone();
        for layer in self.layers.iter_mut() {
            x = layer.forward(&x);
        }
        x
    }
}
