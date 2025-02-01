use crate::layer::Layer;
use crate::tensor::Tensor;
use crate::Float;
use crate::Array2;

pub struct Model {
    pub layers: Vec<Box<dyn Layer>>,
    last_activations: Option<Vec<Tensor>>,
    pub loss_fn: fn(&Tensor, &Array2<Float>) -> Float,
    pub loss_grad_fn: fn(&Tensor, &Array2<Float>) -> Array2<Float>,
}

impl Model {
    // 新しいモデルを生成する
    pub fn new(loss_fn: fn(&Tensor, &Array2<Float>) -> Float, loss_grad_fn: fn(&Tensor, &Array2<Float>) -> Array2<Float>) -> Self {
        Self {
            layers: Vec::new(),
            last_activations: None,
            loss_fn,
            loss_grad_fn,
        }
    }

    // モデルに層を追加する
    pub fn add_layer<L: Layer + 'static>(&mut self, layer: L) {
        self.layers.push(Box::new(layer));
    }

    // 順伝播処理
    // 入力から各層の出力を記録し、最終出力を返します
    pub fn forward(&mut self, input: &Tensor) -> Tensor {
        let mut activations: Vec<Tensor> = vec![input.clone()];
        for layer in self.layers.iter_mut() {
            let a = layer.forward(activations.last().unwrap());
            activations.push(a);
        }
        self.last_activations = Some(activations);
        self.last_activations.as_ref().unwrap().last().unwrap().clone()
    }
    
    // 逆伝播処理
    // forward()で記録された中間出力を用い、損失の勾配を各層で伝播させパラメータを更新します
    pub fn backward(&mut self, target: &Array2<Float>, learning_rate: Float) {
        if let Some(activations) = self.last_activations.take() {
            let output = activations.last().unwrap();
            let mut grad = Tensor::new((self.loss_grad_fn)(output, target));
            for (layer, activation) in self.layers.iter_mut().rev().zip(activations.iter().rev().skip(1)) {
                grad = layer.backward(activation, &grad);
                layer.update_parameters(learning_rate.into());
            }
        }
    }
}
