mod model;
 
use crate::model::xFeatModelConifg;
use burn::Tensor;
use burn::tensor::Shape;
use burn::backend::Wgpu;
 
fn main() {
    println!("Hello, world!");
    type bk = Wgpu<f32, i32>;
 
    let device = Default::default();
 
    let model = xFeatModelConifg::new().init::<bk>(&device);
 
    let input = Tensor::<bk, 4>::zeros(Shape::new([1, 1, 584, 584]), &device);
    println!("my model {model}");
 
    let out1 = model.forward(input);
 
    println!("out1 : {}", out1.shape());
}