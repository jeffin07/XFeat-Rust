use burn::Tensor;
use burn::module::Module;
use burn::config::Config;
use burn::tensor::backend::Backend;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{BatchNorm,BatchNormConfig, Relu};
 
// BasicLayer
#[derive(Module,Debug)]
pub struct BasicLayer<B: Backend>{
 
    conv1: Conv2d<B>,
    batchnorm: BatchNorm<B>,
    relu: Relu
}
 
impl<B: Backend> BasicLayer<B>{
 
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B,4>{
        let [dim, height, width] = input.dims();
        let x  = input.reshape([dim, 1, height, width]);
 
        let x = self.conv1.forward(x);
        let x = self.batchnorm.forward(x);
        let x = self.relu.forward(x);
 
        return x
    }
}
 
 
#[derive(Config,Debug)]
pub struct BasicLayerConfig{
    in_channel: usize,
    out_channel: usize,
    kernel_size: usize,
    stride: usize,
}
 
impl BasicLayerConfig{
 
    pub fn init<B: Backend>(&self, device: &B::Device) ->BasicLayer<B>{
        BasicLayer{
            conv1: Conv2dConfig::new([self.in_channel, self.out_channel], [self.kernel_size, self.kernel_size]).with_stride([self.stride, self.stride]).init(device),
            batchnorm: BatchNormConfig::new(self.out_channel).init(device),
            relu: Relu::new()
        }
    }
}
 
#[derive(Module, Debug)]
pub struct BasicBlock<B: Backend>{
    block: Vec<BasicLayer<B>>,
}

impl<B: Backend> BasicBlock<B>{

    pub fn new<const N:usize>(conf: [(usize, usize, usize, usize); N], device: &B::Device)-> Self {

        let mut blocks_: Vec<BasicLayer<B>> = Vec::new();
        for i in (0..N){
            blocks_.push(BasicLayerConfig::new(conf[i].0, conf[i].1, conf[i].2, conf[i].3).init(device))
        }

        Self{
            block: blocks_
        }

    }

    pub fn forward(&self, input: Tensor<B, 3>)-> Tensor<B, 3>{

        for basic_layer in &self.block{
            let  input = basic_layer.forward(input.clone());
            // let input = x;

        }
        return input;
    }
}

// xFeat Model

#[derive(Module, Debug)]
pub struct xFeatModel<B: Backend>{

    block1: BasicBlock<B>,
    // block2: BasicLayer<B>,
    // block3: BasicLayer<B>,
    // block4: BasicLayer<B>,
    // block5: BasicLayer<B>,
}

impl<B: Backend> xFeatModel<B>{

    pub fn forward(&self, input: Tensor<B, 3>)-> Tensor<B,3>{
        let x = self.block1.forward(input);
        
        return x
    }
}
#[derive(Config, Debug)]
pub struct xFeatModelConifg{
    // config : (in_channles, out_channles, kernel_size, stride)
    // block1_conf: (usize, usize, usize, usize),
}

impl xFeatModelConifg{

    pub fn init<B: Backend>(&self, device: &B::Device) ->xFeatModel<B>{

        xFeatModel{
            block1: BasicBlock::<B>::new::<1>([(1, 4, 3, 1)],device)
        }
    }
}