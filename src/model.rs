use burn::Tensor;
use burn::module::Module;
use burn::config::Config;
use burn::tensor::backend::Backend;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{BatchNorm,BatchNormConfig, PaddingConfig2d, Relu};
 
// BasicLayer
#[derive(Module,Debug)]
pub struct BasicLayer<B: Backend>{
 
    conv1: Conv2d<B>,
    batchnorm: BatchNorm<B>,
    relu: Relu
}
 
impl<B: Backend> BasicLayer<B>{
 
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B,4>{
       
        let x = self.conv1.forward(input);
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
    padding: usize,
}
 
impl BasicLayerConfig{
 
    pub fn init<B: Backend>(&self, device: &B::Device) ->BasicLayer<B>{
        BasicLayer{
            conv1: Conv2dConfig::new([self.in_channel, self.out_channel], [self.kernel_size, self.kernel_size])
                .with_stride([self.stride, self.stride]).with_padding(PaddingConfig2d::Explicit(self.padding,self.padding,self.padding,self.padding)).init(device),
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

    pub fn new<const N:usize>(conf: [(usize, usize, usize, usize, usize); N], device: &B::Device)-> Self {

        let mut blocks_: Vec<BasicLayer<B>> = Vec::new();
        for i in 0..N {
            blocks_.push(BasicLayerConfig::new(conf[i].0, conf[i].1, conf[i].2, conf[i].3, conf[i].4).init(device))
        }

        Self{
            block: blocks_
        }

    }

    pub fn forward(&self, input: Tensor<B, 4>)-> Tensor<B, 4>{

        let mut x = input.clone();
        for basic_layer in &self.block{
            x = basic_layer.forward(x);

        }
        return x;
    }
}

// xFeat Model

#[derive(Module, Debug)]
pub struct xFeatModel<B: Backend>{

    block1: BasicBlock<B>,
    block2: BasicBlock<B>,
    block3: BasicBlock<B>,
    block4: BasicBlock<B>,
    block5: BasicBlock<B>,
}

impl<B: Backend> xFeatModel<B>{

    pub fn forward(&self, input: Tensor<B, 4>)-> Tensor<B,4>{
        let x = self.block1.forward(input);
        let x = self.block2.forward(x);
        let x = self.block3.forward(x);
        let x = self.block4.forward(x);
        let x = self.block5.forward(x);
        
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
            block1: BasicBlock::<B>::new::<4>([(1, 4, 3, 1, 1), (4, 8, 3, 2, 1), (8, 8, 3, 1, 1), (8, 24, 3, 2, 1)],device),
            block2: BasicBlock::<B>::new::<2>([(24, 24, 3, 1, 1), (24, 24, 3, 1, 1)],device),
            block3: BasicBlock::<B>::new::<3>([(24, 64, 3, 2, 1), (64, 64, 3, 1, 1), (64, 64, 3, 1, 0)],device),
            block4: BasicBlock::<B>::new::<3>([(64, 64, 3, 2, 1), (64, 64, 3, 1, 1), (64, 64, 3, 1, 1)],device),
            block5: BasicBlock::<B>::new::<4>([(64, 128, 3, 2, 1), (128, 128, 3, 1, 1), (128, 128, 3, 1, 1), (128, 64, 3, 1, 0)],device),

        }
    }
}