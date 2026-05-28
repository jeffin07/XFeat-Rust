use burn::Tensor;
use burn::module::Module;
use burn::config::Config;
use burn::tensor::backend::Backend;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{BatchNorm,BatchNormConfig, PaddingConfig2d, Relu, Sigmoid};
use burn::nn::modules::{
    pool::{AvgPool2d, AvgPool2dConfig},
    interpolate::{Interpolate2dConfig,InterpolateMode},
};
 
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

// skip
#[derive(Module, Debug)]
pub struct Skip<B: Backend>{

    avgpool: AvgPool2d,//AvgPool2dConfig::new([4, 4]).with_stride([2, 2]).init(device)
    conv: Conv2d<B>
}

impl<B: Backend> Skip<B>{

    pub fn new(device: &B::Device)-> Self{

        Skip{

            avgpool: AvgPool2dConfig::new([4, 4]).with_strides([4, 4]).init(),
            conv: Conv2dConfig::new([1, 24], [1, 1])
                .with_stride([1, 1]).with_padding(PaddingConfig2d::Explicit(0, 0, 0, 0)).init(device),
        }
    }

    pub fn forward(&self, input: Tensor<B, 4>)-> Tensor<B, 4>{

        let x = self.avgpool.forward(input);
        let x = self.conv.forward(x);

        return x
    }
}

// fusion block
#[derive(Module, Debug)]
pub struct  FusionBlock<B: Backend>{

    l1: BasicBlock<B>,
    l2: BasicBlock<B>,
    l3: Conv2d<B>
}

impl<B: Backend> FusionBlock<B>{

    pub fn new(device: &B::Device)-> Self{

        FusionBlock{
            l1: BasicBlock::<B>::new::<1>([(64, 64, 3, 1, 1)],device),
            l2: BasicBlock::<B>::new::<1>([(64, 64, 3, 1, 1)],device),
            l3: Conv2dConfig::new([64, 64], [1, 1]).with_padding(PaddingConfig2d::Explicit(0, 0, 0, 0)).init(device)
        }
    }

    pub fn forward(&self, input: Tensor<B, 4>)-> Tensor<B, 4>{

        let x = self.l1.forward(input);
        let x = self.l2.forward(x);
        let x = self.l3.forward(x);

        return x
    }
}

// heatmap

#[derive(Module, Debug)]
pub struct  HeatMapHead<B: Backend>{

    l1: BasicBlock<B>,
    l2: BasicBlock<B>,
    l3: Conv2d<B>,
    act: Sigmoid,
}

impl<B: Backend> HeatMapHead<B>{

    pub fn new(device: &B::Device)-> Self{

        HeatMapHead{
            l1: BasicBlock::<B>::new::<1>([(64, 64, 3, 1, 0)],device),
            l2: BasicBlock::<B>::new::<1>([(64, 64, 3, 1, 0)],device),
            l3: Conv2dConfig::new([64, 1], [1, 1]).with_stride([1, 1]).with_padding(PaddingConfig2d::Explicit(0, 0, 0, 0)).init(device),
            act: Sigmoid::new()
        }
    }

    pub fn forward(&self, input: Tensor<B, 4>)-> Tensor<B, 4>{

        let x = self.l1.forward(input);
        let x = self.l2.forward(x);
        let x = self.l3.forward(x);
        let x = self.act.forward(x);

        return x
    }
}

// keypoint Head
#[derive(Module, Debug)]
pub struct KeyPointHead<B: Backend>{

    l1: BasicBlock<B>,
    l2: BasicBlock<B>,
    l3: BasicBlock<B>,
    l4: Conv2d<B>,
}

impl<B: Backend> KeyPointHead<B>{

    pub fn new(device: &B::Device)-> Self{

        KeyPointHead{
            l1: BasicBlock::<B>::new::<1>([(64, 64, 3, 1, 0)],device),
            l2: BasicBlock::<B>::new::<1>([(64, 64, 3, 1, 0)],device),
            l3: BasicBlock::<B>::new::<1>([(64, 64, 3, 1, 0)],device),
            l4: Conv2dConfig::new([64, 65], [1, 1]).with_stride([1, 1]).with_padding(PaddingConfig2d::Explicit(0, 0, 0, 0)).init(device),
        }
    }

    pub fn forward(&self, input: Tensor<B, 4>)-> Tensor<B, 4>{

        let x = self.l1.forward(input);
        let x = self.l2.forward(x);
        let x = self.l3.forward(x);
        let x = self.l4.forward(x);

        return x
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
    fusion: FusionBlock<B>,
    headmap: HeatMapHead<B>,
    // keypoint: KeyPointHead<B>,
    skip: Skip<B>,
}

impl<B: Backend> xFeatModel<B>{

    pub fn forward(&self, input: Tensor<B, 4>)-> Tensor<B,4>{
        let x1 = self.block1.forward(input.clone());
        let x2 = self.block2.forward(x1 + self.skip.forward(input));
        let mut x3 = self.block3.forward(x2);
        let mut x4 = self.block4.forward(x3.clone());
        let x5 = self.block5.forward(x4.clone());
        // let x6 = self.fusion.forward(x5);
        // let x7 = self.headmap.forward(x6);
        // let x = self.keypoint.forward(x); Need to create fusion pyramid

        let x4_interpolate = Interpolate2dConfig::new().with_output_size(Some([x3.dims()[2], x3.dims()[3]])).with_mode(InterpolateMode::Linear).init();
        let x5_interpolate = Interpolate2dConfig::new().with_output_size(Some([x3.dims()[2], x3.dims()[3]])).with_mode(InterpolateMode::Linear).init();
        let x4 = x4_interpolate.forward(x4);
        let x5 = x5_interpolate.forward(x5);

        let feats = self.fusion.forward(x3 + x4 + x5);
        let heatmap = self.headmap.forward(feats);
        
        return heatmap
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
            fusion: FusionBlock::<B>::new(device),
            headmap: HeatMapHead::<B>::new(device),
            // keypoint: KeyPointHead::<B>::new(device),
            skip: Skip::<B>::new(device),

        }
    }
}