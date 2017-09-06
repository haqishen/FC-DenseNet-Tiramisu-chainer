# chainer FC-DenseNets (Tiramisu103)

This is a chainer implementation of FC-DenseNets (also named Tiramisu103)

The network architecture was described in [The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation](https://arxiv.org/abs/1611.09326). However I found some contradiction between the content of the paper and [their official implementation on Theano](https://github.com/SimJeg/FC-DenseNet).

This implementation is basically a translation of the official theano implementation, while having some uncertainty:

* Original code use 3x3 deconvolution to do upsampling, but I don't know how to double the resolution size of the feature maps precisely by 3x3 deconvolution in chainer. So I use 2x2 deconv instead. https://github.com/haqishen/chainer-FC-DenseNet-Tiramisu/blob/master/Tiramisu.py#L61

* The author said that there are 1088 channels in the first upsampling dense block (https://github.com/SimJeg/FC-DenseNet/blob/master/FC-DenseNet.py#L98-L108) I'm not sure, as I can only see 848 channels there. 
