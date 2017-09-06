import chainer
from chainer import Variable
import chainer.links as L
import chainer.functions as F
import numpy as np
from pdb import set_trace as st

class BRCD(chainer.Chain):
    # batch normalization
    # relu
    # convolution 3x3
    # dropout
    def __init__(self, in_channel, out_channel, conv_size, stride=1, pad=1, drop_ratio=0.2):
        self.drop_ratio = drop_ratio
        super(BRCD, self).__init__(
            bn  =L.BatchNormalization(in_channel),
            conv=L.Convolution2D(in_channel, out_channel, conv_size, stride=stride, pad=pad),
        )
    def __call__(self, x):
        return F.dropout(self.conv(F.leaky_relu(self.bn(x),slope=0.1)), ratio=self.drop_ratio)


class DenseBlock(chainer.Chain):
    def __init__(self, n_layers, in_channel, growth_rate, drop_ratio, is_up=False):
        super(DenseBlock, self).__init__()
        self.is_up = is_up
        in_chs = [in_channel+i*growth_rate for i in range(n_layers)]
        self.links = [None] * n_layers
        for i in range(n_layers):
            self.links[i] = ('BRCD{}'.format(i), BRCD(in_chs[i], growth_rate, conv_size=3, drop_ratio=drop_ratio))
            self.add_link(*self.links[i])

    def __call__(self, stack):
        new_features = []
        for i in range(len(self.links)):
            _, link = self.links[i]
            h = link(stack)
            stack = F.concat((stack, h), axis=1)
            if self.is_up:
                new_features.append(h)
        if self.is_up:
            return F.concat(new_features, axis=1)
        else:
            return stack


class TransitionDown(chainer.Chain):
    def __init__(self, in_channel):
        super(TransitionDown, self).__init__(
            brcd=BRCD(in_channel, in_channel, conv_size=1, stride=1, pad=0)
        )
    def __call__(self, x):
        return F.max_pooling_2d(self.brcd(x), 2, stride=2, pad=0)


class TransitionUp(chainer.Chain):
    def __init__(self, in_channel, out_channel):
        super(TransitionUp, self).__init__(
            # I'm not sure how to double the resolution size of the feature maps precisely by 3x3 deconv in chainer.
            # So here use 2x2 deconv with stride=2.
            deconv=L.Deconvolution2D(in_channel, out_channel, 2, stride=2)
        )
    def __call__(self, x, skip_connection):
        return F.concat((self.deconv(x), skip_connection), axis=1)
        

class Tiramisu103(chainer.Chain):
    def __init__(self, n_classes, in_channels=3):
        n_channels_first_conv = 48
        growth_rate = 16
        DB_layers = [0,4,5,7,10,12,15]
        drop_ratio = 0.2

        in_chs = [n_channels_first_conv + growth_rate*sum(DB_layers[:i+1]) for i in range(len(DB_layers))]
        up_chs = [growth_rate*i for i in DB_layers[1:]]

        super(Tiramisu103, self).__init__(
            conv=L.Convolution2D(in_channels, in_chs[0], 3, stride=1, pad=1),
            DenseBlockDown1=DenseBlock(DB_layers[1], in_chs[0], growth_rate, drop_ratio),
            TransitionDown1=TransitionDown(in_chs[1]), 
            DenseBlockDown2=DenseBlock(DB_layers[2], in_chs[1], growth_rate, drop_ratio),
            TransitionDown2=TransitionDown(in_chs[2]),
            DenseBlockDown3=DenseBlock(DB_layers[3], in_chs[2], growth_rate, drop_ratio),
            TransitionDown3=TransitionDown(in_chs[3]),
            DenseBlockDown4=DenseBlock(DB_layers[4], in_chs[3], growth_rate, drop_ratio),
            TransitionDown4=TransitionDown(in_chs[4]),
            DenseBlockDown5=DenseBlock(DB_layers[5], in_chs[4], growth_rate, drop_ratio),
            TransitionDown5=TransitionDown(in_chs[5]),

            DenseBlockMid=DenseBlock(DB_layers[6], in_chs[5], growth_rate, drop_ratio, is_up=True),

            TransitionUp5=TransitionUp(up_chs[5], up_chs[5]), 
            DenseBlockUp5=DenseBlock(DB_layers[5], in_chs[6], growth_rate, drop_ratio, is_up=True),
            TransitionUp4=TransitionUp(up_chs[4], up_chs[4]), 
            DenseBlockUp4=DenseBlock(DB_layers[4], in_chs[5], growth_rate, drop_ratio, is_up=True),
            TransitionUp3=TransitionUp(up_chs[3], up_chs[3]), 
            DenseBlockUp3=DenseBlock(DB_layers[3], in_chs[4], growth_rate, drop_ratio, is_up=True),
            TransitionUp2=TransitionUp(up_chs[2], up_chs[2]), 
            DenseBlockUp2=DenseBlock(DB_layers[2], in_chs[3], growth_rate, drop_ratio, is_up=True),
            TransitionUp1=TransitionUp(up_chs[1], up_chs[1]), 
            DenseBlockUp1=DenseBlock(DB_layers[1], in_chs[2], growth_rate, drop_ratio, is_up=True),

            classify=L.Convolution2D(up_chs[0], n_classes, 3, stride=1, pad=1),
        )

    def __call__(self, x, t):
        x = self.forward(x)
        self.loss = F.softmax_cross_entropy(x, t)
        self.accuracy = calculate_accuracy(x, t)

        return self.loss

    def forward(self, x):

        x = self.conv(x) # 3 -> 48
        skip1 = self.DenseBlockDown1(x) # 48 + 4*16 = 112
        x = self.TransitionDown1(skip1) 
        skip2 = self.DenseBlockDown2(x) # 112 + 5*16 = 192
        x = self.TransitionDown2(skip2) 
        skip3 = self.DenseBlockDown3(x) # 192 + 7*16 = 304
        x = self.TransitionDown3(skip3) 
        skip4 = self.DenseBlockDown4(x) # 304 + 10*16 = 464
        x = self.TransitionDown4(skip4) 
        skip5 = self.DenseBlockDown5(x) # 464 + 12*16 = 656
        x = self.TransitionDown5(skip5) 

        x = self.DenseBlockMid(x) # 15*16 = 240
        x = self.TransitionUp5(x, skip5) # 656 + 240 = 896
        del skip5
        x = self.DenseBlockUp5(x) # 12*16 = 192
        x = self.TransitionUp4(x, skip4) # 464 + 192 = 656
        del skip4
        x = self.DenseBlockUp4(x) # 10*16 = 160
        x = self.TransitionUp3(x, skip3) # 304 + 160 = 464
        del skip3
        x = self.DenseBlockUp3(x) # 7*16 = 112
        x = self.TransitionUp2(x, skip2) # 192 + 112 = 304
        del skip2
        x = self.DenseBlockUp2(x) # 5*16 = 80
        x = self.TransitionUp1(x, skip1) # 112 + 80 = 192
        del skip1
        x = self.DenseBlockUp1(x) # 4*16 = 64

        return F.leaky_relu(self.classify(x), slope=0.1)



def calculate_accuracy(predictions, truths):
    predictions = predictions.data.argmax(1)
    truths = truths.data
    no_count = (truths==-1).sum()
    count = (predictions == truths).sum()
    acc = count / float(truths.size-no_count)
    return acc


def unit_test():
    chainer.cuda.check_cuda_available()
    chainer.cuda.get_device(0).use()

    # model = Tiramisu103(2)
    model = Tiramisu103(2)
    model.to_gpu(0)
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # y = model.forward(x)
    for i in range(3):

        x = np.random.rand(1,3,224,224).astype(np.float32)
        x = chainer.Variable(chainer.cuda.cupy.asarray(x))
        t = (np.random.rand(1,224,224) * 2).astype(np.int32)
        t = chainer.Variable(chainer.cuda.cupy.asarray(t))
        loss = model(x,t)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        print('| acc : %.2f' % model.accuracy)
        print('| loss: %.2f' % model.loss.data)

        del x,t

if __name__ == '__main__':
    unit_test()
    print('| ok!')

