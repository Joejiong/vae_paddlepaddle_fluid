import paddle
import paddle.fluid as fluid
import numpy as np
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, Linear, Conv2DTranspose
from paddle.fluid.dygraph.base import to_variable



class VAE_CNN(fluid.dygraph.Layer):
    def __init__(self, name_scope):
        super(VAE_CNN, self).__init__(name_scope)

        self.conv1 = Conv2D(num_channels=1, num_filters=16, filter_size=4, stride=2, padding=(15,15), act="relu")
        self.conv2 = Conv2D(num_channels=16, num_filters=16, filter_size=4, stride=2, padding=(15,15), act="relu")
        # self.conv3 = Conv2D(num_channels=64, num_filters=64, filter_size=4, stride=2, padding=(15,15), act="relu")
        self.fc_encode1 = Linear(12544, 1024, act="relu")
        self.fc_encode2 = Linear(12544, 1024, act="relu")


        self.fc_decode1 = Linear(1024, 12544, act="relu")
        # self.fc_decode1 = Linear(784, 25088, act="relu")
        # self.fc_decode2 = Linear(25088, 25088, act="relu")

        self.dconv1 = Conv2DTranspose(num_channels=16, num_filters=16, filter_size=4, stride=2, padding=(15,15), act="relu")
        self.dconv2 = Conv2DTranspose(num_channels=16, num_filters=1, filter_size=4, stride=2, padding=(15,15), act="relu")

        self.fc_decode3 = Linear(12544, 12544, act="relu")
        self.fc_decode4 = Linear(784, 784, act="sigmoid")
        
    
    def encoder(self, x):
   
        x = fluid.layers.reshape(x, shape=[-1, 1, 28, 28])
        # print("x1",x.shape)
        
        x = self.conv1(x)
        # print(x.shape)

        x = fluid.layers.dropout(x, dropout_prob=0.5)
        x = self.conv2(x)
        # print("x2",x.shape)
        x = fluid.layers.dropout(x, dropout_prob=0.5)

        # print("x4",x.shape) 
        x = fluid.layers.flatten(x)
        # print("x5",x.shape) 
        x1 = self.fc_encode1(x)
        x2 = self.fc_encode2(x)
        return x1,x2
        
        # return fluid.layers.flatten(x), fluid.layers.flatten(x) 

    def reparameterize(self, mu, logvar):
        std = fluid.layers.exp(0.5*logvar)
        # eps = torch.randn_like(std)
        eps = np.random.randn(*std.shape).astype('float32')
        eps = to_variable(eps)
        
        return mu + eps*std
        
    def decoder(self, z):
        # print("z", z.shape)
        x = self.fc_decode1(z)
        # print("x11",x.shape)
        # x = self.fc_decode2(x)
        # print("x12",x.shape)
        x = fluid.layers.reshape(x, shape=[-1, 16, 28, 28])
        # print("x13",x.shape)
        x = self.dconv1(x)
        # print("x14",x.shape)
        x = fluid.layers.dropout(x,dropout_prob=0.5)
        # print("x15",x.shape)
        x = self.dconv2(x)
        print("x16",x.shape)
      
        x = fluid.layers.flatten(x)
        # print("x flatten", x.shape)
        x = self.fc_decode4(x)
        # print()
        return x
    
    def forward(self, x):
        mu, logvar = self.encoder(fluid.layers.reshape(x, (-1, 1, 28, 28)))
        # mu, logvar = self.encode(x.reshape(10, 784))
        z = self.reparameterize(mu, logvar)
        # print("forword z",z.shape)
        # print('z_before decode',z.shape)
        return self.decoder(z), mu, logvar
