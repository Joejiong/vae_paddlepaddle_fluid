# 导入需要的包
import paddle
import paddle.fluid as fluid
import numpy as np
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, Linear, Conv2DTranspose
from paddle.fluid.dygraph.base import to_variable



class VAE_Linear(fluid.dygraph.Layer):
    def __init__(self, name_scope, num_classes=1):
        super(VAE_Linear, self).__init__(name_scope)

        # 创建卷积和池化层块，每个卷积层使用Sigmoid激活函数，后面跟着一个2x2的池化
        self.fc1 = Linear(784, 400)
        self.fc21 = Linear(400, 20)
        self.fc22 = Linear(400, 20)

        self.fc3 = Linear(20, 400)
        self.fc4 = Linear(400, 784)

    def encode(self, x):
        # print("x",x.shape)
        # h = Linear(input_dim=400, output_dim=64, act='relu')
        x1 =  self.fc1(x)
        # print("x1",x1.shape)
        h = fluid.layers.relu(x1)
        # print("h",h.shape)
        return self.fc21(h), self.fc22(h) # mu, log_var

    def reparameterize(self, mu, logvar):
        std = fluid.layers.exp(0.5*logvar)
        # eps = torch.randn_like(std)
        eps = np.random.randn(*std.shape).astype('float32')
        eps = to_variable(eps)
        # mu = to_variable(mu)
        # std = to_variable(std)
        # print(std.type)
        # print(eps.type)
        # print(mu.type)
        return mu + eps*std
        # return mu
    
    def decode(self, z):
        # print("z_cool",z.shape)
        # print("z_cool",z.type)
        z1 = self.fc3(z)
        # print("z1",z1.shape)
        h3 = fluid.layers.relu(z1)
        # print("h3",h3.shape)
        return fluid.layers.sigmoid(self.fc4(h3))

    # 网络的前向计算过程
    def forward(self, x):
        mu, logvar = self.encode(fluid.layers.reshape(x, (-1, 784)))
        # mu, logvar = self.encode(x.reshape(10, 784))
        z = self.reparameterize(mu, logvar)
        # print("forword z",z.shape)
        # print('z_before decode',z.shape)
        return self.decode(z), mu, logvar
