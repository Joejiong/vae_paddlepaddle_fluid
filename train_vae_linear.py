# -*- coding: utf-8 -*-

# vae 手写数字生成

import os
import random
import paddle
import paddle.fluid as fluid
import numpy as np
from paddle_vae_linear import VAE_Linear
from PIL import Image  
import PIL 
import cv2
from torchvision.utils import save_image


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    # print("recon",recon_x.shape)
    # print("x",x.shape)
    x = fluid.layers.reshape(x, shape=[-1, 784])
    # print("x",x.shape)
    BCE = fluid.layers.sigmoid_cross_entropy_with_logits(recon_x, x)

    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * fluid.layers.sum(1 + logvar - mu*mu - fluid.layers.exp(logvar))
    
    BCE = fluid.layers.reduce_sum(BCE, dim=1)
    KLD = fluid.layers.reduce_sum(KLD, dim=1)
    # print("bce",BCE.shape)
    # print("kld",KLD.shape)
    return BCE + KLD
    # return x
def train2(epoch):
    model.train()
    train_loss = 0
    epoch_num = 1
    train_loader = paddle.batch(paddle.dataset.mnist.train(), batch_size=100)
    for epoch in range(epoch_num):
        for batch_idx, (data, _) in enumerate(train_loader):
            opt = fluid.optimizer.Momentum(learning_rate=0.001, momentum=0.9, parameter_list=model.parameters())
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            # optimizer.step()
            avg_loss = fluid.layers.mean(loss)
                
            if batch_idx % 1000 == 0:
                print("epoch: {}, batch_id: {}, loss is: {}".format(epoch, batch_idx, avg_loss.numpy()))
            avg_loss.backward()
            opt.minimize(avg_loss)
            model.clear_gradients()
        #     if batch_idx % 1000 == 0:
        #         print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #             epoch, batch_idx * len(data), len(train_loader.dataset),
        #             100. * batch_idx / len(train_loader),
        #             loss.item() / len(data)))

        # print('====> Epoch: {} Average loss: {:.4f}'.format(
        #       epoch, train_loss / len(train_loader.dataset)))

# 定义训练过程
def train(model):
    print('start training ... ')
    model.train()
    epoch_num = 2
    opt = fluid.optimizer.Momentum(learning_rate=0.001, momentum=0.9, parameter_list=model.parameters())
    # 使用Paddle自带的数据读取器
    train_loader = paddle.batch(paddle.dataset.mnist.train(), batch_size=100)
    valid_loader = paddle.batch(paddle.dataset.mnist.test(), batch_size=100)
    for epoch in range(epoch_num):
        for batch_id, data in enumerate(train_loader()):
            # 调整输入数据形状和类型
            x_data = np.array([item[0] for item in data], dtype='float32').reshape(-1, 1, 28, 28)
            # y_data = np.array([item[1] for item in data], dtype='int64').reshape(-1, 1)
            # 将numpy.ndarray转化成Tensor
            img = fluid.dygraph.to_variable(x_data)
            recon_batch, mu, logvar = model(img)

            # logits = model(img)
            # 计算损失函数
            # loss = fluid.layers.softmax_with_cross_entropy(logits, label)

            loss = loss_function(recon_batch, img, mu, logvar)
            avg_loss = fluid.layers.mean(loss)
            
            if batch_id % 1000 == 0:
                print("epoch: {}, batch_id: {}, loss is: {}".format(epoch, batch_id, avg_loss.numpy()))
            avg_loss.backward()
            opt.minimize(avg_loss)
            model.clear_gradients()
        #
        print("----------------")
        
        model.eval()
        accuracies = []
        losses = []
        test_loss = 0
        for batch_id, data in enumerate(valid_loader()):
        # 调整输入数据形状和类型
            x_data = np.array([item[0] for item in data], dtype='float32').reshape(-1, 1, 28, 28)
            img = fluid.dygraph.to_variable(x_data)
            recon_batch, mu, logvar = model(img)
            test_loss = loss_function(recon_batch, img, mu, logvar)
            avg_loss = fluid.layers.mean(test_loss)
            if batch_id % 1000 == 0:
                print("epoch: {}, batch_id: {}, [validation] loss is: {}".format(epoch, batch_id, avg_loss.numpy()))
            

            if batch_id == 0:
                n = min(img.shape[0], 8)
                comparison = paddle.fluid.layers.concat([img[:n],
                                      fluid.layers.reshape(recon_batch[:n], (8, 1, 28, 28))], axis=1)                
                print(comparison.shape)
                comparison = comparison.numpy()[:n]
                # comparison = 
                imgs = np.append(comparison[:,0,:,:], comparison[:,1,:,:], axis=1)
                print(comparison.reshape(8, 1, 28, 56).shape)  
                comparison = comparison.reshape(224, 56) 
                # comparison = np.squeeze(comparison)
                # print(comparison)
                # comparison = np.squeeze(comparison, axis=1)
                # comparison = np.squeeze(comparison.numpy(), axis=1)
                img = Image.fromarray(comparison.astype(np.uint8))
                img.save(os.path.join('My_results/vae_reconstruction_'+ str(epoch) + '.png'))
                

                random_z = np.random.rand(1, 20).astype('float32')
                
                b = fluid.layers.create_tensor(dtype="float32")
                fluid.layers.assign(random_z, b)
                fluid.dygraph.to_variable(b)
                random_z = model.decode(b)
                random_z = random_z.numpy()
                # print(sample.shape)
                # print(random_z.numpy())
                cv2.imwrite('My_results/cv_random_z_gen_img_'+ str(epoch) + '.jpg', random_z.reshape(28, 28)*255.0)
                random_z = np.concatenate((random_z),axis=0)
                # random_z = np.concatenate((random_z),axis=0)

                # print(random_z.shape)
                # img = Image.fromarray(random_z.reshape(28*8, 28*8).astype(np.uint8)*255.0)

                # img.save(os.path.join('My_results/random_z_gen_img_'+ str(epoch) + '.png'))
        # print("[validation] accuracy/loss: {}/{}".format(np.mean(accuracies), np.mean(losses)))
        model.train()
        print("-----------------------")


        # img = Image.fromarray(sample.astype(np.uint8))
        # img.save(os.path.join('results/reconstruction_'+ str(epoch) + '.png', nrow=8))

    # 保存模型参数
    fluid.save_dygraph(model.state_dict(), 'mnist')




if __name__ == '__main__':
    print('nihao')
    # 创建模型
    with fluid.dygraph.guard():
        model = VAE_Linear("VAE", num_classes=10)
        #启动训练过程
        train(model)