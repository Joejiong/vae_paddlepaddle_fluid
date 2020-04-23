# VAE PaddlePaddle Fluid 

###### 2020.4.23

This is a paddle implementation of the paper Auto-Encoding Variational Bayes by Kingma and Welling.
 - It uses ReLUs and the momentum optimizer, instead of sigmoids and adagrad. 
 - These changes make the network converge much faster.（参考竞品实现描述）
https://github.com/pytorch/examples/tree/master/vae

VAE模型可以用做：

 - 图片reconstruction 
 - 随机数生成图片 
 - img segmentation
 - denoising
 - Learned MNIST manifold 无监督聚类
 
相关环境

- Pytorch: 1.4+
- paddlepaddle: 1.7.1
- Python: 3.7+
 
  
 ## 关键技术点/子模块设计与实现方案

主要包括：

 - encoder 
 - decoder
 - 变分重采样（重要函数 reparameterize, 内部不能完全paddle实现）
 - 重要函数 Loss （KLloss+BCE）BCEloss暂时dev里面不能用（内部不能完全paddle实现）
 
 ## Usage
### Prerequisites
1. PaddlePaddle/fluid/dygraph
2. Pytorch
2. Python packages : numpy, scipy, PIL(or Pillow), matplotlib


### paddle：
VAElinear:
 
 - train_vae_linear.py 
 - vae_linear.py
 

CNNVAE:

 - train_cnn_vae.py
 - cnn_vae.py

### pytorch：
VAElinear:

- pytorch_vae.py


### Command
paddle: 
```
python3 train_vae_linear.py
python3 train_cnn_vae.py
```
pytorch: 
```
python3 pytorch_vae.py
python3 pytorch_vae_cnn.py
```
## problems
当前版本还有点问题，restore img 已经没问题了，其他任务还有待提升
由于缺乏没有torch image process模块，所以暂时paddle的result不是很好看，可以用暂时用torch的模块。
## Todo:
- conv fix bug when generating results 
- latent manifold clustering
- image process utils.py 
