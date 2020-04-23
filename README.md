# vae_paddlepaddle_fluid
This is a paddle implementation of the paper Auto-Encoding Variational Bayes by Kingma and Welling.
- Pytorch: 1.4+
- paddlepaddle: 1.7.1
- Python: 3.7+

 - It uses ReLUs and the momentum optimizer, instead of sigmoids and adagrad. 
 - These changes make the network converge much faster.（参考竞品实现描述）
https://github.com/pytorch/examples/tree/master/vae

VAE模型可以用做：

 - 图片reconstruction 
 - 随机数生成图片 
 - img segmentation
 - denoising
 - Learned MNIST manifold 无监督聚类
 
 ##关键技术点/子模块设计与实现方案

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

