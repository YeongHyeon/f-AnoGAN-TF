f-AnoGAN: Fast unsupervised anomaly detection with generative adversarial networks
=====

TensorFlow implementation of <a href="https://www.sciencedirect.com/science/article/abs/pii/S1361841518302640">f-AnoGAN</a> with MNIST dataset [1].  
The base model <a href="https://github.com/YeongHyeon/WGAN-TF">WGAN</a> is also implemented with TensorFlow.   

## Summary

### f-AnoGAN architecture  
<div align="center">
  <img src="./figures/fanogan.png" width="650">  
  <p>The architecture of f-AnoGAN [1].</p>
  <img src="./figures/anomaly_score.png" width="750">  
  <p>The logic for calculating anomaly score [1].</p>
</div>

### Graph in TensorBoard
<div align="center">
  <img src="./figures/graph.png" width="650">  
  <p>Graph of f-AnoGAN.</p>
</div>

### Problem Definition
<div align="center">
  <img src="./figures/definition.png" width="450">  
  <p>'Class-1' is defined as normal and the others are defined as abnormal.</p>
</div>

## Results

### Training Phase-1 (WGAN Training)

#### Training graph of Phase-1
The rear half of the graph represents the state of the training phase 2.  

<div align="center">


|Term Real|Term Fake|
|:---:|:---:|
|<img src="./figures/f-AnoGAN_mean_real.svg" width="300">|<img src="./figures/f-AnoGAN_mean_fake.svg" width="300">|

|Loss D (Discriminator)|Loss G (Generator)|
|:---:|:---:|
|<img src="./figures/f-AnoGAN_loss_d.svg" width="300">|<img src="./figures/f-AnoGAN_loss_g.svg" width="300">|

</div>

#### Result of Phase-1
<div align="center">

|z:2|z:2 (latent space walking)|
|:---:|:---:|
|<img src="./figures/z02.png" width="250">|<img src="./figures/z02_lw.png" width="250">|

|z:64|z:128|
|:---:|:---:|
|<img src="./figures/z64.png" width="250">|<img src="./figures/z128.png" width="250">|

### Training Phase-2 (izi Training)

#### Training graph of Phase-2
The front half of the graph represents the state of the training phase 1.  

<div align="center">

|Term izi|Term ziz|Loss E (Encoder)|
|:---:|:---:|:---:|
|<img src="./figures/f-AnoGAN_mean_izi.svg" width="300">|<img src="./figures/f-AnoGAN_mean_ziz.svg" width="300">|<img src="./figures/f-AnoGAN_loss_e.svg" width="300">|

</div>

#### Result of Phase-2
<div align="center">
  <img src="./figures/restoring.png" width="800">  
  <p>Restoration result by f-AnoGAN.</p>
</div>

### Test Procedure
<div align="center">
  <img src="./figures/test-box.png" width="400">
  <p>Box plot with encoding loss of test procedure.</p>
</div>

<div align="center">
  <p>
    <img src="./figures/in_in01.png" width="130">
    <img src="./figures/in_in02.png" width="130">
    <img src="./figures/in_in03.png" width="130">
  </p>
  <p>Normal samples classified as normal.</p>

  <p>
    <img src="./figures/in_out01.png" width="130">
    <img src="./figures/in_out02.png" width="130">
    <img src="./figures/in_out03.png" width="130">
  </p>
  <p>Abnormal samples classified as normal.</p>

  <p>
    <img src="./figures/out_in01.png" width="130">
    <img src="./figures/out_in02.png" width="130">
    <img src="./figures/out_in03.png" width="130">
  </p>
  <p>Normal samples classified as abnormal.</p>

  <p>
    <img src="./figures/out_out01.png" width="130">
    <img src="./figures/out_out02.png" width="130">
    <img src="./figures/out_out03.png" width="130">
  </p>
  <p>Abnormal samples classified as abnormal.</p>
</div>


## Environment
* Python 3.7.4  
* Tensorflow 1.14.0  
* Numpy 1.17.1  
* Matplotlib 3.1.1  
* Scikit Learn (sklearn) 0.21.3  

## Reference
[1] Schlegl, Thomas, et al (2019). <a href="https://www.sciencedirect.com/science/article/abs/pii/S1361841518302640">f-AnoGAN: Fast unsupervised anomaly detection with generative adversarial networks</a>. Medical image analysis 54 (2019): 30-44.
