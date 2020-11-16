f-AnoGAN: Fast unsupervised anomaly detection with generative adversarial networks
=====

Example of Anomaly Detection using <a href="https://www.sciencedirect.com/science/article/abs/pii/S1361841518302640">f-AnoGAN</a>.

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


## Environment
* Python 3.7.4  
* Tensorflow 1.14.0  
* Numpy 1.17.1  
* Matplotlib 3.1.1  
* Scikit Learn (sklearn) 0.21.3  

## Reference
[1] Schlegl, Thomas, et al (2019). <a href="https://www.sciencedirect.com/science/article/abs/pii/S1361841518302640">f-AnoGAN: Fast unsupervised anomaly detection with generative adversarial networks</a>. Medical image analysis 54 (2019): 30-44.
