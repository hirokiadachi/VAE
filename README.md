# Variational Autoencoder (VAE)
VAE is proposed by Kingma and Welling in 2014 on ICLR.<br>
This code has implemented with pytorch version 1.0 and python3.<br>
If you execute this source code, you type the command as shown below on terminal.

```
python3 main.py
```
This VAE can choice network type (i.e. FC only or CNNs).<br>
Default network is FC only.<br>
Usage is shown below:<br>
* **FC only: fully connected only**
```
python3 main.py --network_type 'fc'
```

* **CNNs: convolutional neural networks**
```
python3 main.py --network_type 'cnn'
```

# Result
![VAE Result](./image/samples100.jpg)

# Extra items
* [Auto-Encoding Variational Bayes](https://arxiv.org/pdf/1312.6114.pdf)
* [Tutorial on Variational Autoencoders](https://arxiv.org/pdf/1606.05908.pdf)
* [VAE's Slide](./Explanation_VAE_jp.pdf)
* [Chainer official implementation](https://github.com/chainer/chainer/tree/master/examples/vae)
