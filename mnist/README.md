# How to run those notebooks

Those notebooks assume that you have a pretrained MNIST classifier located at "./drive/My Drive/Data/models/mnist.pth". If you run those notebooks using Google Colab, you might want too upload this model (available in this folder), and change the path manually. To change the path manually look for the following line in the notebooks:

```python
pretrained_clf = pretrained_mnist_model(pretrained='./drive/My Drive/Data/models/mnist.pth')
```