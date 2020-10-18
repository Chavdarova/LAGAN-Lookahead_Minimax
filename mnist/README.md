# MNIST experiments

## Dependencies
- Python 3.6 or later
- [PyTorch](https://pytorch.org/) 1.6

## Reproducing

We recommend reproducing our MNIST results in [Google Colab](https://colab.research.google.com/).

The performance metrics require a pretrained MNIST classifier, available in this directory as `mnist.pth`.
To run the notebooks using Google Colab, it is necessary to upload this model from  this directory, and change the path manually. 
The default location for the metrics classifier is `./drive/My Drive/Data/models/mnist.pth`. 
To change the path modify the following line in the notebooks:
```python
pretrained_clf = pretrained_mnist_model(pretrained='./drive/My Drive/Data/models/mnist.pth')
```