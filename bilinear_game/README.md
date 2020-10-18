# Biliear game experiments

## Dependencies
- Python 3.6 or later
- [PyTorch](https://pytorch.org/) 1.6

## Reproducing: Stochastic \& Full-batch setting

To reproduce our results on the stochastic and the full-batch bilinear example see 
`LA-MM_stochastic_bilinear.ipynb` and `batch_methods.ipynb`, respectively.
To plot SVRE results for the former setting, upload the given `svre_results.json` file to your drive, or modify the following line accordingly:  

```
with open("./svre_results.json", 'r') as fs:
    ...
```

We recommend reproducing our MNIST results in [Google Colab](https://colab.research.google.com/).
