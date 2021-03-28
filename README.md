# LAGAN: Lookahead-Minmax

Source code of our paper [Taming GANs with Lookahead–Minmax](https://openreview.net/pdf?id=ZW0yXJyNmoG), ICLR 2021. 
Equal contribution with [Matteo Pagliardini](https://github.com/mpagli), 
and joint work with Sebastian Stich, François Fleuret, and Martin Jaggi.

To tackle the known challenges of minmax optimization of: *(i)* rotational joint vector field and *(ii)* sensitivity to noise induced by the stochastic gradient, we propose the *Lookahead-Minmax* algorithm. 
It consists of periodically taking an iterate that lies on a convex combination of the current and a past iterate.

![lookahead-minmax illustration](lookahead_minmax.png?raw=true)

---------------------------

The subdirectories contain:
- `lagan`: the experiements on CIFAR10, SVHN and ImageNet. 
- `mnist`: the experiement on MNIST. 
- `bilinear_game`: the [Colab-notebooks](https://colab.research.google.com/) to reproduce the results on batch and stochastic bilinear game.


See also: [paper](https://openreview.net/pdf?id=ZW0yXJyNmoG), [poster](https://drive.google.com/file/d/1WoxLl8fx_xby0oOf4Yhceo1Oy3JR99oa/view?usp=sharing), and [slides](https://drive.google.com/file/d/1pxPxAxKep0vPeJQNhpqhLm9XvWS2x6iU/view?usp=sharing).



### Citation
```
@inproceedings{chavdarova2021taming,
title={{Taming GANs with Lookahead-Minmax}},
author={Tatjana Chavdarova and Matteo Pagliardini and Sebastian U Stich and Fran{\c{c}}ois Fleuret and Martin Jaggi},
booktitle={International Conference on Learning Representations},
year={2021}
}
```