# Discovering Pattern Structure Using Differentiable Compositing
Patterns, which are collections of elements arranged in regular or near-regular arrangements, are an important graphic art form and widely used due to their elegant simplicity and aesthetic appeal. When a pattern is encoded as a flat image without the underlying structure, manually editing the pattern is tedious and challenging as one has to both preserve the individual element shapes and their original relative arrangements. State-of-the-art deep learning frameworks that operate at the pixel level are unsuitable for manipulating such patterns. Specifically, these methods can easily disturb the shapes of the individual elements or their arrangement, and thus fail to preserve the latent structures of the input patterns. We present a novel differentiable compositing operator using pattern elements and use it to discover structures, in the form of a layered representation of graphical objects, directly from raw pattern images. This operator allows us to adapt current deep learning based image methods to effectively handle patterns. We evaluate our method on a range of patterns and demonstrate superiority in the context of pattern manipulations when compared against state-of-the-art pixel-based pixel- or point-based alternatives.

Website: http://geometry.cs.ucl.ac.uk/projects/2020/diffcompositing/

[![Short Video](http://geometry.cs.ucl.ac.uk/projects/2020/diffcompositing/paper_docs/teaser.png)](https://www.youtube.com/embed/KM7PIyb06dc)

We present a differentiable function F to composite a set of discrete elements into a pattern image. This directly connects vector graphics to image-based losses (e.g., L_2 loss, style loss) and allows us to optimize discrete elements to minimize losses on the composited image. Minimizing an L_2 loss gives us a decomposition of an existing flat pattern image into a set of depth-ordered discrete elements that can be edited individually. Minimizing a style loss allows us to make a pattern tileable or expand a pattern image into a larger pattern composed of discrete elements.

## Illustration
<img src="http://geometry.cs.ucl.ac.uk/projects/2020/diffcompositing/paper_docs/compile.png">


## Requirements
kornia


## Fitting
To run l2 fitting on pattern 115
```python
PYTHONPATH=.:$PYTHONPATH python ./DC/optimization_l2.py --pattern pattern_115 --version 91 --lr 0.1 --non_white --soft_elements --layers --sample 8
```
## Citation
```
@article{reddy2020discovering,
  title={Discovering pattern structure using differentiable compositing},
  author={Reddy, Pradyumna and Guerrero, Paul and Fisher, Matt and Li, Wilmot and Mitra, Niloy J},
  journal={ACM Transactions on Graphics (TOG)},
  volume={39},
  number={6},
  pages={1--15},
  year={2020},
  publisher={ACM New York, NY, USA}
}
```


<span style="color:red">This library might contain reminents of few experiments that are not part of the paper </span>.
