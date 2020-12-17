# Discovering Pattern Structure Using Differentiable Compositing

[![Everything Is AWESOME](http://geometry.cs.ucl.ac.uk/projects/2020/diffcompositing/paper_docs/teaser.png)](https://www.youtube.com/embed/KM7PIyb06dc "Everything Is AWESOME")


## Illustration
<img src="http://geometry.cs.ucl.ac.uk/projects/2020/diffcompositing/paper_docs/compile.png">


##Requirements
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
