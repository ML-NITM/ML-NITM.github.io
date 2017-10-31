---
layout: page
title: Assignments
permalink: /assignment/
---
## Assignment - 1 
31-09-2017 ( Tuesday)

### This assignment is to get familiar with Basic Machine Learning Workflow

The task are as follows:-
 - Download numerical dataset (dataset will all `attributes` or `features` in numerical representation) from [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/index.php) also make sure the dataset is in `classification` category.
 - Study the `dataset` and identify `features` and `classes`. 
 - Load the dataset using either `numpy` or `pandas` (Tips is given below)
 - Split the dataset into `train(75%)` and `test(25%)`, do some experiment by varying the `train(x%)` and the `test(%y)` 
 - Build a `classifier` using `sklearn` library, you may use `KNNClassifier` and `try different ML algorithms` and then compare (atleast 2 new algorithm). 
 - `Train` the `classifier` with the `train` dataset
 - `Test` the `classifier` with the `Test` dataset
 - `Visualize` the datapoint where there is `miss-classification`.
 - Plot the data using `scatter plot` by choosing any two features or try all combinations.


### Load CSV file with Numpy
```python
import numpy as np
from sklearn.model_selection import train_test_split
filename="data/pima-indians-diabetes.data"
raw_data = open(filename,'r')
data = np.loadtxt(raw_data,delimiter=',')
print(data.shape)
```

### Loading CSV File with Pandas


```python
import pandas
filename="data/pima-indians-diabetes.data"
names = ['preg','plas','pres','skin','test','mass','pedi','age','class']
data = pandas.read_csv(filename,names=names)
print(data.shape)

```

