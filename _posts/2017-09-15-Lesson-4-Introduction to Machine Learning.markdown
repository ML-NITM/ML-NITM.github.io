---
layout: post
title:  "Lesson-4:Introduction to Machine Learning"
date:   2017-09-12 1:31:56 +0530
categories: lesson
---
## Machine learning: the problem setting
A Machine learning problem considers a set of `n samples` of data and then tries to predict properties of `unknown data`. 

Key types of Machine Learning problems 

# Supervised Learning

Learn to predict target values from labelled data

This problem can be either:

- `classification:`samples belong to two or more classes and we want to learn from already labeled data how to predict the class of unlabeled data. `Target values are discrete classes`

- `Regression:` if the desired output consists of one or more continuous variables, then the task is called regression. `Target values are continuous values` 

![png](/images/supervise-example.png)
[photo from coursera: applied machine learning using python](https://www.coursera.org/learn/python-machine-learning/lecture/hrHXm/key-concepts-in-machine-learning)


# Unsupervised Learning
Here the training data consists of a set of `input vectors` `x` without any corresponding `target` values. The goal in such problems may be to discover groups of similar examples within the data, where it is called [`clustering`](https://en.wikipedia.org/wiki/Cluster_analysis). `Find structure in unlabeled data `. 

- Find groups of similar instances in the data(clustering). 

- Finding unusual patterns (outlier detection)

`Finding useful structure or knowledge in data when no labels are available.`

## Training set and testing set
Machine learning is about learning some properties of a data set and applying them to new data. This is why a common practice in machine learning to evaluate an algorithm is to split the data at hand into two sets, one that we call the `training set` on which we learn data properties and one that we call the `testing set` on which we test these properties.

## Machine Learning Workflow

![png](/images/machinel-workflow.png)

# K- Nearest Neighbor Classifier algorithm

 Given a training set `X_train` with labels `y_train`, and given a new instance `x_test` to be classified: 
- Find the most similar instance ( lets call these `X_NN`) to `X_test` that are in `X_train`
- Get the labels `y_NN` for the instances in `X_NN`
- Predict the label for `X_test` by combining the labels `y_NN` example: simple majority vote.

A Nearest Neighbor algorithm needs 4 things 
- A distance metric (typically Euclidean)
- How many `nearest` neighbor to look at? i.e, `k`
- Optional weighting function on the neighbor points
- How to aggregate the classes of the neighbor points example: simple majority vote


# Import required modules and data file


```python
%matplotlib notebook
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
```

# Load data file


```python
iris = load_iris()
```

# Create Train-Test Split


```python
iris_X = iris.data
iris_y = iris.target
np.random.seed(0)
indices = np.random.permutation(len(iris_X))
iris_X_train = iris_X[indices[:-10]]
iris_y_train = iris_y[indices[:-10]]
iris_X_test  = iris_X[indices[-10:]]
iris_y_test  = iris_y[indices[-10:]]
```

# Create and fit a nearest-neighbor classifier


```python
knn = KNeighborsClassifier()
knn.fit(iris_X_train, iris_y_train) 
```




    KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
               metric_params=None, n_jobs=1, n_neighbors=5, p=2,
               weights='uniform')



# Make prediction


```python
knn.predict(iris_X_test)
```




    array([1, 2, 1, 0, 0, 0, 2, 1, 2, 0])



# Calculate the score


```python
knn.score(iris_X_test,iris_y_test)
```




    0.90000000000000002



## References
[http://scikit-learn.org/stable/tutorial/basic/tutorial.html#machine-learning-the-problem-setting]

[http://scikit-learn.org/stable/tutorial/basic/tutorial.html#machine-learning-the-problem-setting]: http://scikit-learn.org/stable/tutorial/basic/tutorial.html#machine-learning-the-problem-setting