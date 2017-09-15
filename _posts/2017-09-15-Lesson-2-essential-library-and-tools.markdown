---
layout: post
title:  "Lesson-2: Essential Libraries and Tools"
date:   2017-09-15 2:31:56 +0530
categories: lesson
---
# Numpy

Numpy is one of the fundamental packages for scientific computing in python. The core functionality of NumPy is the `ndarray` class, a multidimensional(n-d) array. All elements of the arrays must be of the same type. 

Sample usage of NumPy:
{%highlight python%}
import numpy as np
x = np.array([[1,2,3],[4,5,6]])
print(x)
{%endhighlight%}
OUTPUT: 
{%highlight bash%}
[[1 2 3]
 [4 5 6]]
{%endhighlight%}

# SciPy

SciPy is a collection of functions for scientific computing in Python. functionality like advanced linear algebra routines, mathematical function optimization, signal processing, statistical distributions etc. 

Sample usage of SciPy:



```python
import numpy as np
from scipy import sparse
# Create a 2D NumPy array with diagonal of ones, and zeros everywhere else
eye = np.eye(4)
print(eye)
```

    [[ 1.  0.  0.  0.]
     [ 0.  1.  0.  0.]
     [ 0.  0.  1.  0.]
     [ 0.  0.  0.  1.]]


[click [numpy.eye](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.eye.html) for more information]


```python
#Convert the NumPy array to a SciPy sparse matrix in CSR Format
#only the Non-Zero entries are stored
sparse_matrix = sparse.csr_matrix(eye)
print(sparse_matrix)
```

      (0, 0)	1.0
      (1, 1)	1.0
      (2, 2)	1.0
      (3, 3)	1.0


[click [CSR format](http://www.scipy-lectures.org/advanced/scipy_sparse/csr_matrix.html) for more information]

# Matplotlib

`matplotlib` is the primary scientific plotting library in Python.

Sample usage: 


```python
%matplotlib inline
import matplotlib.pyplot as plt
#Generate a sequence of numbers from -10 to 10 with 100 steps in between
x = np.linspace(-10,10,100)
# Create a second array using sine
y = np.sin(x)
# The plot function  makes a line chart of one array against another
plt.plot(x,y,marker="x")
```
[np.linspace](https://docs.scipy.org/doc/numpy/reference/generated/numpy.linspace.html) for more information

OUTPUT:

![png](/images/Lesson1_4_1.png)

`Simple line plot of the sine function using matplotlib`

# Pandas

Pandas is a python library for data wrangling and analysis.

Sample usage:


```python
import pandas as pd
# Create a simple dataset of people
data = {'Name': ["John","Anna","Peter","Linda","Ajay"],
       'Location':["New York","Paris","Berlin","London","New Delhi"],
       'Age':[23,25,26,21,34]}
data_pandas = pd.DataFrame(data)
display(data_pandas)
```


<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Location</th>
      <th>Name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>23</td>
      <td>New York</td>
      <td>John</td>
    </tr>
    <tr>
      <th>1</th>
      <td>25</td>
      <td>Paris</td>
      <td>Anna</td>
    </tr>
    <tr>
      <th>2</th>
      <td>26</td>
      <td>Berlin</td>
      <td>Peter</td>
    </tr>
    <tr>
      <th>3</th>
      <td>21</td>
      <td>London</td>
      <td>Linda</td>
    </tr>
    <tr>
      <th>4</th>
      <td>34</td>
      <td>New Delhi</td>
      <td>Ajay</td>
    </tr>
  </tbody>
</table>
</div>


### Some Query operation


```python
# Select all rows that have an age column greater than 30
display(data_pandas[data_pandas.Age > 30])
```


<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Location</th>
      <th>Name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>34</td>
      <td>New Delhi</td>
      <td>Ajay</td>
    </tr>
  </tbody>
</table>
</div>


