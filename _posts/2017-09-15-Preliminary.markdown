---
layout: post
title:  "Preliminary : Introduction to Jupyter notebook and Scientific Computing Tools in Python"
date:   2017-09-21 1:31:56 +0530
categories: lesson
---

# Introduction

Machine learning is about extracting knowledge from data. It is a research field at the intersection of `statistics`, `artificial inteligence`, and `computer science`.

# Installation
System configuration we tested on is  `ubuntu 16.04` with `python3`. For easy installation of packages in python, we recommend installing `pip` - a package managment tool for python. 

{% highlight bash %}
sudo apt-get install python3-pip
{% endhighlight %}
we will be using [scikitlearn](http://scikit-learn.org/stable/index.html) an open source machine learning library. Since `scikitlearn` has dependencies on other packages, we also need to install them.

for `Python 2.7` 
{%highlight bash%}
pip install numpy scipy matplotlib ipython scikit-learn pandas
{%endhighlight%}

for `Python 3` 

{%highlight bash%}
pip3 install numpy scipy matplotlib ipython scikit-learn pandas
{%endhighlight%}

# Jupyter Notebook
For interactive python, you can use [jupyter](http://jupyter.org/).

Install Jupyter

{%highlight bash%}
pip install jupyter
{%endhighlight%}

To start `Jupyter`

{%highlight bash%}
jupyter notebook
{%endhighlight%}



Introduction to Jupyter Notebooks and Scientific Computing tools in Python
==================

* You can run a cell by pressing ``[shift] + [Enter]`` or by pressing the "play" button in the menu.

![](/images/ipython_run_cell.png)

* You can get help on a function or object by pressing ``[shift] + [tab]`` after the opening parenthesis ``function(``

![](/images/ipython_help-1.png)

* You can also get help by executing ``function?``

![](/images/ipython_help-2.png)

## Numpy Arrays

Manipulating `numpy` arrays is an important part of doing machine learning
in python.

Numpy is one of the fundamental packages for scientific computing in python. The core functionality of NumPy is the `ndarray` class, a multidimensional(n-d) array. All elements of the arrays must be of the same type. 

Sample usage of NumPy:
{%highlight python%}
# Importing numpy package and import it as np, through out the program we can use np 
import numpy as np
x = np.array([[1,2,3],[4,5,6]])
print(x)
{%endhighlight%}
OUTPUT: 
{%highlight bash%}
[[1 2 3]
 [4 5 6]]
{%endhighlight%}

Another Example:

```python
import numpy as np

# Setting a random seed for reproducibility
rnd = np.random.RandomState(seed=123)

# Generating a random array
X = rnd.uniform(low=0.0, high=1.0, size=(3, 5))  # a 3 x 5 array

print(X)
```

    [[ 0.69646919  0.28613933  0.22685145  0.55131477  0.71946897]
     [ 0.42310646  0.9807642   0.68482974  0.4809319   0.39211752]
     [ 0.34317802  0.72904971  0.43857224  0.0596779   0.39804426]]


# Accessing Elements of Numpy Array

(Note that NumPy arrays use 0-indexing just like other data structures in Python.)

## Get a single element


```python
# (here: an element in the first row and column)
print(X[0, 0])
```

    0.696469185598


## get a row 


```python
# (here: 2nd row)
print(X[1])
```

    [ 0.42310646  0.9807642   0.68482974  0.4809319   0.39211752]


## get a column


```python
# (here: 2nd column)
print(X[:, 1])
```

    [ 0.28613933  0.9807642   0.72904971]


## Transposing an array


```python
print(X.T)
```

    [[ 0.69646919  0.42310646  0.34317802]
     [ 0.28613933  0.9807642   0.72904971]
     [ 0.22685145  0.68482974  0.43857224]
     [ 0.55131477  0.4809319   0.0596779 ]
     [ 0.71946897  0.39211752  0.39804426]]


$$\begin{bmatrix}
    1 & 2 & 3 & 4 \\
    5 & 6 & 7 & 8
\end{bmatrix}^T
= 
\begin{bmatrix}
    1 & 5 \\
    2 & 6 \\
    3 & 7 \\
    4 & 8
\end{bmatrix}
$$



## Creating a row vector


```python
# of evenly spaced numbers over a specified interval.
y = np.linspace(0, 12, 5)
# 1st parameter: 1st element, 2nd parameter: last element, 3rd parameter: number of elements.
print(y)
```

    [  0.   3.   6.   9.  12.]


## Turning the row vector into a column vector


```python
print(y[:, np.newaxis])
```

    [[  0.]
     [  3.]
     [  6.]
     [  9.]
     [ 12.]]


## Getting the shape or reshaping an array


```python
# Generating a random array
rnd = np.random.RandomState(seed=123)
X = rnd.uniform(low=0.0, high=1.0, size=(3, 5))  # a 3 x 5 array
print(X.shape)
print(X.reshape(5, 3).shape)
print(X.shape)
```

    (3, 5)
    (5, 3)
    (3, 5)


There is much, much more to know, but these few operations are fundamental to what we'll
do during this tutorial.

## SciPy Sparse Matrices

In some machine learning tasks, especially those associated with textual analysis, the data may be mostly zeros.  Storing all these zeros is very inefficient, and representing in a way that only contains the "non-zero" values can be much more efficient.  We can create and manipulate sparse matrices as follows:


```python
from scipy import sparse
# Create a random array with a lot of zeros
rnd = np.random.RandomState(seed=123)
X = rnd.uniform(low=0.0, high=1.0, size=(10, 5))
print(X)
```

    [[ 0.69646919  0.28613933  0.22685145  0.55131477  0.71946897]
     [ 0.42310646  0.9807642   0.68482974  0.4809319   0.39211752]
     [ 0.34317802  0.72904971  0.43857224  0.0596779   0.39804426]
     [ 0.73799541  0.18249173  0.17545176  0.53155137  0.53182759]
     [ 0.63440096  0.84943179  0.72445532  0.61102351  0.72244338]
     [ 0.32295891  0.36178866  0.22826323  0.29371405  0.63097612]
     [ 0.09210494  0.43370117  0.43086276  0.4936851   0.42583029]
     [ 0.31226122  0.42635131  0.89338916  0.94416002  0.50183668]
     [ 0.62395295  0.1156184   0.31728548  0.41482621  0.86630916]
     [ 0.25045537  0.48303426  0.98555979  0.51948512  0.61289453]]



```python
# set the majority of elements to zero
X[X < 0.7] = 0  #setting all elements with value < 0.7 to 0
print(X)
```

    [[ 0.          0.          0.          0.          0.71946897]
     [ 0.          0.9807642   0.          0.          0.        ]
     [ 0.          0.72904971  0.          0.          0.        ]
     [ 0.73799541  0.          0.          0.          0.        ]
     [ 0.          0.84943179  0.72445532  0.          0.72244338]
     [ 0.          0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.          0.        ]
     [ 0.          0.          0.89338916  0.94416002  0.        ]
     [ 0.          0.          0.          0.          0.86630916]
     [ 0.          0.          0.98555979  0.          0.        ]]



```python
# turn X into a CSR (Compressed-Sparse-Row) matrix
X_csr = sparse.csr_matrix(X)
print(X_csr)
```

      (0, 4)	0.719468969786
      (1, 1)	0.980764198385
      (2, 1)	0.729049707384
      (3, 0)	0.737995405732
      (4, 1)	0.849431794078
      (4, 2)	0.724455324861
      (4, 4)	0.72244338257
      (7, 2)	0.893389163117
      (7, 3)	0.944160018204
      (8, 4)	0.866309157883
      (9, 2)	0.985559785611



```python
# Converting the sparse matrix to a dense array
print(X_csr.toarray())
```

    [[ 0.          0.          0.          0.          0.71946897]
     [ 0.          0.9807642   0.          0.          0.        ]
     [ 0.          0.72904971  0.          0.          0.        ]
     [ 0.73799541  0.          0.          0.          0.        ]
     [ 0.          0.84943179  0.72445532  0.          0.72244338]
     [ 0.          0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.          0.        ]
     [ 0.          0.          0.89338916  0.94416002  0.        ]
     [ 0.          0.          0.          0.          0.86630916]
     [ 0.          0.          0.98555979  0.          0.        ]]


##  visualization of data using matplotlib

The most common tool for this in Python is [`matplotlib`](http://matplotlib.org).  It is an extremely flexible package, and we will go over some basics here.

Since we are using Jupyter notebooks, let us use one of IPython's convenient built-in "[magic functions](https://ipython.org/ipython-doc/3/interactive/magics.html)", the "matoplotlib inline" mode, which will draw the plots directly inside the notebook.


```python
%matplotlib inline
```


```python
import matplotlib.pyplot as plt
```


```python
# Plotting a line
x = np.linspace(0, 10, 100)
plt.plot(x, np.sin(x));
```


![png](/images/02.Scientific_Computing_Tools_in_Python-Bernard_files/02.Scientific_Computing_Tools_in_Python-Bernard_32_0.png)



```python
# Scatter-plot points
x = np.random.normal(size=500)
y = np.random.normal(size=500)
plt.scatter(x, y);
```


![png](/images/02.Scientific_Computing_Tools_in_Python-Bernard_files/02.Scientific_Computing_Tools_in_Python-Bernard_33_0.png)



```python
# Showing images using imshow
# - note that origin is at the top-left by default!

x = np.linspace(1, 12, 100)
y = x[:, np.newaxis]

im = y * np.sin(x) * np.cos(y)
print(im.shape)

plt.imshow(im);
```

    (100, 100)



![png](/images/02.Scientific_Computing_Tools_in_Python-Bernard_files/02.Scientific_Computing_Tools_in_Python-Bernard_34_1.png)



```python
# Contour plots 
# - note that origin here is at the bottom-left by default!
plt.contour(im);
```


![png](/images/02.Scientific_Computing_Tools_in_Python-Bernard_files/02.Scientific_Computing_Tools_in_Python-Bernard_35_0.png)



```python
# 3D plotting
from mpl_toolkits.mplot3d import Axes3D
ax = plt.axes(projection='3d')
xgrid, ygrid = np.meshgrid(x, y.ravel())
ax.plot_surface(xgrid, ygrid, im, cmap=plt.cm.viridis, cstride=2, rstride=2, linewidth=0);
```


![png](/images/02.Scientific_Computing_Tools_in_Python-Bernard_files/02.Scientific_Computing_Tools_in_Python-Bernard_36_0.png)


There are many, many more plot types available.  One useful way to explore these is by
looking at the [matplotlib gallery](http://matplotlib.org/gallery.html).

References:

The content in this post is largely taken from the github repository [https://github.com/amueller/scipy-2017-sklearn](https://github.com/amueller/scipy-2017-sklearn)
