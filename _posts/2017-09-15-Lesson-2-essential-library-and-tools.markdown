---
layout: post
title:  "Lesson-2: Essential Libraries and Tools"
date:   2017-09-15 16:31:56 +0530
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
