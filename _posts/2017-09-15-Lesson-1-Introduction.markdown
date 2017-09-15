---
layout: post
title:  "Lesson-1: Installation"
date:   2017-09-15 3:31:56 +0530
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
