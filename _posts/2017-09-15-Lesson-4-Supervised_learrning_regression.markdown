---
layout: post
title:  "Lesson-5: Supervised Learning -- Regression Analysis"
date:   2017-09-12 1:31:56 +0530
categories: lesson
---

# Supervised Learning -- Regression Analysis

In regression we are trying to predict a continuous output variable -- in contrast to the nominal variables we were predicting in the previous classification examples. 

Let's start with a simple toy example with one feature dimension (explanatory variable) and one target variable. We will create a dataset out of a sine curve with some noise:


```python
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
```


```python
x = np.linspace(-3, 3, 100)   # -3 : start , 3-Stop, 100: number of points
print(x)
```

    [-3.         -2.93939394 -2.87878788 -2.81818182 -2.75757576 -2.6969697
     -2.63636364 -2.57575758 -2.51515152 -2.45454545 -2.39393939 -2.33333333
     -2.27272727 -2.21212121 -2.15151515 -2.09090909 -2.03030303 -1.96969697
     -1.90909091 -1.84848485 -1.78787879 -1.72727273 -1.66666667 -1.60606061
     -1.54545455 -1.48484848 -1.42424242 -1.36363636 -1.3030303  -1.24242424
     -1.18181818 -1.12121212 -1.06060606 -1.         -0.93939394 -0.87878788
     -0.81818182 -0.75757576 -0.6969697  -0.63636364 -0.57575758 -0.51515152
     -0.45454545 -0.39393939 -0.33333333 -0.27272727 -0.21212121 -0.15151515
     -0.09090909 -0.03030303  0.03030303  0.09090909  0.15151515  0.21212121
      0.27272727  0.33333333  0.39393939  0.45454545  0.51515152  0.57575758
      0.63636364  0.6969697   0.75757576  0.81818182  0.87878788  0.93939394
      1.          1.06060606  1.12121212  1.18181818  1.24242424  1.3030303
      1.36363636  1.42424242  1.48484848  1.54545455  1.60606061  1.66666667
      1.72727273  1.78787879  1.84848485  1.90909091  1.96969697  2.03030303
      2.09090909  2.15151515  2.21212121  2.27272727  2.33333333  2.39393939
      2.45454545  2.51515152  2.57575758  2.63636364  2.6969697   2.75757576
      2.81818182  2.87878788  2.93939394  3.        ]



```python
rng = np.random.RandomState(1234)
y = np.sin(4*x) + x + rng.uniform(size=len(x))
```


```python
plt.plot(x, y, 'o');
```


![png](/images/06.Supervised_Learning-Regression-Bernard_files/06.Supervised_Learning-Regression-Bernard_5_0.png)


Linear Regression
=================

The first model that we will introduce is the so-called simple linear regression. Here, we want to fit a line to the data, which 

One of the simplest models again is a linear one, that simply tries to predict the data as lying on a line. One way to find such a line is `LinearRegression` (also known as [*Ordinary Least Squares (OLS)*](https://en.wikipedia.org/wiki/Ordinary_least_squares) regression).
The interface for LinearRegression is exactly the same as for the classifiers before, only that ``y`` now contains float values, instead of classes.

As we remember, the scikit-learn API requires us to provide the target variable (`y`) as a 1-dimensional array; scikit-learn's API expects the samples (`X`) in form a 2-dimensional array -- even though it may only consist of 1 feature. Thus, let us convert the 1-dimensional `x` NumPy array into an `X` array with 2 axes:



```python
print('Before: ', x.shape)
X = x[:, np.newaxis]
print('After: ', X.shape)
```

    Before:  (100,)
    After:  (100, 1)


Again, we start by splitting our dataset into a training (75%) and a test set (25%):


```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
```

Next, we use the learning algorithm implemented in `LinearRegression` to **fit a regression model to the training data**:


```python
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)



After fitting to the training data, we paramerterized a linear regression model with the following values.


```python
print('Weight coefficients: ', regressor.coef_)
print('y-axis intercept: ', regressor.intercept_)
```

    Weight coefficients:  [ 0.90662492]
    y-axis intercept:  0.527378950118


Since our regression model is a linear one, the relationship between the target variable (y) and the feature variable (x) is defined as 

$$y = weight \times x + \text{intercept}$$.

Plugging in the min and max values into thos equation, we can plot the regression fit to our training data:


```python
min_pt = X.min() * regressor.coef_[0] + regressor.intercept_
max_pt = X.max() * regressor.coef_[0] + regressor.intercept_

plt.plot([X.min(), X.max()], [min_pt, max_pt])
plt.plot(X_train, y_train, 'o');
```


![png](/images/06.Supervised_Learning-Regression-Bernard_files/06.Supervised_Learning-Regression-Bernard_16_0.png)


Similar to the estimators for classification, we use the `predict` method to predict the target variable. And we expect these predicted values to fall onto the line that we plotted previously:


```python
y_pred_train = regressor.predict(X_train)
```


```python
plt.plot(X_train, y_train, 'o', label="data")
plt.plot(X_train, y_pred_train, 'o', label="prediction")
plt.plot([X.min(), X.max()], [min_pt, max_pt], label='fit')
plt.legend(loc='best')
```




    <matplotlib.legend.Legend at 0x7f5134f579b0>




![png](/images/06.Supervised_Learning-Regression-Bernard_files/06.Supervised_Learning-Regression-Bernard_19_1.png)


As we can see in the plot above, the line is able to capture the general slope of the data, but not many details.

Next, let's try the test set:


```python
y_pred_test = regressor.predict(X_test)
```


```python
plt.plot(X_test, y_test, 'o', label="data")
plt.plot(X_test, y_pred_test, 'o', label="prediction")
plt.plot([X.min(), X.max()], [min_pt, max_pt], label='fit')
plt.legend(loc='best');
```


![png](/images/06.Supervised_Learning-Regression-Bernard_files/06.Supervised_Learning-Regression-Bernard_23_0.png)


Again, scikit-learn provides an easy way to evaluate the prediction quantitatively using the ``score`` method. For regression tasks, this is the R<sup>2</sup> score. Another popular way would be the Mean Squared Error (MSE). As its name implies, the MSE is simply the average squared difference over the predicted and actual target values

$$MSE = \frac{1}{n} \sum_{i=1}^{n} (\text{predicted}_i - \text{true}_i)^2$$


```python
regressor.score(X_test, y_test)
```




    0.78094571218071562



KNeighborsRegression
=======================
As for classification, we can also use a neighbor based method for regression. We can simply take the output of the nearest point, or we could average several nearest points. This method is less popular for regression than for classification, but still a good baseline.


```python
from sklearn.neighbors import KNeighborsRegressor
kneighbor_regression = KNeighborsRegressor(n_neighbors=1)
kneighbor_regression.fit(X_train, y_train)
```




    KNeighborsRegressor(algorithm='auto', leaf_size=30, metric='minkowski',
              metric_params=None, n_jobs=1, n_neighbors=1, p=2,
              weights='uniform')



Again, let us look at the behavior on training and test set:


```python
y_pred_train = kneighbor_regression.predict(X_train)

plt.plot(X_train, y_train, 'o', label="data", markersize=10)
plt.plot(X_train, y_pred_train, 's', label="prediction", markersize=4)
plt.legend(loc='best');
```


![png](/images/06.Supervised_Learning-Regression-Bernard_files/06.Supervised_Learning-Regression-Bernard_29_0.png)


On the training set, we do a perfect job: each point is its own nearest neighbor!


```python
y_pred_test = kneighbor_regression.predict(X_test)

plt.plot(X_test, y_test, 'o', label="data", markersize=8)
plt.plot(X_test, y_pred_test, 's', label="prediction", markersize=4)
plt.legend(loc='best');
```


![png](/images/06.Supervised_Learning-Regression-Bernard_files/06.Supervised_Learning-Regression-Bernard_31_0.png)



```python
kneighbor_regression.score(X_test, y_test)
```




    0.93296629159871636



# Compare the KNeighborsRegressor and LinearRegression on the boston housing dataset.


```python
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


boston = load_boston()
X = boston.data
y = boston.target

print('X.shape:', X.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.25,
                                                    random_state=42)

linreg = LinearRegression()
knnreg = KNeighborsRegressor(n_neighbors=1)

linreg.fit(X_train, y_train)
print('Linear Regression Train/Test: %.3f/%.3f' %
      (linreg.score(X_train, y_train),
       linreg.score(X_test, y_test)))

knnreg.fit(X_train, y_train)
print('KNeighborsRegressor Train/Test: %.3f/%.3f' %
      (knnreg.score(X_train, y_train),
       knnreg.score(X_test, y_test)))

```

    X.shape: (506, 13)
    Linear Regression Train/Test: 0.748/0.684
    KNeighborsRegressor Train/Test: 1.000/0.474

 References:

The content in this post is largely taken from the github repository [https://github.com/amueller/scipy-2017-sklearn](https://github.com/amueller/scipy-2017-sklearn)
