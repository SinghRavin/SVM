# Support Vector Machines

**Problem 9.5**

Part i)

Here we can see that non-linear kernel SVMs are good at finding non-linear boundaries.

We could get similar results with logistic regression by adding higher order terms but this process is more complicated and time consuming.

**Problem 9.7**

Part b)

In this scenario, the 'best' cost value is found to be 2.5 and its corresponding accuracy was above 90%. There is no clear trend between the model's accuracy and cost value.

Part c) & d)

The cost values were explored within the range of 1 to 20.

The following table reveals all the results found,

|Model|'Best' Cost value (approx.)|
| --- | --- |
|Kernel = 'rbf', Gamma = 'scale'|8.75|
|kernel = 'rbf', Gamma = 'auto'|No unique best value exists|
|Kernel = 'poly', Degree = 1|2 or 3.75|
|Kernel = 'poly', Degree = 2|8.75|
|Kernel = 'poly', Degree = 3|3.75|
|Kernel = 'poly', Degree = 4|1.25|


**Problem 9.8**

The following table reveals all the results found,

|Model|'Best' Cost value (approx.)|Training accuracy score|Testing accuracy score|
| --- | --- | --- | --- |
|Linear SVC|3.38|81.25%|77.78%|
|kernel = 'rbf', Gamma = default|No unique best value exists|61.50%|59.63%|
|Kernel = 'poly', Degree = 2|No unique best value exists|61.50%|59.63%|

We can see that the two models (rbf with default gamma, and poly with degree 2) have the same results.

Part h)

Overall the Linear SVC gives the best results with a unique 'best' value for c of 3.38 and corresponding accuracy of around 80%.

*Useful readings:*

Gareth James, Daniela Witten, Trevor Hastie, and Robert Tibshirani. 2014. An Introduction to Statistical Learning: with Applications in R. Springer Publishing Company, Incorporated.
