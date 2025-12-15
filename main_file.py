#!/usr/bin/env python
# coding: utf-8

# In[90]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import sklearn.model_selection as sms
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# # Problem 9.5

# # Part a)

# In[15]:


np.random.seed(30)
x1 = np.random.uniform(-0.5, 0.5, 500)
x2 = np.random.uniform(-0.5, 0.5, 500)

y = []
for i in range(0,500):
    if (x1[i])**2 - (x2[i])**2 > 0:
        n = 1
    else:
        n = 2
    y.append(n)

df = pd.DataFrame({'X1': x1,'X2':x2})


# # Part b)

# In[16]:


plt.scatter(x1,x2,c=y)


# # Part c) & d)

# In[17]:


model = LogisticRegression(solver='liblinear', random_state=0)
model.fit(df,y)
plt.scatter(x1,x2,c=model.predict(df))


# # Part e) & f)

# In[18]:


new_df = pd.DataFrame({'X1sq': (x1)**2,'X2sq':(x2)**2})
new_model = LogisticRegression(solver='liblinear', random_state=0)
new_model.fit(new_df,y)
plt.scatter(x1,x2,c=new_model.predict(new_df))


# # Part g)

# In[53]:


linearSVCmodel = svm.LinearSVC()
linearSVCmodel.fit(df, y)
plt.scatter(x1,x2,c=linearSVCmodel.predict(df))


# # Part h)

# In[54]:


nonlinearSVCmodel = svm.SVC()
nonlinearSVCmodel.fit(df, y)
plt.scatter(x1,x2,c=nonlinearSVCmodel.predict(df))


# # Problem 9.7

# # Part a)

# In[21]:


data = pd.read_csv(r"G:\BGSU_MSAS\As_a_Student\Spring_2020\CS_7200\Assignment_4\cs7200_sp2020_a04_singh\Auto.csv")


# In[22]:


new_var = []
for i in range(0,data.shape[0]):
    if data.iat[i,0] > np.median(data.mpg):
        n = 1
    else:
        n = 0
    new_var.append(n)

newdata = data.assign(New_Var = new_var)
X = newdata.iloc[:,[0,1,2,3,4,5,6,7]]
Y = newdata.iloc[:,[9]]


# # Part b)

# In[39]:


accuracy = []
for i in np.linspace(1, 20, num=20):
    model = svm.LinearSVC(C=i)
    model.fit(X, Y)
    cv_accuracy_scores_rf = sms.cross_val_score(model, X, Y, cv=5,scoring='accuracy')
    accuracy.append(np.mean(cv_accuracy_scores_rf))

plt.plot(np.linspace(1, 20, num=20),accuracy)
plt.ylabel('Mean 5-Fold Accuracy value')
plt.xlabel('Cost')
plt.title('Linear SVC')


# # Part c)

# # Kernel = 'rbf' with gamma = 'scale'

# In[42]:


accuracy = []
for i in np.linspace(1, 20, num=20):
    model = svm.SVC(C=i,kernel='rbf',gamma='scale')
    model.fit(X, Y)
    cv_accuracy_scores_rf = sms.cross_val_score(model, X, Y, cv=5,scoring='accuracy')
    accuracy.append(np.mean(cv_accuracy_scores_rf))

plt.plot(np.linspace(1, 20, num=20),accuracy)
plt.ylabel('Mean 5-Fold Accuracy value')
plt.xlabel('Cost')
plt.title('Kernel = rbf with gamma = scale')


# # Kernel = 'rbf' with gamma = 'auto'

# In[48]:


accuracy = []
for i in np.linspace(1, 20, num=20):
    model = svm.SVC(C=i,kernel='rbf',gamma='auto')
    model.fit(X, Y)
    cv_accuracy_scores_rf = sms.cross_val_score(model, X, Y, cv=5,scoring='accuracy')
    accuracy.append(np.mean(cv_accuracy_scores_rf))

plt.plot(np.linspace(1, 20, num=20),accuracy)
plt.ylabel('Mean 5-Fold Accuracy value')
plt.xlabel('Cost')
plt.title('Kernel = rbf with gamma = auto')


# # Kernel = 'poly' with degree = 1

# In[49]:


accuracy = []
for i in np.linspace(1, 20, num=20):
    model = svm.SVC(C=i,kernel='poly',degree=1)
    model.fit(X, Y)
    cv_accuracy_scores_rf = sms.cross_val_score(model, X, Y, cv=5,scoring='accuracy')
    accuracy.append(np.mean(cv_accuracy_scores_rf))

plt.plot(np.linspace(1, 20, num=20),accuracy)
plt.ylabel('Mean 5-Fold Accuracy value')
plt.xlabel('Cost')
plt.title('Kernel = poly with degree = 1')


# # Kernel = 'poly' with degree = 2

# In[50]:


accuracy = []
for i in np.linspace(1, 20, num=20):
    model = svm.SVC(C=i,kernel='poly',degree=2)
    model.fit(X, Y)
    cv_accuracy_scores_rf = sms.cross_val_score(model, X, Y, cv=5,scoring='accuracy')
    accuracy.append(np.mean(cv_accuracy_scores_rf))

plt.plot(np.linspace(1, 20, num=20),accuracy)
plt.ylabel('Mean 5-Fold Accuracy value')
plt.xlabel('Cost')
plt.title('Kernel = poly with degree = 2')


# # Kernel = 'poly' with degree = 3

# In[51]:


accuracy = []
for i in np.linspace(1, 20, num=20):
    model = svm.SVC(C=i,kernel='poly',degree=3)
    model.fit(X, Y)
    cv_accuracy_scores_rf = sms.cross_val_score(model, X, Y, cv=5,scoring='accuracy')
    accuracy.append(np.mean(cv_accuracy_scores_rf))

plt.plot(np.linspace(1, 20, num=20),accuracy)
plt.ylabel('Mean 5-Fold Accuracy value')
plt.xlabel('Cost')
plt.title('Kernel = poly with degree = 3')


# # Kernel = 'poly' with degree = 4

# In[52]:


accuracy = []
for i in np.linspace(1, 20, num=20):
    model = svm.SVC(C=i,kernel='poly',degree=4)
    model.fit(X, Y)
    cv_accuracy_scores_rf = sms.cross_val_score(model, X, Y, cv=5,scoring='accuracy')
    accuracy.append(np.mean(cv_accuracy_scores_rf))

plt.plot(np.linspace(1, 20, num=20),accuracy)
plt.ylabel('Mean 5-Fold Accuracy value')
plt.xlabel('Cost')
plt.title('Kernel = poly with degree = 4')


# # Problem 9.8

# # Part a)

# In[123]:


data = pd.read_csv(r"G:\BGSU_MSAS\As_a_Student\Spring_2020\CS_7200\Assignment_4\cs7200_sp2020_a04_singh\OJ.csv")
category = {'Yes': 1,'No': 0}
data.Store7 = [category[item] for item in data.Store7]
X = data.iloc[:,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]]
Y = data.iloc[:,[0]]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.252)


# # Part b) & c)

# In[124]:


model = svm.LinearSVC(C=0.01)
model.fit(X_train, Y_train)

accuracy_score(Y_train, model.predict(X_train))


# In[125]:


accuracy_score(Y_test, model.predict(X_test))


# # Part d)

# In[126]:


accuracy = []
for i in np.linspace(0.01, 10, num=1000):
    model = svm.LinearSVC(C=i)
    model.fit(X, Y)
    cv_accuracy_scores_rf = sms.cross_val_score(model, X, Y, cv=5,scoring='accuracy')
    accuracy.append(np.mean(cv_accuracy_scores_rf))

plt.plot(np.linspace(0.01, 10, num=1000),accuracy)
plt.ylabel('Mean 5-Fold Accuracy value')
plt.xlabel('Cost')


# In[127]:


accuracy = []
for i in np.linspace(3, 4, num=100):
    model = svm.LinearSVC(C=i)
    model.fit(X, Y)
    cv_accuracy_scores_rf = sms.cross_val_score(model, X, Y, cv=5,scoring='accuracy')
    accuracy.append(np.mean(cv_accuracy_scores_rf))

plt.plot(np.linspace(3, 4, num=100),accuracy)
plt.ylabel('Mean 5-Fold Accuracy value')
plt.xlabel('Cost')


# # Part e)

# In[128]:


model = svm.LinearSVC(C=3.38)
model.fit(X_train, Y_train)

accuracy_score(Y_train, model.predict(X_train))


# In[129]:


accuracy_score(Y_test, model.predict(X_test))


# # Part f)

# In[130]:


accuracy = []
for i in np.linspace(0.01, 10, num=1000):
    model = svm.SVC(C=i,kernel='rbf')
    model.fit(X, Y)
    cv_accuracy_scores_rf = sms.cross_val_score(model, X, Y, cv=5,scoring='accuracy')
    accuracy.append(np.mean(cv_accuracy_scores_rf))

plt.plot(np.linspace(0.01, 10, num=1000),accuracy)
plt.ylabel('Mean 5-Fold Accuracy value')
plt.xlabel('Cost')


# In[131]:


model = svm.SVC(kernel='rbf')
model.fit(X_train, Y_train)

accuracy_score(Y_train, model.predict(X_train))


# In[132]:


accuracy_score(Y_test, model.predict(X_test))


# # Part g)

# In[133]:


accuracy = []
for i in np.linspace(0.01, 10, num=1000):
    model = svm.SVC(C=i,kernel='poly',degree=2)
    model.fit(X, Y)
    cv_accuracy_scores_rf = sms.cross_val_score(model, X, Y, cv=5,scoring='accuracy')
    accuracy.append(np.mean(cv_accuracy_scores_rf))

plt.plot(np.linspace(0.01, 10, num=1000),accuracy)
plt.ylabel('Mean 5-Fold Accuracy value')
plt.xlabel('Cost')


# In[134]:


model = svm.SVC(kernel='poly',degree=2)
model.fit(X_train, Y_train)

accuracy_score(Y_train, model.predict(X_train))


# In[135]:


accuracy_score(Y_test, model.predict(X_test))


# In[ ]:




