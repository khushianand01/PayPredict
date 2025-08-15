#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#ssc_p is 10th grade percentage
#ssc_b is board for 10th grade
#hsc_p is 12th grade percentage
#hsc_b is board for 12th grade
#hsc_s is field of study in 12th 
#degree_p is percentage of marks obtained in undergraduate degree
#degree_t is field of study in undergraduate degree
#etest_p percentage of marks obtained in employment test


# In[2]:


data = pd.read_csv('Placement_Data_Full_Class.csv')
data


# In[3]:


data.head()


# In[4]:


data.isna().sum()


# In[5]:


data.shape


# In[6]:


data.dtypes


# In[7]:


data.duplicated().sum()


# In[8]:


data.info()


# In[9]:


data.describe()


# In[10]:


data.columns


# In[11]:


data.nunique()


# In[12]:


data['salary'].mean


# In[13]:


duplicated_rows = data[data.duplicated(subset=['ssc_p'])]
print(duplicated_rows)


# In[14]:


m =data['degree_t'].value_counts()
m


# In[15]:


m =data['status'].value_counts()
m


# In[16]:


data.keys()


# In[17]:


from pandas.api.types import is_numeric_dtype

for col in data.columns:
    if is_numeric_dtype(data[col]):
        print('\nColumn: %s' % col)
        print('\tMean = %.2f' % data[col].mean())
        print('\tStandard deviation = %.2f' % data[col].std())
        print('\tMinimum = %.2f' % data[col].min())
        print('\tMaximum = %.2f' % data[col].max())


# In[18]:


data.corr()


# In[19]:


data.skew()


# In[20]:


plt.hist(data['ssc_p'],bins=20)
plt.title("ssc percentage distribution")
plt.xlabel("percentage")
plt.ylabel("ssc_p")
plt.show()


# In[21]:


plt.hist(data['hsc_p'],bins=20)
plt.title("hsc percentage distribution")
plt.xlabel("percentage")
plt.ylabel("hsc_p")
plt.show()


# In[22]:


data["gender"].value_counts().plot(kind = 'pie',autopct='%.2f')
plt.show()


# In[23]:


data["status"].value_counts().plot(kind = 'pie',autopct='%.2f')
plt.show()


# In[24]:


sns.countplot(x='degree_t', data=data)
plt.show()


# In[25]:


sns.barplot(x=data['gender'],y=data['salary'])


# In[26]:


data["degree_t"].value_counts().plot(kind = 'bar')
plt.show()


# In[27]:


sns.scatterplot(x=data['ssc_p'], y=data['hsc_p'],hue=data['status'])


# In[28]:


sns.scatterplot(x=data['etest_p'],y=data['mba_p'], hue=data['status'])


# In[29]:


sns.boxplot(x=data['specialisation'],y=data['salary'])


# In[30]:


sns.heatmap(pd.crosstab(data['status'],data['workex']))


# In[31]:


data["status"].value_counts().plot(kind = 'bar')
plt.show()


# In[32]:


sns.distplot(data[data['gender']=='M']['salary'],hist=False)
sns.distplot(data[data['gender']=='F']['salary'],hist=False)


# In[33]:


sns.boxplot(x=data['gender'],y=data['salary'])


# In[34]:


columns_of_interest = ['ssc_p', 'ssc_b', 'hsc_p', 'hsc_b', 'hsc_s', 'degree_p', 'degree_t', 'mba_p', 'status', 'salary',]  # Specify the columns you're interested in

# Create a DataFrame containing only the columns of interest
subset_data = data[columns_of_interest]

# Plot the heatmap
plt.figure(figsize=(7, 5))
sns.heatmap(subset_data.corr(), annot=True)
plt.show()


# In[35]:


columns_of_interest = ['ssc_p', 'hsc_p', 'degree_p', 'mba_p', 'degree_t']  # Specify the columns you're interested in

# Create a DataFrame containing only the columns of interest
subset_data = data[columns_of_interest]
sns.pairplot(subset_data, hue="degree_t")
plt.show()


# In[36]:


columns_of_interest = ['ssc_p', 'hsc_p', 'degree_p', 'mba_p', 'status']  # Specify the columns you're interested in

# Create a DataFrame containing only the columns of interest
subset_data = data[columns_of_interest]
sns.pairplot(subset_data, hue="status")
plt.show()


# In[37]:


placed_data = data[data['status'] == 'Placed']
placed_data


# In[38]:


Not_placed_data = data[data['status'] == 'Not Placed']
Not_placed_data


# In[39]:


plt.figure(figsize=(7, 5))
sns.boxplot(x='status', y='ssc_p', data=data)
plt.show()

plt.figure(figsize=(7, 5))
sns.boxplot(x='status', y='hsc_p', data=data)
plt.show()

plt.figure(figsize=(7, 5))
sns.boxplot(x='status', y='degree_p', data=data)
plt.show()

plt.figure(figsize=(7, 5))
sns.boxplot(x='status', y='mba_p', data=data)
plt.show()


# In[40]:


plt.figure(figsize=(7, 5))
sns.boxplot(x='degree_t', y='ssc_p', data=data)
plt.show()

plt.figure(figsize=(7, 5))
sns.boxplot(x='degree_t', y='hsc_p', data=data)
plt.show()

plt.figure(figsize=(7, 5))
sns.boxplot(x='degree_t', y='degree_p', data=data)
plt.show()

plt.figure(figsize=(7, 5))
sns.boxplot(x='degree_t', y='mba_p', data=data)
plt.show()


# In[41]:


data = pd.read_csv('Placement_Data_Full_Class.csv')


# In[42]:


data


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[43]:


print(train.columns)


# In[ ]:





# In[ ]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


# In[ ]:


train, test = train_test_split(data, test_size=0.3)

print(train.shape)
print(test.shape)


# In[ ]:


x_train = train[['ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'mba_p']]
y_train = train['status']


x_test = test[['ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'mba_p']]
y_test = test['status']


# In[ ]:


data.head()


# In[ ]:


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)


# In[ ]:


X_train


# In[ ]:


X_test


# TRAINING MODEL

# LOGISTIC REGRESSION

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model = LogisticRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print('The accuracy of the Logistic Regression is', accuracy)


# In[ ]:


from sklearn.metrics import f1_score
y_pred = model.predict(X_test)
f1 = f1_score(y_test, y_pred, average='micro')
print("F1 score:", f1)


# KNN

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

classifier = KNeighborsClassifier(n_neighbors=2)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy is:", accuracy)


# SUPPORT VECTOR MACHINE

# In[ ]:


from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

classifier = SVC()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy is:", accuracy)


# TESTING

# TAKING USER INPUT

# In[ ]:


ssc_p=float(input("Enter value of 'ssc_p': "))
hsc_p=float(input("Enter value of 'hsc_p': "))
degree_p=float(input("Enter value of 'degree_p': "))
etest_p=float(input("Enter value of 'etest_p': "))
mba_p=float(input("Enter value of 'mba_p': "))


# OUTPUT FOR KNN

# In[ ]:


model=KNeighborsClassifier( n_neighbors=8,metric='manhattan')

model.fit(X_train,y_train)

test_list = [ssc_p, hsc_p, degree_p, etest_p, mba_p]

test_df = pd.DataFrame(test_list)

test = test_df.transpose()

test_pred = model.predict(test)

if test_pred[0] == 0:
    print("Not Placed")
else:
    print("Placed")


# OUTPUT FOR SUPPORT VECTOR MACHINE

# In[ ]:


from sklearn.svm import SVC
import pandas as pd

model = SVC(kernel='linear', C=1.0)

model.fit(X_train, y_train)

test_list = [ssc_p, hsc_p, degree_p, etest_p, mba_p]
test_df = pd.DataFrame([test_list])

test_pred = model.predict(test_df)

if test_pred[0] == 0:
    print("Not Placed")
else:
    print("Placed")


# OUTPUT FOR LOGISTIC REGRESSION

# In[ ]:


from sklearn.linear_model import LogisticRegression
import pandas as pd

model = LogisticRegression()

model.fit(X_train, y_train)

test_list = [ssc_p, hsc_p, degree_p, etest_p, mba_p]
test_df = pd.DataFrame([test_list])

test_pred = model.predict(test_df)

if test_pred[0] == 0:
    print("Not Placed")
else:
    print("Placed")

