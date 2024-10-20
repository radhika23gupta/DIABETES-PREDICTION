#!/usr/bin/env python
# coding: utf-8

# # Final RBL IMPLEMENTATION

# # Importing Necessary Libraries

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import warnings 
warnings.filterwarnings('ignore')
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # Loading Datasets

# In[2]:


df=pd.read_csv("C:\\Users\\Radhika Gupta\\Desktop\\RBL\\diabetes.csv")


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.columns


# In[6]:


df.dtypes


# In[7]:


df.info()


# In[8]:


df.describe()


# # Data Cleaning

# In[9]:


#dropping duplicate values - checking if there are any duplicate rows and dropping if any
df=df.drop_duplicates()


# In[10]:


#check for missing values, count them and print the sum for every column
df.isnull().sum() #conclusion :- there are no null values in this dataset


# In[11]:


#checking for 0 values in 5 columns , Age & DiabetesPedigreeFunction do not have have minimum 0 value so no need to replace , also no. of pregnancies as 0 is possible as observed in df.describe
print(df[df['BloodPressure']==0].shape[0])
print(df[df['Glucose']==0].shape[0])
print(df[df['SkinThickness']==0].shape[0])
print(df[df['Insulin']==0].shape[0])
print(df[df['BMI']==0].shape[0])#replacing 0 values with median of that column


# In[12]:


df['Glucose']=df['Glucose'].replace(0,df['Glucose'].mean())#normal distribution
df['BloodPressure']=df['BloodPressure'].replace(0,df['BloodPressure'].mean())#normal distribution
df['SkinThickness']=df['SkinThickness'].replace(0,df['SkinThickness'].median())#skewed distribution
df['Insulin']=df['Insulin'].replace(0,df['Insulin'].median())#skewed distribution
df['BMI']=df['BMI'].replace(0,df['BMI'].median())#skewed distribution


# # Data Visualization

# In[13]:


sns.countplot(x='Outcome', data=df)
plt.show()


# In[14]:


#histogram for each  feature
df.hist(bins=10,figsize=(10,10))
plt.show()


# In[15]:


plt.figure(figsize=(16,12))
sns.set_style(style='whitegrid')
plt.subplot(3,3,1)
sns.boxplot(x='Glucose',data=df)
plt.subplot(3,3,2)
sns.boxplot(x='BloodPressure',data=df)
plt.subplot(3,3,3)
sns.boxplot(x='Insulin',data=df)
plt.subplot(3,3,4)
sns.boxplot(x='BMI',data=df)
plt.subplot(3,3,5)
sns.boxplot(x='Age',data=df)
plt.subplot(3,3,6)
sns.boxplot(x='SkinThickness',data=df)
plt.subplot(3,3,7)
sns.boxplot(x='Pregnancies',data=df)
plt.subplot(3,3,8)
sns.boxplot(x='DiabetesPedigreeFunction',data=df)


# In[16]:


from pandas.plotting import scatter_matrix
scatter_matrix(df,figsize=(20,20));
# we can come to various conclusion looking at these plots for example  if you observe 5th plot in pregnancies with insulin, you can conclude that women with higher number of pregnancies have lower insulin


# # Feature Selection

# In[17]:


corrmat=df.corr()
sns.heatmap(corrmat, annot=True)


# In[18]:


df_selected=df.drop(['BloodPressure','Insulin','DiabetesPedigreeFunction'],axis='columns')


# # Handling Outliers

# In[19]:


from sklearn.preprocessing import QuantileTransformer
x=df_selected
quantile  = QuantileTransformer()
X = quantile.fit_transform(x)
df_new=quantile.transform(X)
df_new=pd.DataFrame(X)
df_new.columns =['Pregnancies', 'Glucose','SkinThickness','BMI','Age','Outcome']
df_new.head()


# In[20]:


plt.figure(figsize=(16,12))
sns.set_style(style='whitegrid')
plt.subplot(3,3,1)
sns.boxplot(x=df_new['Glucose'],data=df_new)
plt.subplot(3,3,2)
sns.boxplot(x=df_new['BMI'],data=df_new)
plt.subplot(3,3,3)
sns.boxplot(x=df_new['Pregnancies'],data=df_new)
plt.subplot(3,3,4)
sns.boxplot(x=df_new['Age'],data=df_new)
plt.subplot(3,3,5)
sns.boxplot(x=df_new['SkinThickness'],data=df_new)


# # Split the Data Frame into X and y

# In[21]:


target_name='Outcome'
y= df_new[target_name]#given predictions - training data 
X=df_new.drop(target_name,axis=1)#dropping the Outcome column and keeping all other columns as X


# In[22]:


X.head() # contains only independent features


# In[23]:


y.head() #contains dependent feature


# # TRAIN TEST SPLIT

# In[24]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.2,random_state=0)#splitting data in 80% train, 20%test


# In[25]:


X_train.shape,y_train.shape


# In[26]:


X_test.shape,y_test.shape


# #  Algorithms Implementation

# # Decision Tree

# In[27]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
dt = DecisionTreeClassifier(random_state=42)


# In[28]:


# Create the parameter grid based on the results of random search 
params = {
    'max_depth': [5, 10, 20,25],
    'min_samples_leaf': [10, 20, 50, 100,120],
    'criterion': ["gini", "entropy"]
}


# In[29]:


grid_search = GridSearchCV(estimator=dt, 
                           param_grid=params, 
                           cv=4, n_jobs=-1, verbose=1, scoring = "accuracy")


# In[30]:


best_model=grid_search.fit(X_train, y_train)


# In[31]:


dt_pred=best_model.predict(X_test)


# In[32]:


print("Classification Report is:\n",classification_report(y_test,dt_pred))
print("\n F1:\n",f1_score(y_test,dt_pred))
print("\n Precision score is:\n",precision_score(y_test,dt_pred))
print("\n Recall score is:\n",recall_score(y_test,dt_pred))
print("\n Confusion Matrix:\n")
sns.heatmap(confusion_matrix(y_test,dt_pred))


# In[33]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Create the Decision Tree classifier
dt = DecisionTreeClassifier(random_state=42)

# Train the Decision Tree classifier
dt.fit(X_train, y_train)

# Predict on the test set
y_pred = dt.predict(X_test)

# Calculate accuracy
dt_accuracy = accuracy_score(y_test, y_pred)
dt_accuracy = round(dt_accuracy, 4)
print("Decision Tree Accuracy:", dt_accuracy)


# # Random Forest 

# In[34]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV


# In[35]:


# define models and parameters
model = RandomForestClassifier()
n_estimators = [1800]
max_features = ['sqrt', 'log2']


# In[36]:


# define grid search
grid = dict(n_estimators=n_estimators,max_features=max_features)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)


# In[37]:


# Create the Random Forest classifier
rf = RandomForestClassifier(random_state=42)

# Train the Random Forest classifier
rf.fit(X_train, y_train)

# Predict on the test set
rf_pred = rf.predict(X_test)

# Print the Classification Report
print("Classification Report is:\n", classification_report(y_test, rf_pred))

# Calculate F1 score
f1 = f1_score(y_test, rf_pred)
print("\n F1 score is:", f1)

# Calculate Precision score
precision = precision_score(y_test, rf_pred)
print("\n Precision score is:", precision)

# Calculate Recall score
recall = recall_score(y_test, rf_pred)
print("\n Recall score is:", recall)

# Plot the Confusion Matrix
confusion_mat = confusion_matrix(y_test, rf_pred)
print("\n Confusion Matrix:\n")
sns.heatmap(confusion_mat)


# In[38]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Create the Random Forest classifier
rf = RandomForestClassifier(random_state=42)

# Train the Random Forest classifier
rf.fit(X_train, y_train)

# Predict on the test set
y_pred = rf.predict(X_test)

# Calculate accuracy
rf_accuracy = accuracy_score(y_test, y_pred)
rf_accuracy = round(rf_accuracy, 4)
print("Random Forest Accuracy:", rf_accuracy)


# # Support Vector Machine

# In[39]:


from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score


# In[40]:


model = SVC()
kernel = ['poly', 'rbf', 'sigmoid']
C = [50, 10, 1.0, 0.1, 0.01]
gamma = ['scale']


# In[41]:


# define grid search
grid = dict(kernel=kernel,C=C,gamma=gamma)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='f1',error_score=0)


# In[42]:


grid_result = grid_search.fit(X, y)


# In[43]:


svm_pred=grid_result.predict(X_test)


# In[44]:


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Create the SVM classifier
svm = SVC(kernel='rbf', gamma='auto', random_state=42)

# Train the SVM classifier
svm.fit(X_train, y_train)

# Predict on the test set
y_pred = svm.predict(X_test)

# Calculate accuracy
svm_accuracy = accuracy_score(y_test, y_pred)
svm_accuracy = round(svm_accuracy, 4)
print("SVM Accuracy:", svm_accuracy)


# # LogisticRegression

# In[45]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, confusion_matrix
import seaborn as sns

# Create the Logistic Regression classifier
lr = LogisticRegression()

# Train the Logistic Regression classifier
lr.fit(X_train, y_train)

# Predict on the test set
lr_pred = lr.predict(X_test)

# Print the Classification Report
print("Classification Report is:\n", classification_report(y_test, lr_pred))

# Calculate F1 score
f1 = f1_score(y_test, lr_pred)
print("\n F1 score is:", f1)

# Calculate Precision score
precision = precision_score(y_test, lr_pred)
print("\n Precision score is:", precision)

# Calculate Recall score
recall = recall_score(y_test, lr_pred)
print("\n Recall score is:", recall)

# Plot the Confusion Matrix
confusion_mat = confusion_matrix(y_test, lr_pred)
print("\n Confusion Matrix:\n")
sns.heatmap(confusion_mat)


# In[46]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Create the Logistic Regression classifier
log_reg = LogisticRegression(random_state=42)

# Train the Logistic Regression classifier
log_reg.fit(X_train, y_train)

# Predict on the test set
y_pred = log_reg.predict(X_test)

# Calculate accuracy
log_reg_accuracy = accuracy_score(y_test, y_pred)
log_reg_accuracy = round(log_reg_accuracy, 4)
print("Logistic Regression Accuracy:", log_reg_accuracy)


# # Comparision result of the algorithms 

# In[53]:


import matplotlib.pyplot as plt

labels = ['SVM','RF','LG','DT']

accuracies = [svm_accuracy,rf_accuracy,log_reg_accuracy,dt_accuracy]

plt.bar(labels, accuracies)
plt.ylim([0, 1])
plt.title('Accuracy Comparison')
plt.xlabel('Algorithm')
plt.ylabel('Accuracy')
# Add text annotations on top of each data point
for i, j in zip(labels, accuracies):
    plt.text(i, j, str(j), ha='center', va='bottom')
plt.show()


# # StackingClassifier

# In[48]:


from sklearn.ensemble import StackingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Create the base classifiers
svm = SVC(kernel='rbf', gamma='auto', random_state=42)
rf = RandomForestClassifier(random_state=42)
dt = DecisionTreeClassifier(random_state=42)
log_reg = LogisticRegression(random_state=42)

# Create the stacking classifier
estimators = [('svm', svm), ('rf', rf), ('dt', dt), ('log_reg', log_reg)]
stacking = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())

# Train the stacking classifier
stacking.fit(X_train, y_train)

# Predict on the test set
y_pred = stacking.predict(X_test)

# Calculate accuracy
stacking_accuracy = accuracy_score(y_test, y_pred)
stacking_accuracy = round(stacking_accuracy, 4)
print("Stacking Accuracy:", stacking_accuracy)


# In[49]:


from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Create the base classifiers
rf = RandomForestClassifier(random_state=42)
gb = GradientBoostingClassifier(random_state=42)
log_reg = LogisticRegression(random_state=42)

# Create the stacking classifier
estimators = [('rf', rf), ('gb', gb), ('log_reg', log_reg)]
stacking = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())

# Train the stacking classifier
stacking.fit(X_train, y_train)

# Predict on the test set
y_pred = stacking.predict(X_test)

# Calculate accuracy
stacking_accuracyf = accuracy_score(y_test, y_pred)
stacking_accuracyf = round(stacking_accuracyf, 4)
print("Stacking Accuracy:", stacking_accuracyf)


# # AdaBoostClassifier

# In[50]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Create the base classifiers
svm = SVC(kernel='rbf', gamma='auto', random_state=42)
rf = RandomForestClassifier(random_state=42)
dt = DecisionTreeClassifier(random_state=42)
log_reg = LogisticRegression(random_state=42)

# Create the AdaBoost classifier
adaboost = AdaBoostClassifier(base_estimator=svm, n_estimators=50, random_state=42)

# Add the base classifiers to the AdaBoost classifier
adaboost.estimators_ = [svm, rf, dt, log_reg]

# # Train the AdaBoost classifier
# adaboost.fit(X_train, y_train)

# # Predict on the test set
# y_pred = adaboost.predict(X_test)

# Calculate accuracy
adaboost_accuracy = accuracy_score(y_test, y_pred)
adaboost_accuracy = round(adaboost_accuracy, 4)
print("AdaBoost Accuracy:", adaboost_accuracy)


# # GradientBoostingClassifier

# In[51]:


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Create the base classifiers
svm = SVC(kernel='rbf', gamma='auto', random_state=42)
rf = RandomForestClassifier(random_state=42)
dt = DecisionTreeClassifier(random_state=42)
log_reg = LogisticRegression(random_state=42)

# Create the GradientBoostingClassifier
gradient_boost = GradientBoostingClassifier(n_estimators=100, random_state=42)

# Set the base estimators for GradientBoostingClassifier
gradient_boost.estimators_ = [svm, rf, dt, log_reg]

# Train the GradientBoostingClassifier
gradient_boost.fit(X_train, y_train)

# Predict on the test set
y_pred = gradient_boost.predict(X_test)

# Calculate accuracy
gradient_boost_accuracy = accuracy_score(y_test, y_pred)
gradient_boost_accuracy = round(gradient_boost_accuracy, 4)
print("GradientBoostingClassifier Accuracy:", gradient_boost_accuracy)


# # Final Comparision

# In[52]:


import matplotlib.pyplot as plt

labels = ['ST','GB','AB']

accuracies = [stacking_accuracy,gradient_boost_accuracy, adaboost_accuracy]

plt.bar(labels, accuracies)
plt.ylim([0, 1])
plt.title('Accuracy Comparison')
plt.xlabel('Algorithm')
plt.ylabel('Accuracy')
# Add text annotations on top of each data point
for i, j in zip(labels, accuracies):
    plt.text(i, j, str(j), ha='center', va='bottom')
plt.show()


# In[ ]:




