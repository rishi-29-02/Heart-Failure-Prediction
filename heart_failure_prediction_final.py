import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import RFE

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import pickle

import warnings
warnings.filterwarnings('ignore')

"""## Load and read the dataset"""


data = pd.read_csv('heart_failure_clinical_records_dataset.csv')
print('Data read successfully')

data_1 = data.copy()

print(data.head())

print(data.shape)

"""The dataset contains 299 instances and 13 features."""

print(data.info())

print(data.describe(include='all'))

"""The dataset doesn't contain any missing values.

The dataset contains the data of people of age more than 39 years.
"""

print(data.corr())

"""## Data Visulaization"""

plt.figure(figsize=(14, 14))
sns.heatmap(data.corr(), annot=True)
#plt.show()

fig = plt.figure(figsize = (20,20))
ax = fig.gca() #get current axis
data.hist(ax=ax)
#plt.show()

fig = plt.figure(figsize=(20, 20))
ax = fig.gca()
data.plot(kind='density', subplots=True, layout=(4, 4), sharex=False, ax=ax)
#plt.show()

fig = plt.figure(figsize=(20, 20))
ax = fig.gca()
data.plot(kind='box', subplots=True, layout=(4, 4), sharex=False, ax=ax)
#plt.show()

"""**Age**"""

bins = [40, 60, 80, np.nan]
labels = ['Adults', 'Senior', 'Super_Senior']

data['age_group'] = pd.cut(data['age'], bins=bins, labels=labels)
data.head()

sns.barplot(x='age_group', y='DEATH_EVENT', data=data)
#plt.show()

print('Percentage of Super Senior people lose their life :', data['DEATH_EVENT'][data['age_group']=='Super_Senior'].value_counts(normalize=True)[1]*100)

"""People whose age is more than 80 yrs are more prone to heart failure.

**Anaemia: Decrease of red blood cells or hemoglobin**
"""

sns.barplot(x='anaemia', y='DEATH_EVENT', data=data)
#plt.show()

"""People who have Anaemia are more prone to heart Failure.

**creatinine_phosphokinase : Level of the CPK enzyme in the blood**

Total CPK normal values: 10 to 120 micrograms per liter (mcg/L)
"""

bins = [10, 120, np.nan]
labels = ['Normal','Abnormal']
data['creatinine_phosphokinase_group'] = pd.cut(data['creatinine_phosphokinase'], bins=bins, labels=labels)
print(data.head())

sns.barplot(x='creatinine_phosphokinase_group', y='DEATH_EVENT', data=data)
#plt.show()

"""People who have abnormal level of the CPK enzyme in the blood are more prone to heart Failure.

**Diabetes**
"""

sns.barplot(x='diabetes', y='DEATH_EVENT', data=data)
#plt.show()

print(data['DEATH_EVENT'][data['diabetes']==1].value_counts(normalize=True))

print(data['DEATH_EVENT'][data['diabetes']==0].value_counts(normalize=True))

"""Having diabetes doesn't matter to the heart failure.

Moreover from the heatmap we get that the correlation between Death event and diabetes is very less i.e -0.001943

**ejection_fraction : Percentage of blood leaving the heart at each contraction**
"""

bins = [0, 41, 50, 70, np.nan]
labels = ['too low', 'borderline', 'Normal', 'high']
data['ejection_fraction_category'] = pd.cut(data['ejection_fraction'], bins=bins, labels=labels)
print(data.head())

data['ejection_fraction_category'].value_counts()

sns.barplot(x='ejection_fraction_category', y='DEATH_EVENT', data=data)
#plt.show()

data['DEATH_EVENT'][data['ejection_fraction_category']=='too low'].value_counts(normalize=True)*100

"""If person's ejection_fraction is in the too low category then they have more chances of Heart Failure.

**high_blood_pressure**
"""

sns.barplot(x='high_blood_pressure', y='DEATH_EVENT', data=data)
plt.show()

print('Percentage of people resulted in Heart Failure having high blood pressure : ', 
      data['DEATH_EVENT'][data['high_blood_pressure']==1].value_counts(normalize=True)[1]*100)

"""The person having high blood pressure is more to heart failure.

**platelets : Platelets in the blood (kiloplatelets/mL)**
"""

bins =[0, 150000, 250000, np.nan]
labels =['low', 'normal', 'high']
data['platelets_category'] = pd.cut(data['platelets'], bins=bins, labels=labels)
print(data.head())

data['platelets_category'].value_counts()

sns.barplot(x='platelets_category', y='DEATH_EVENT', data=data)
plt.show()

"""The people with low blood platelets are more prone to Heart Failure.

**serum_creatinine	: Level of serum creatinine in the blood (mg/dL)**
"""

bins= [0, 0.74, 1.35, np.nan]
labels=['low', 'normal', 'high']
data['serum_creatinine_category'] = pd.cut(data['serum_creatinine'], bins=bins, labels=labels)
print(data.head())

sns.barplot(x='serum_creatinine_category', y='DEATH_EVENT', data=data)
#plt.show()

"""Thus people with more serum_creatinine is more prone to heart failure.

**serum_sodium : Level of serum sodium in the blood (mEq/L)**
"""

bins=[0, 135, 145, np.nan]
labels=['low', 'normal', 'high']
data['serum_sodium_category'] = pd.cut(data['serum_sodium'], bins=bins, labels=labels)
print(data.head())

sns.barplot(x='serum_sodium_category', y='DEATH_EVENT', data=data)
#plt.show()

"""Those people whose serum sodium is not normal are prone heart failure.

**Sex**
"""

sns.barplot(x='sex', y='DEATH_EVENT', data=data)
#plt.show()

"""Sex has no realtion to the death event. It doesn't matter whether one is male of feamle.

Moreover from the heatmap correlation in between death_event and sex is -0.004316.

**Smoking**
"""

sns.barplot(x='smoking', y='DEATH_EVENT', data=data)
#plt.show()

"""Smoking has no realtion to the death event. It doesn't matter whether one smoke or not.

Moreover from the heatmap correlation in between death_event and smoking is -0.012623.

**Time**
"""

plt.plot(data['time'])
#plt.show()

"""## Seperating the Dataset"""

array = data_1.values
X = array[:, :12]
Y = array[:, 12]

"""## Splitting the data into training and testing"""

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.67, random_state=1)

"""## Rescale the Data"""

# Standard Scalar
scalar = StandardScaler()
rescaled_X = scalar.fit_transform(x_train)

print(rescaled_X[:5])

"""## Feature Seletion"""

# Recursive Feature Elimination

model = LogisticRegression()
rfe = RFE(model, 6)
fit = rfe.fit(rescaled_X, y_train)

print(data_1.columns)
print('Num features : ', fit.n_features_)
print('Selected features : ', fit.support_)
print('Features ranking : ', fit.ranking_)

"""The selected features : 
Age, anaemia, ejection fraction, serum_creatinine, serum_sodium, time
"""

transformed_X = fit.transform(rescaled_X)
print(transformed_X[:5])

"""## Builing the Model"""

# Spot checking Algorithms

models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('SVM', SVC()))
models.append(('GNB', GaussianNB()))
models.append(('ETC', ExtraTreesClassifier()))

# evaluate each model
results = []
names = []
for name, model in models:
  kfold = KFold(n_splits=10, random_state=1)
  cv_results = cross_val_score(model, transformed_X, y_train,cv=kfold, scoring='accuracy')
  results.append(cv_results)
  names.append(name)
  print(name, ':', cv_results.mean()*100)

# compare Algorithms
fig = plt.figure(figsize=(14, 8))
fig.suptitle('Algorithm Comparison')
ax =fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
#plt.show()

"""Thus Linear Discriminant Analysis comes out to be the best model with accuracy of 89%.

# Make Predictions
"""


rescaled_x_test = scalar.transform(x_test)
transformed_x_test = fit.transform(rescaled_x_test)

lda = LinearDiscriminantAnalysis()
lda.fit(transformed_X, y_train)
predictions = lda.predict(transformed_x_test)
print('The accurcay score of the test dataset : ', accuracy_score(y_test, predictions))
print('\nThe confusion matrix : \n', confusion_matrix(y_test, predictions))
print('\nFinally the classification report : \n', classification_report(y_test, predictions))
print('Score : ', lda.score(transformed_x_test, y_test))

"""## Model deployment"""




steps = [('scaler', StandardScaler()),
         ('RFE', RFE(LogisticRegression(), 6)),
         ('lda', LinearDiscriminantAnalysis())]

pipeline = Pipeline(steps)
pipeline.fit(x_train, y_train)
predictions = pipeline.predict(x_test)
print('The accurcay score of the test dataset : ', accuracy_score(y_test, predictions))
print('\nThe confusion matrix : \n', confusion_matrix(y_test, predictions))
print('\nFinally the classification report : \n', classification_report(y_test, predictions))
print('Score : ', pipeline.score(x_test, y_test))


# saving the model
pickle.dump(pipeline, open('model.pkl', 'wb'))

# load the model to compare the results
model = pickle.load(open('model.pkl', 'rb'))
A = [[53, 1, 91, 0, 20, 1, 418000, 1.4, 139, 0, 0, 43]]
predictions = model.predict(A)
print(predictions)




