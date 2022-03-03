# -*- coding: utf-8 -*-

# Your directory
import os
os.chdir("C:\\Users\\swati\\Desktop\\Minku Work")
#####

# Import train file, we will use this for both train and test
import pandas as pd
df = pd.read_csv("train.txt")

# Drop NA values since random forest cannot handle NAs
df =df.dropna()

# issues is it is not predicting well those cases where the person did not renew
## IMBALANCED DATASET
df.groupby('renewal').size()

# Balance it, take equal rows from both cases to train and test


df_1 = df[df.renewal==0]
df_2 = df[df.renewal==1][0:4780]

df = pd.concat([df_1, df_2])
# train test split
# Import train_test_split function
from sklearn.model_selection import train_test_split

x = df.iloc[:,1:-1]
y = df.iloc[:,-1]

#convert categorical strings into numeric
x.residence_area_type = x.residence_area_type.astype('category')
x['residence_area_type1'] = x.residence_area_type.cat.codes

x.sourcing_channel = x.sourcing_channel.astype('category')
x['sourcing_channel1'] = x.sourcing_channel.cat.codes

#drop string columns
x = x[['perc_premium_paid_by_cash_credit', 'age_in_days', 'Income',
       'Count_3-6_months_late', 'Count_6-12_months_late',
       'Count_more_than_12_months_late', 'application_underwriting_score',
       'no_of_premiums_paid',
       'residence_area_type1', 'sourcing_channel1','premium']]


# Split dataset into training set and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2) # 70% training and 30% test

#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(x_train,y_train)

#fit on test set
y_pred=clf.predict(x_test)
probs = clf.predict_proba(x_test)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#74% accuracy

# Confusion matrix
confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
print (confusion_matrix)


# Need to optimize hyper parameters
# number of trees
'''
n_estimators = [500,800,1500,2500]
# max features to include at every split
max_features = ['auto']
# max num of levels in tree
max_depth = [10,20,30,40]
max_depth.append(None)
# min samples required to split a node
min_samples_split = [2,5,10,15]
# min samples required at each leaf node
min_samples_leaf = [1,5,10,15]

grid_param = {'n_estimators':n_estimators,
              'max_features':max_features,
              'max_depth':max_depth,
              'min_samples_split':min_samples_split,
              'min_samples_leaf':min_samples_leaf
              }

from sklearn.model_selection import RandomizedSearchCV
RFC = RandomForestClassifier(random_state=1)
RFC_random = RandomizedSearchCV(estimator=RFC,
                                param_distributions=grid_param,
                                n_iter= 500,
                                cv=3,verbose=2,random_state=42,
                                n_jobs=1) 

RFC_random.fit(x_train,y_train)
print(RFC_random.best_params_)
'''
# use the best parameters
randmf = RandomForestClassifier(n_estimators = 500, min_samples_split = 2,
                                min_samples_leaf= 5, max_features = 'auto', max_depth= 10) 
randmf.fit( x_train, y_train) 

#fit on test set
y_pred=randmf.predict(x_test)
probs = randmf.predict_proba(x_test)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#74% accuracy

# Confusion matrix
confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
print (confusion_matrix)




# probability Dataframe
df_prob = pd.DataFrame(data=probs,  columns=["Prob_0", "Prob_1"])
df_prob['Prediction'] = y_pred.tolist()
##Feature Importance
feature_imp = pd.Series(clf.feature_importances_,index=x.columns).sort_values(ascending=False)
feature_imp

#Plotting it in a chart

import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline
# Creating a bar plot
sns.barplot(x=feature_imp, y=feature_imp.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()


# Effort vs % improvement in renewal probability

import numpy as np

df_target = df_prob[df_prob['Prediction']==0]

df_target['change_in_p'] = df_target["Prob_1"].apply(lambda x: 0.5-x)

change_in_p = df_target['change_in_p']    
effort = -5*np.log(1-((change_in_p*100)/20))

