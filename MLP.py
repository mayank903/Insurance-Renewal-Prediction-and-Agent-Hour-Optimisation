"""
Created on Tue Jan 11 20:58:21 2022

@author: mayank
"""
#%%
# Your directory
import os
os.chdir("C:\\Users\\mayan\\Desktop\\T2\\ML\\Group Project\\Random Forest Classifier-20220214T181446Z-001\\Random Forest Classifier")
#####

# Import train file, we will use this for both train and test
import pandas as pd
import numpy as np
df = pd.read_csv("train.txt")

df = df.dropna()

# Handle outliers
# Outliers
import matplotlib.pyplot as plt

fig, axs = plt.subplots(2,2)
fig.suptitle('Outliers')
axs[0,0].boxplot(df.perc_premium_paid_by_cash_credit)
axs[0, 0].set_title('% premium cash credit',fontsize="medium")
axs[0,1].boxplot(df.age_in_days)
axs[0, 1].set_title('age_in_days',fontsize="medium")
axs[1,0].boxplot(df.Income)
axs[1, 0].set_title('% Income',fontsize="medium")
axs[1,1].boxplot(df.application_underwriting_score)
axs[1, 1].set_title('application_underwriting_score',fontsize="medium")


df = df[df.age_in_days<25000]
df = df[df.application_underwriting_score>98]


## IMBALANCED DATASET
df.groupby('renewal').size()

# Balance it, take equal rows from both cases to train and test


df_1 = df[df.renewal==0]
rows = df_1.shape[0]
df_2 = df[df.renewal==1].sample(n=rows,replace=False)
df = pd.concat([df_1, df_2])

x = df.iloc[:,1:-1]
y = df.iloc[:,-1]

#convert categorical strings into numeric
#x.residence_area_type = x.residence_area_type.astype('category')
#x['residence_area_type1'] = x.residence_area_type.cat.codes
'''
one_hot_residence_area_type =pd.get_dummies(x.residence_area_type)

x = x.join(one_hot_residence_area_type)
'''


#x.sourcing_channel = x.sourcing_channel.astype('category')
#x['sourcing_channel1'] = x.sourcing_channel.cat.codes
one_hot_sourcing_channel =pd.get_dummies(x.sourcing_channel)

x = x.join(one_hot_sourcing_channel)
x = x.drop('sourcing_channel',axis = 1)

'''
#drop string columns
x = x[['perc_premium_paid_by_cash_credit', 'age_in_days', 'Income',
       'Count_3-6_months_late', 'Count_6-12_months_late',
       'Count_more_than_12_months_late', 'application_underwriting_score',
       'no_of_premiums_paid',
       'residence_area_type1', 'sourcing_channel1','premium']]
'''

# train test split
# Import train_test_split function
from sklearn.model_selection import train_test_split

# select only significant variables
x = x.drop('Income',axis = 1)
x = x.drop('premium',axis = 1)
x = x.drop('residence_area_type',axis = 1)

# Split dataset into training set and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2) # 80% training and 20% test

#%%
##### BUild a multi layer perceptron #####
from keras.models import Sequential
from keras.layers import Dense, Dropout
#import tensorflow as tf
model_mlp = Sequential()
model_mlp.add(Dense(10, input_dim=len(x_train.columns), activation='relu'))
#model_mlp.add(Dense(15, activation='sigmoid'))
model_mlp.add(Dense(1, activation='sigmoid'))

model_mlp.compile(loss = 'binary_crossentropy', optimizer = 'Adamax', metrics = ['accuracy'])

model_mlp.fit(x_train,y_train, epochs=100, verbose=0, batch_size=20)

score=model_mlp.evaluate(x_train, y_train)
print(score[1])
score=model_mlp.evaluate(x_test, y_test)
print(score[1])

from sklearn.metrics import confusion_matrix
pred=model_mlp.predict(x_train)
pred=np.where(pred>0.5,1,0)
cm=confusion_matrix(pred, y_train)

print(cm)
accuracy=(cm[0,0]+cm[1,1])/sum(sum(cm))
print(accuracy)

pred=model_mlp.predict(x_test)
pred=np.where(pred>0.5,1,0)
pred = pred.tolist()
pred = [i[0] for i in pred]
pred = pd.DataFrame(pred)
y_test = y_test.tolist()
y_test = pd.DataFrame(y_test)
cm=confusion_matrix(pred, y_test)
print(cm)
accuracy=(cm[0,0]+cm[1,1])/sum(sum(cm))
print(accuracy)

confusion_matrix = pd.crosstab(y_test.iloc[:,0], pred.iloc[:,0], rownames=['Actual'], colnames=['Predicted'])
print (confusion_matrix)

import matplotlib.pyplot as plt
plt.plot(df.x, df.total_mobile_subscription) 