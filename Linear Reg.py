# -*- coding: utf-8 -*-
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


#%%
'''
# checking assumptions
# do not run for model building
# 1. Appropriate outcome type - is outcome binary or not

# 2. Linearity of independent variables and log odds
# if p is probability of positive outcome then logit(p) = log(p/(1-p))
# Test using Box Sidwell test
# 1. Keep only conitnuous variables
# 2. For each conitnuous variable add interaction term, ex Age will have Age*ln(Age)

# Define continuous variables
continuous_var = ['perc_premium_paid_by_cash_credit', 'age_in_days', 'Income', 'application_underwriting_score','no_of_premiums_paid','premium']

# Add logit transform interaction terms (natural log) for continuous variables e.g.. Age * Log(Age)
for var in continuous_var:
    x[f'{var}:Log_{var}'] = x[var].apply(lambda x: x * np.log(x))

# Keep columns related to continuous variables
cols_to_keep = continuous_var + x.columns.to_list()[-6:]

# Redefining variables to include interaction terms
X_lt = x[cols_to_keep]

# Add constant term
import statsmodels.api as sm
X_lt_constant = sm.add_constant(X_lt, prepend=False)
  
# Building model and fit the data (using statsmodel's Logit)
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families

X_lt_constant.isna().sum()
X_lt_constant = X_lt_constant.notnull()

#X_lt_constant = X_lt_constant.dropna()
logit_results = GLM(y, X_lt_constant, family=families.Binomial()).fit()

# Display summary results
print(logit_results.summary())

# hints at non-linear association

'''



#%%
'''
#3. No strongly influencial outliers
# Use GLM method for logreg here so that we can retrieve the influence measures

logit_results = GLM(y, x, family=families.Binomial()).fit()

# Get influence measures
influence = logit_results.get_influence()

# Obtain summary df of influence measures
summ_df = influence.summary_frame()

# Filter summary df to Cook's distance values only
diagnosis_df = summ_df[['cooks_d']]

# Set Cook's distance threshold
cook_threshold = 4 / len(x)

# Append absolute standardized residual values 
from scipy import stats
diagnosis_df['std_resid'] = stats.zscore(logit_results.resid_pearson)
diagnosis_df['std_resid'] = diagnosis_df['std_resid'].apply(lambda x: np.abs(x))

# Find observations which are BOTH outlier (std dev > 3) and highly influential
extreme = diagnosis_df[(diagnosis_df['cooks_d'] > cook_threshold) & 
                       (diagnosis_df['std_resid'] > 3)]

# Show top 5 highly influential outlier observations
extreme.sort_values("cooks_d", ascending=False).head()

# we can remove the outliers if the fit isnt good

#x = x.drop([4388, 1502,372,4602,1444])
#y = y.drop([4388, 1502,372,4602,1444])


'''


#%%
'''
#4. Absence of Multicollinearity
# Check VIF

from statsmodels.stats import outliers_influence
def calc_vif(df):
    vif = pd.DataFrame()
    vif["variables"] = df.columns
    vif["VIF"] = [outliers_influence.variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    return(vif)

calc_vif(x)
# age_in_days and application_underwriting_score have multicolliniarity, might affect
# accuracy
'''
#%%

#5. Independence of Observations: Assumed
#6. Sufficiently large data size: Assumed

#%%

# Check significant variables
import statsmodels.api as sm
logit_model=sm.Logit(y,x)
result=logit_model.fit()
print(result.summary2())

# train test split
# Import train_test_split function
from sklearn.model_selection import train_test_split

# select only significant variables
x = x[['perc_premium_paid_by_cash_credit', 'age_in_days',
       'Count_3-6_months_late', 'Count_6-12_months_late',
       'Count_more_than_12_months_late', 'application_underwriting_score',
       'no_of_premiums_paid','sourcing_channel1']]

# Split dataset into training set and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2) # 80% training and 20% test

# residence area and premium not significant

# Build and fit model
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(x_train, y_train)

y_pred = logreg.predict(x_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(x_test, y_test)))

#Confusion Matrix

confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
print (confusion_matrix)