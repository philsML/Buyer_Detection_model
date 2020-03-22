# -*- coding: utf-8 -*-
"""
ESCP
Machine Learning Assignment
"""
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 50)
import matplotlib.pyplot as plt

raw=pd.read_csv("C:/Users/phili/OneDrive/Documents/ESCP/MBD28 Machine Learning with Python/Assignment/train.csv")

#1. REORDER DATAFRAME
raw.head(10)
cols = raw.columns.tolist()
print(cols)
cols = cols[1:13] + cols[14:15] + cols[24:27] + cols[13:14] + cols[15:24] + cols[0:1]
raw = raw[cols]
raw.head(10)
len(raw)
#data-set contains 31.428 values


#2. DATA-CHECK

#2.1 missing values 
na={}
for i in cols:
    na[i]=raw[i].isnull().sum()
print(na)
#no missing values in any columns

#2.2 unique values
raw.describe()
unique={}
for i in cols:
    unique[i]=len(raw[i].unique())
print(unique)
#only id is uniqe with 31.428 unique values -> drop
#visitTime, C1 & C10 with very high number of unique values (>20.000) ... noise! -> for visitTime & purchaseTime possibly binning
raw.drop(["id"], axis="columns",inplace=True)

#2.3 correlation
correl=raw.corr()
correlations={}
for i in cols[1:27]:
    correlations[i]=((correl[i]>0.75).sum()+(correl[i]<-0.75).sum()-1)
print(correlations)
#some extremely high correlations in visitTime, purchaseTime, hour, N4, N8, N9, N10 and label -> check for exact relationship

correl[0:3]
#visitTime and hour correlate highly with another -> drop visitTime as hour is already the binned version of visitTime
#purchaseTime correlates perfectly (!!) with label, apparently only logged when purchase happens -> drop!
correl[22:25]
#N4 and N8 correlate at 82% -> possibly drop one (or PCA)
#N9 and N10 correlate at 90% -> possibly drop one (or PCA)

raw.drop(["visitTime","purchaseTime"], axis="columns",inplace=True)
print(raw)

#2.4 outliers

#numerical columns (N1-N10, hour) first
cols=raw.columns.tolist()
raw.hist(column=cols[13:23], grid=True, figsize=(12,25), layout=(6,2), bins=8)
#no outliers as no "distribution" - apperently in the numerical columns almost all values are 0 -> check exactly
non_zeros={}
for i in cols[13:23]:
    non_zeros[i]=sum(raw[i].value_counts()[1:])
print(non_zeros)
#especially N5-N7 as well as N9 & N10 with extremely high share of zeros (>99%) -> probably little information gain (check later!), thus candidates for dropping
raw.hist(column=cols[0])
#hour is (relatively normally) distributed as expected

#categorical columns (C1-C12) next
raw.hist(column=cols[1:13], grid=True, figsize=(12,25), layout=(6,2), bins=8) #distance plays no role here!! only count
small_categories={}
for i in cols[1:13]:
    small_categories[i]=(raw[i].value_counts()<50).sum()
print(small_categories)
#several categorical features with many very small categories (C1, C3,C10) -> let's check share
share_sc={}
for i in cols[1:13]:
    share_sc[i]=((raw[i].value_counts()<10).sum()/len(raw[i].value_counts()))*100
print(share_sc)
#especially for C1 and C10 the share of very small categories (<10 observations) is huge (>99.9%) -> check feature importance with random forest, drop if low for one-hot encoding

#dependent column last
raw.label.value_counts()
#serious class imbalance -> oversample minority classes before model-building


#3. PREPROCESSING

#3.1 Data Types
raw.dtypes
#categorical columns have to be made such
categorical_cols=cols[1:13]
categoricals={}
for i in categorical_cols:
    categoricals[i]="object"
raw = raw.astype(categoricals)

#3.2 Normalization
from sklearn.preprocessing import MinMaxScaler

numerical_cols=cols[0:1]+cols[13:23]
raw[numerical_cols]=MinMaxScaler().fit_transform(raw[numerical_cols])
raw[numerical_cols].describe()
#all numerical columns now run from 0 to 1

#3.3 oversample minority classes
from sklearn.model_selection import train_test_split

x_col=cols[0:23]
y_col=cols[23]

X_train, X_test, y_train, y_test = train_test_split(raw[x_col], raw[y_col], test_size=0.33, stratify=raw.label, random_state=0)

from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=123)
res_x, res_y = sm.fit_resample(X=X_train,y=y_train)

len(res_x), len(res_y)
(res_y==1).sum(),(res_y==-1).sum()
#two equally long arrays (for independent & dependent variables) with equally large dependent classes (1 & -1)

data=pd.DataFrame(res_x,columns=cols[0:23])
data["label"]=res_y

types={}
for i in cols:
    if i=="label":
        types[i]="int64"
    elif i[0]=="C":
        types[i]="object"
    else:
        types[i]="float64"
data = data.astype(types)
#oversampled dataset with correct labels and types


#4. FEATURE IMPORTANCE

#4.1 Importance Check
X = data[cols[0:23]]
y = data[cols[23]]

from sklearn.ensemble import RandomForestClassifier
RFC = RandomForestClassifier(n_estimators=300, criterion="entropy", min_samples_leaf=3, random_state=123)

RFC.fit(X, y)
importances = RFC.feature_importances_
std = np.std([tree.feature_importances_ for tree in RFC.estimators_],axis=0)
indices = np.argsort(importances)[::-1]      

feature_importance={}
for i in indices:
    feature_importance[cols[i]]=round(importances[i],4)
print(feature_importance)
#top-5 important: N10, N9, N8, N3, N6 | least important: C1, C4, C11, N2, C5, N7, N5 (don't even reach 0.001) 

feature_rank=[]
for i in indices:
    feature_rank.append(cols[i])
plt.figure(figsize=(12,4))
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), feature_rank)
plt.xlim([-1, X.shape[1]])
plt.show()
#N4 and N8 correlate at 82% -> drop N4
#N9 and N10 correlate at 90% -> drop N9

cols_to_drop = feature_rank[1:2] + feature_rank[5:6] + feature_rank[16:]
data.drop(cols_to_drop,axis="columns",inplace=True)
cols = data.columns.tolist()

#4.2 Recheck Importance after Column-Drop
X = data[cols[0:14]]
y = data[cols[14]]

RFC.fit(X, y)
importances = RFC.feature_importances_
std = np.std([tree.feature_importances_ for tree in RFC.estimators_],axis=0)
indices = np.argsort(importances)[::-1]      

feature_importance={}
for i in indices:
    feature_importance[cols[i]]=round(importances[i],4)
print(feature_importance)
#N10 and N8 have gained weight by dropping N9 and N4 respectively

final_cols=X.columns.to_list()

#5 MODEL-BUILDING

X_train=X
y_train=y

#5.1 Random Forest
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
skf = StratifiedKFold(n_splits=10,shuffle=False)


#5.1.1 GridsearchCV for Random Forest
RFC_parameters={"n_estimators":list(range(50,201,50)),
                "criterion":("gini","entropy"),
                "min_samples_leaf":list(range(1,5,1))
                }

RFC=RandomForestClassifier(random_state=0,n_jobs=4)
RFGS = GridSearchCV(RFC, RFC_parameters, scoring="roc_auc", refit=True, verbose=2)
RFGS.fit(X=X_train,y=y_train)
RFGS_results = pd.DataFrame(RFGS.cv_results_)
RFGS_results.sort_values("rank_test_score",inplace=True)
RFGS.best_params_

#5.1.2 check for right class weights
AUC_train_weight=[]
AUC_test_weight=[]
weight_1=[5,4,3,2,1,1,1,1,1]
weight_2=[1,1,1,1,1,2,3,4,5]
weights=["5-1","4-1","3-1","2-1","1-1","1-2","1-3","1-4","1-5"]
for i, j in zip(weight_1,weight_2):
    RFC = RandomForestClassifier(n_estimators=50, criterion="entropy", min_samples_leaf=4, class_weight={1:i,-1:j}, random_state=0)
    scores=cross_validate(RFC,X=X_train,y=y_train,return_train_score=True,cv=skf,scoring=["roc_auc"])
    scores=pd.DataFrame(scores)
    AUC_train_weight.append(scores["train_roc_auc"].mean())
    AUC_test_weight.append(scores["test_roc_auc"].mean())

fig, ax = plt.subplots()
line1, = ax.plot(weights, AUC_train_weight, label='Train')
line2, = ax.plot(weights, AUC_test_weight, label='Test')
ax.legend()
plt.xlabel('Weigths')
plt.ylabel('AUC with Weights')
plt.title('Optimal Weights of Positive Class')
plt.show()
#best weight ratio seems to be a simple 1-1

#5.1.3 apply on test-split
RFC = RandomForestClassifier(n_estimators=50, criterion="entropy", min_samples_leaf=4, class_weight={1:1,-1:1}, random_state=0)
RFC_model = RFC.fit(X_train,y_train)

from sklearn.metrics import plot_roc_curve

ax = plt.gca()
plt.plot([0,1],[0,1],color="grey")
RFC_disp = plot_roc_curve(RFC,X_test[final_cols],y_test,ax=ax,alpha=0.8,color="red")
plt.show()
#optimal threshold below 0.5


#5.2 Logistic Regression
from sklearn.linear_model import LogisticRegression

#5.2.1 GridsearchCV for Logistic Regression
LRC_parameters={"tol": [0.001,0.01,0.1],
                "C": [0.1,0.5,1,2],
                "max_iter": [5,10,25,50]
                }

LRC=LogisticRegression(random_state=0,n_jobs=4)
LRGS = GridSearchCV(LRC, LRC_parameters, scoring="roc_auc", refit=True, verbose=2)
LRGS.fit(X=X_train,y=y_train)
LRGS_results = pd.DataFrame(LRGS.cv_results_)
LRGS_results.sort_values("rank_test_score",inplace=True)
LRGS.best_params_


#5.2.2 check for right class weights
AUC_train_weight=[]
AUC_test_weight=[]
for i, j in zip(weight_1,weight_2):
    LRC = LogisticRegression(tol=0.001, C=0.1, max_iter=25, class_weight={1:i,-1:j},solver="lbfgs")
    scores=cross_validate(LRC,X=X_train,y=y_train,return_train_score=True,cv=skf,scoring=["roc_auc"])
    scores=pd.DataFrame(scores)
    AUC_train_weight.append(scores["train_roc_auc"].mean())
    AUC_test_weight.append(scores["test_roc_auc"].mean())

fig, ax = plt.subplots()
line1, = ax.plot(weights, AUC_train_weight, label='Train')
line2, = ax.plot(weights, AUC_test_weight, label='Test')
ax.legend()
plt.xlabel('Weights')
plt.ylabel('AUC at Weights')
plt.title('Optimal Weights')
plt.show()
#optimal weight-ratio seems to be 3-1

#5.2.3 apply on test-split
LRC = LRC = LogisticRegression(tol=0.001, C=0.1, max_iter=25, class_weight={1:3,-1:1},solver="lbfgs")
LRC_model = LRC.fit(X_train,y_train)

ax = plt.gca()
plt.plot([0,1],[0,1],color="grey")
RFC_disp = plot_roc_curve(RFC,X_test[final_cols],y_test,ax=ax,alpha=0.8,color="red")
LRC_disp = plot_roc_curve(LRC,X_test[final_cols],y_test,ax=ax,alpha=0.8,color="blue")
plt.show()
#optimal threshold below 0.5 -> check later


#5.3 Direct comparison of two best models 
AUC_train_models=[]
AUC_test_models=[]
scores=cross_validate(RFC,X=X_train,y=y_train,return_train_score=True,cv=skf,scoring=["roc_auc"])
scores=pd.DataFrame(scores)
AUC_train_models.append(scores["train_roc_auc"].mean())
AUC_test_models.append(scores["test_roc_auc"].mean())

scores=cross_validate(LRC,X=X_train,y=y_train,return_train_score=True,cv=skf,scoring=["roc_auc"])
scores=pd.DataFrame(scores)
AUC_train_models.append(scores["train_roc_auc"].mean())
AUC_test_models.append(scores["test_roc_auc"].mean())

fig, ax = plt.subplots()
line1, = ax.plot(["Random Forest","Logistic Regression"], AUC_train_models, label='Train')
line2, = ax.plot(["Random Forest","Logistic Regression"], AUC_test_models, label='Test')
ax.legend()
plt.xlabel('Models')
plt.ylabel('AUC at Models')
plt.title('Optimal Model')
plt.show()
#random forest beats logistic regression by far


#5.4 determine right threshold for prediction 
from sklearn.metrics import confusion_matrix, classification_report

probabilities_testset = RFC_model.predict_proba(X_test[final_cols])

prediction_test_01, prediction_test_02, prediction_test_03, prediction_test_04 = [], [], [], []

def det_threshold(thresholds,y,probabilities):
    results=pd.DataFrame(y)
    for t in thresholds:
        prediction=[]
        for i in probabilities:
            if i>t:
                prediction.append(1)
            else:
                prediction.append(-1)
        results[t]=prediction
        conf_m = pd.DataFrame(confusion_matrix(y, prediction), index=["true_-1","true_1"], columns=["pred_-1","pred_1"])
        print(t,conf_m)
    return results
    
det_threshold(np.arange(0.05,0.75,0.05),y_test,probabilities_testset[0:,1:2])    
#optimal threshold seems to be between 0.15 and 0.20
threshs = det_threshold(np.arange(0.15,0.21,0.01),y_test,probabilities_testset[0:,1:2])    
#optimal threshold is 0.17
t_cols=threshs.columns.to_list()

test_class_report=classification_report(y_test,threshs[t_cols[3]])
print(test_class_report)
#recall of 0.79 means only missing 20% of potential customers
#precision of 0.25 means only "over-targeting" by factor 4


#6. APPLICATION OF MODEL

raw=pd.read_csv("C:/Users/phili/OneDrive/Documents/ESCP/MBD28 Machine Learning with Python/Assignment/test1.csv")
cols = raw.columns.tolist()
cols = cols[1:13] + cols[14:15] + cols[24:27] + cols[13:14] + cols[15:24] + cols[0:1]
raw = raw[cols]
raw.drop(["id"], axis="columns",inplace=True)
raw.drop(["visitTime","purchaseTime"], axis="columns",inplace=True)
raw = raw.astype(categoricals)
raw[numerical_cols]=MinMaxScaler().fit_transform(raw[numerical_cols])
raw.drop(cols_to_drop,axis="columns",inplace=True)
X_apply=raw[final_cols]
y_apply=raw.label


class_probabilities = RFC_model.predict_proba(X_apply)
X_apply["class_prob_-1"], X_apply["class_prob_1"] = class_probabilities[0:,0:1], class_probabilities[0:,1:2]
prediction_class=[]
for i in X_apply["class_prob_1"]:
    if i>0.17:
        prediction_class.append(1)
    else:
        prediction_class.append(-1)

data=pd.read_csv("C:/Users/phili/OneDrive/Documents/ESCP/MBD28 Machine Learning with Python/Assignment/test1.csv")
data["class_prob_-1"] = X_apply["class_prob_-1"]
data["class_prob_1"] = X_apply["class_prob_1"]
data["prediction_class"] = prediction_class

pd.DataFrame.to_csv(data,path_or_buf="C:/Users/phili/OneDrive/Documents/ESCP/MBD28 Machine Learning with Python/Assignment/test1_predicted.csv",sep=",")