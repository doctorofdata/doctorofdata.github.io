---
title: 'Modeling Heart Disease with AI'
date: 2026-03-08
permalink: /posts/2026/03/modeling-heart-disease/
excerpt_separator: <!--more-->
---

# Machine Learning- An Approach to Choosing Model Parameters <a class = "anchor" id = "top"></a>

Introduction- In this notebook, we will use the Optuna library to conduct experiments which can be used to set model hyperparameters for effectively implementing various machine learning algorithms


**Table of Contents**

* [Background](#back)
* [Setup](#setup)
* [Experimentation](#exp)

---

# 🫀 Predicting Heart Disease- A Machine Learning Approach to Health <a class = "anchor" id = "back"></a>

---

In this notebook, we will utilize advanced optimization techniques and feature engineering to improve machine learning scores regarding a dataset of patients' metabolic information to see if this data can help with indications as to whether or not an individual has or could develop heart disease.

---

The dataset features are as follows-

<table>
  <tbody>
    <tr>
    <td>Feature</td>
      <td>🧓 Age</td>
      <td>🚹 Sex</td>
      <td>💔 Chest pain type</td>
      <td>💉 BP</td>
      <td>🧈 Cholesterol</td>
        <td>🍬 FBS over 120</td>
        <td>📈 EKG results</td>
        <td>❤️ Max HR</td>
        <td>🏃 Exercise angina</td>
        <td>📉 ST depression</td>
        <td>⛰️ Slope of ST</td>
        <td>🩸 Number of vessels fluro</td>
        <td>🧬 Thallium</td>
        <td>🎯 Heart Disease</td>
    </tr>
  </tbody>
</table>

---

<div style="padding:20px;border-radius:14px;background:linear-gradient(135deg,#141e30,#243b55);color:white;font-family:Segoe UI;box-shadow:0 6px 18px rgba(0,0,0,0.25);">

<p style="font-size:15px;line-height:1.6;">
Each feature is derived from well-established cardiological diagnostics. Together they capture
multiple physiological dimensions of coronary artery disease:
</p>

<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:16px;margin-top:18px;">

  <div style="background:rgba(255,255,255,0.08);padding:14px;border-radius:12px;">
    <h4>🧓 Demographics</h4>
    <ul>
      <li>Age</li>
      <li>Sex</li>
    </ul>
  </div>

  <div style="background:rgba(255,255,255,0.08);padding:14px;border-radius:12px;">
    <h4>🧈 Metabolic Risk</h4>
    <ul>
      <li>Cholesterol</li>
      <li>Fasting Blood Sugar</li>
      <li>Blood Pressure</li>
    </ul>
  </div>

  <div style="background:rgba(255,255,255,0.08);padding:14px;border-radius:12px;">
    <h4>❤️ Symptom Profiles</h4>
    <ul>
      <li>Chest Pain Type</li>
      <li>Exercise-induced Angina</li>
    </ul>
  </div>

  <div style="background:rgba(255,255,255,0.08);padding:14px;border-radius:12px;">
    <h4>📈 Functional Testing</h4>
    <ul>
      <li>EKG Results</li>
      <li>ST Depression</li>
      <li>ST Slope</li>
      <li>Max Heart Rate</li>
    </ul>
  </div>

  <div style="background:rgba(255,255,255,0.08);padding:14px;border-radius:12px;">
    <h4>🩻 Imaging Diagnostics</h4>
    <ul>
      <li>Fluoroscopy Vessels</li>
      <li>Thallium Stress Test</li>
    </ul>
  </div>

</div>
</div>

[Back to top..](#top)

---

# 💻 Setup and Environment <a class = "anchor" id = "setup"></a>


```python
# Install packages
!pip install kaggle kagglehub matplotlib seaborn lazypredict ipywidgets jupyter_contrib_nbextensions optuna catboost -q
```


```python
# Import libraries
import numpy as np
import pandas as pd
from IPython.display import display, HTML
import os
from sklearn.model_selection import train_test_split
import lazypredict
from lazypredict.Supervised import LazyClassifier
import subprocess
import json
import zipfile
from io import BytesIO
import xgboost as xgb
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import catboost as cb
import zipfile
import io
from contextlib import redirect_stdout
```


```python
# Download the data from source
!kaggle competitions download -c playground-series-s6e2 --force
```

    Downloading playground-series-s6e2.zip to /Users/anon/Downloads
      0%|                                               | 0.00/10.2M [00:00<?, ?B/s]
    100%|██████████████████████████████████████| 10.2M/10.2M [00:00<00:00, 1.29GB/s]



```python
!pwd
```

    /Users/anon/Downloads



```python
print('Loading data..')

# Read the data
zf = zipfile.ZipFile('/Users/anon/Downloads/playground-series-s6e2.zip') 
df = pd.read_csv(zf.open('train.csv')).drop('id', axis = 1)
test = pd.read_csv(zf.open('test.csv')) #.drop('id', axis = 1)
```

    Loading data..



```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Sex</th>
      <th>Chest pain type</th>
      <th>BP</th>
      <th>Cholesterol</th>
      <th>FBS over 120</th>
      <th>EKG results</th>
      <th>Max HR</th>
      <th>Exercise angina</th>
      <th>ST depression</th>
      <th>Slope of ST</th>
      <th>Number of vessels fluro</th>
      <th>Thallium</th>
      <th>Heart Disease</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>58</td>
      <td>1</td>
      <td>4</td>
      <td>152</td>
      <td>239</td>
      <td>0</td>
      <td>0</td>
      <td>158</td>
      <td>1</td>
      <td>3.60</td>
      <td>2</td>
      <td>2</td>
      <td>7</td>
      <td>Presence</td>
    </tr>
    <tr>
      <th>1</th>
      <td>52</td>
      <td>1</td>
      <td>1</td>
      <td>125</td>
      <td>325</td>
      <td>0</td>
      <td>2</td>
      <td>171</td>
      <td>0</td>
      <td>0.00</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Absence</td>
    </tr>
    <tr>
      <th>2</th>
      <td>56</td>
      <td>0</td>
      <td>2</td>
      <td>160</td>
      <td>188</td>
      <td>0</td>
      <td>2</td>
      <td>151</td>
      <td>0</td>
      <td>0.00</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Absence</td>
    </tr>
    <tr>
      <th>3</th>
      <td>44</td>
      <td>0</td>
      <td>3</td>
      <td>134</td>
      <td>229</td>
      <td>0</td>
      <td>2</td>
      <td>150</td>
      <td>0</td>
      <td>1.00</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>Absence</td>
    </tr>
    <tr>
      <th>4</th>
      <td>58</td>
      <td>1</td>
      <td>4</td>
      <td>140</td>
      <td>234</td>
      <td>0</td>
      <td>2</td>
      <td>125</td>
      <td>1</td>
      <td>3.80</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>Presence</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Map the outcome to numeric
def convert_outcomes(outcome):

    if outcome == 'Absence':

        return 0

    elif outcome == 'Presence':

        return 1
        
df['Heart Disease'] = df['Heart Disease'].map(convert_outcomes)
```


```python
# Split the data for training
x = df[[i for i in df.columns if i != 'Heart Disease']]
y = df['Heart Disease']

xtrain, xval, ytrain, yval = train_test_split(x, y, test_size = .2, random_state = 100)
```

[Back to top..](#top)

---

# Experimentation 🔧 <a class = "anchor" id = "exp"></a>


```python
# Define objective function for optuna with lgith gbm
def objective(trial):

    dtrain = lgb.Dataset(xtrain, label = ytrain)
    
    params = {"objective": "binary",
              "metric": "auc",
              "verbosity": -1,
              "boosting_type": "gbdt",
              "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
              "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
              "num_leaves": trial.suggest_int("num_leaves", 2, 256),
              "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
              "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
              "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
              "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),}

        
    gbm = lgb.train(params, dtrain)
    preds = gbm.predict(xval)
    pred_labels = np.rint(preds)
    auc = roc_auc_score(yval, pred_labels)

    return auc
```


```python
# Create a study and optimize
print('Initiating Optimization Sequence..')

# Silence optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Conduct study
study = optuna.create_study(direction = 'maximize') 
study.optimize(objective, n_trials = 50, n_jobs = 4) 

# Print the captured output to the console (using the restored stdout)
print("Modeling complete..")

# Print the best hyperparameters and AUC
print('Optimized Light GBM Performance- ')
print(f"Best trial value (AUC): {study.best_value}")
print(f"Best hyperparameters: {study.best_params}")
```

    Initiating Optimization Sequence..
    Modeling complete..
    Optimized Light GBM Performance- 
    Best trial value (AUC): 0.8865625304640333
    Best hyperparameters: {'lambda_l1': 0.005711951061229914, 'lambda_l2': 0.02701383662599308, 'num_leaves': 162, 'feature_fraction': 0.4779878924499187, 'bagging_fraction': 0.9104828961933018, 'bagging_freq': 4, 'min_child_samples': 79}



```python
# Convert the data for modeling
dtrain = lgb.Dataset(x, label = y)

# Fit
gbm = lgb.train(study.best_params, dtrain)

# Predict
gbmtest = gbm.predict(test.drop(['id'], axis = 1))

# Generate submissions
test['Heart Disease'] = gbmtest

# Write to file
test[['id', 'Heart Disease']].to_csv('heart_disease_test_gbm.csv', index = False)
```

    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Auto-choosing row-wise multi-threading, the overhead of testing was 0.012748 seconds.
    You can set `force_row_wise=true` to remove the overhead.
    And if memory is not enough, you can set `force_col_wise=true`.
    [LightGBM] [Info] Total Bins 419
    [LightGBM] [Info] Number of data points in the train set: 630000, number of used features: 13
    [LightGBM] [Info] Start training from score 0.448340



```python
print('Uploading scores to Kaggle..')
!kaggle competitions submit -c playground-series-s6e2 -f heart_disease_test_gbm.csv -m "LightGBM heart disease predictions test set"
```

    Uploading scores to Kaggle..
    100%|██████████████████████████████████████| 6.89M/6.89M [00:04<00:00, 1.62MB/s]
    Successfully submitted to Predicting Heart Disease


```python
!kaggle competitions submissions -c playground-series-s6e2
```

    fileName                               date                        description                                                                 status                     publicScore  privateScore  
    -------------------------------------  --------------------------  --------------------------------------------------------------------------  -------------------------  -----------  ------------  
    heart_disease_test_gbm.csv             2026-03-06 16:43:05.187000  LightGBM heart disease predictions test set                                 SubmissionStatus.COMPLETE  0.95240      0.95401       
    submission.csv                         2026-02-24 19:38:04                                                                                     SubmissionStatus.COMPLETE  0.87823      0.87954       
    submission.csv                         2026-02-23 20:31:03.743000  ensemblemodeling-heart-disease-predictions | LightGBM * XGBoost *CatBoost   SubmissionStatus.COMPLETE  0.87823      0.87954       
    submission.csv                         2026-02-23 20:30:39         StackedEnsemble heart disease predictions test set                          SubmissionStatus.COMPLETE  0.87823      0.87954       
    submission.csv                         2026-02-23 20:01:25         StackedEnsemble heart disease predictions test set                          SubmissionStatus.COMPLETE  0.87641      0.87986       
    heart_disease_test_gbm.csv             2026-02-22 22:20:26         LightGBM heart disease predictions test set                                 SubmissionStatus.COMPLETE  0.95227      0.95398       
    heart_disease_test_gbm.csv             2026-02-22 22:17:42.617000  LightGBM heart disease predictions test set                                 SubmissionStatus.COMPLETE  0.95217      0.95392       
    submission.csv                         2026-02-22 22:04:35.347000  LightGBM Featurized heart disease predictions test set                      SubmissionStatus.COMPLETE  0.95230      0.95379       
    submission.csv                         2026-02-22 21:51:05.337000  heart-disease-predictions:feature-engineering | .95+ auc lightgbm           SubmissionStatus.COMPLETE  0.95179      0.95351       
    submission.csv                         2026-02-22 21:50:42.543000  LightGBM Featurized heart disease predictions test set                      SubmissionStatus.COMPLETE  0.95179      0.95351       
    heart_disease_test_gbm.csv             2026-02-22 21:50:25         LightGBM heart disease predictions test set                                 SubmissionStatus.COMPLETE  0.95233      0.95392       
    submission.csv                         2026-02-22 21:31:52         LightGBM Featurized heart disease predictions test set                      SubmissionStatus.COMPLETE  0.95212      0.95385       
    heart_disease_test_gbm_featurized.csv  2026-02-22 21:11:45         LightGBM Featurized heart disease predictions test set                      SubmissionStatus.COMPLETE  0.95212      0.95385       
    heart_disease_test_gbm.csv             2026-02-22 20:08:58.903000  LightGBM heart disease predictions test set                                 SubmissionStatus.COMPLETE  0.95234      0.95403       
    submission.csv                         2026-02-22 18:59:36         h2o heart disease predictions test set                                      SubmissionStatus.COMPLETE  0.88452      0.88571       
    submission.csv                         2026-02-21 14:04:38.603000  Notebook heart-disease-prediction: comprehensive-modeling                   SubmissionStatus.COMPLETE  0.88460      0.88559       
    submission.csv                         2026-02-21 14:04:13.253000  h2o heart disease predictions test set                                      SubmissionStatus.COMPLETE  0.88460      0.88559       
    submission.csv                         2026-02-20 17:36:38.107000  Notebook Heart Disease Prediction: Comprehensive Model Eva | Version 5      SubmissionStatus.COMPLETE  0.88472      0.88562       
    submission.csv                         2026-02-20 17:36:12.513000  h2o heart disease predictions test set                                      SubmissionStatus.COMPLETE  0.88472      0.88562       
    submission.csv                         2026-02-20 16:03:12.007000  Notebook Heart Disease Prediction: Comprehensive Model Eva | Version 4      SubmissionStatus.COMPLETE  0.88468      0.88564       
    submission.csv                         2026-02-20 16:02:44.473000  h2o heart disease predictions test set                                      SubmissionStatus.COMPLETE  0.88468      0.88564       
    heart_disease_test_h2o.csv             2026-02-20 15:45:56         h2o heart disease predictions test set                                      SubmissionStatus.COMPLETE  0.88463      0.88568       
    heart_disease_test_h2o.csv             2026-02-20 15:30:25         h2o heart disease predictions test set                                      SubmissionStatus.COMPLETE  0.88460      0.88569       
    heart_disease_test_h2o.csv             2026-02-19 16:58:11.307000  h2o heart disease predictions test set                                      SubmissionStatus.COMPLETE  0.88461      0.88563       
    heart_disease_test_h2o.csv             2026-02-19 16:37:17.767000  h2o heart disease predictions test set                                      SubmissionStatus.COMPLETE  0.88407      0.88527       
    heart_disease_test_h2o.csv             2026-02-19 15:41:44.227000  h2o heart disease predictions test set                                      SubmissionStatus.COMPLETE  0.88458      0.88542       
    heart_disease_test_gbm.csv             2026-02-18 17:32:39         LightGBM heart disease predictions test set                                 SubmissionStatus.COMPLETE  0.95238      0.95395       
    heart_disease_test_catboost.csv        2026-02-18 17:26:48.153000  catboost heart disease predictions test set                                 SubmissionStatus.COMPLETE  0.88391      0.88620       
    heart_disease_test_xgb.csv             2026-02-18 17:10:46         xgb heart disease predictions test set                                      SubmissionStatus.COMPLETE  0.88257      0.88532       
    heart_disease_test_rf.csv              2026-02-18 17:10:40         rf heart disease predictions test set                                       SubmissionStatus.COMPLETE  0.87568      0.87834       
    heart_disease_test_catboost.csv        2026-02-18 14:23:04         catboost heart disease predictions test set                                 SubmissionStatus.COMPLETE  0.88410      0.88643       
    heart_disease_test_xgb.csv             2026-02-18 14:18:58         xgb heart disease predictions test set                                      SubmissionStatus.COMPLETE  0.88257      0.88532       
    heart_disease_test_rf.csv              2026-02-18 14:14:17.973000  rf heart disease predictions test set                                       SubmissionStatus.COMPLETE  0.87568      0.87834       



```python
# Get scores from the mode
lgb.plot_importance(gbm)
```




    <Axes: title={'center': 'Feature importance'}, xlabel='Feature importance', ylabel='Features'>




    
![png](heart-disease-predictions-feature-engineering-blog-post_files/heart-disease-predictions-feature-engineering-blog-post_18_1.png)
    


[Back to top..](#top)
