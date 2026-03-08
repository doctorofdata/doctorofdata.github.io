---
layout: post
title:  "Time-Series Forecasting w/ Prophet"
excerpt_separator: <!--more-->
---

In this post, we use a generalized additive model to forecast store sales using contrbuting factors as variables to help improve the accuracy of demand predictions.

<!--more-->

# Forecasting at Scale: Time-Series with Prophet

In this exercise we will use parallel computing to implement prophet models for each product in the data and conduct forecasting to submit to kaggle for evaluation


```python
# Import libraries
import pandas as pd
import time
from multiprocessing import Pool, cpu_count
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
%matplotlib inline
import zipfile
import itertools
from prophet import Prophet
import os
import sys
import logging
import random 

# List contents
for dirname, _, filenames in os.walk('/kaggle/input'):
    
    for filename in filenames:
        
        print(os.path.join(dirname, filename))
```

    /kaggle/input/store-sales-time-series-forecasting/oil.csv
    /kaggle/input/store-sales-time-series-forecasting/sample_submission.csv
    /kaggle/input/store-sales-time-series-forecasting/holidays_events.csv
    /kaggle/input/store-sales-time-series-forecasting/stores.csv
    /kaggle/input/store-sales-time-series-forecasting/train.csv
    /kaggle/input/store-sales-time-series-forecasting/test.csv
    /kaggle/input/store-sales-time-series-forecasting/transactions.csv



```python
# Load data
train = pd.read_csv('/kaggle/input/store-sales-time-series-forecasting/train.csv')
test = pd.read_csv('/kaggle/input/store-sales-time-series-forecasting/test.csv')
holidays = pd.read_csv('/kaggle/input/store-sales-time-series-forecasting/holidays_events.csv').rename({'date': 'ds', 'description': 'holiday'}, axis = 1)
oil = pd.read_csv('/kaggle/input/store-sales-time-series-forecasting/oil.csv').rename({'date': 'ds', 'dcoilwtico': 'oil'}, axis = 1).fillna(method = 'bfill')
```


```python
# Inspect data
print(f"Unique Item Families:    {train['family'].nunique()}") 
print(f"# Stores:                {train['store_nbr'].nunique()}")
print(f"Dataset start date:      {min(train['date'])}")
print(f"Dataset end date:        {max(train['date'])}")
print(f"Test set start date:     {min(test['date'])}")
print(f"Test set end date:       {max(test['date'])}")

interval = pd.date_range(min(train['date']), max(train['date']), freq = 'd')

print(f"Num Days:               {len(interval)}")
print(f"Train Shape:            {train['date'].nunique()}")
```

    Unique Item Families:    33
    # Stores:                54
    Dataset start date:      2013-01-01
    Dataset end date:        2017-08-15
    Test set start date:     2017-08-16
    Test set end date:       2017-08-31
    Num Days:               1688
    Train Shape:            1684



```python
oil.head()
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
      <th>ds</th>
      <th>oil</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2013-01-01</td>
      <td>93.14</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2013-01-02</td>
      <td>93.14</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2013-01-03</td>
      <td>92.97</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2013-01-04</td>
      <td>93.12</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2013-01-07</td>
      <td>93.20</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Fix dates
train['ds'] = pd.to_datetime(train['date'])
test['ds'] = pd.to_datetime(test['date'])
oil['ds'] = pd.to_datetime(oil['ds'])

# Create a master label for training 
train['item'] = train['store_nbr'].astype(str) + '_' + train['family']
test['item'] = test['store_nbr'].astype(str) + '_' + test['family']

# Merge data
test1 = test.merge(oil, how = 'left', on = 'ds').drop('date', axis = 1)
train1 = train.merge(oil, how = 'left', on = 'ds').drop('date', axis = 1)
```


```python
test.shape
```




    (28512, 7)




```python
test1.shape
```




    (28512, 7)




```python
test1.head()
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
      <th>id</th>
      <th>store_nbr</th>
      <th>family</th>
      <th>onpromotion</th>
      <th>ds</th>
      <th>item</th>
      <th>oil</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3000888</td>
      <td>1</td>
      <td>AUTOMOTIVE</td>
      <td>0</td>
      <td>2017-08-16</td>
      <td>1_AUTOMOTIVE</td>
      <td>46.8</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3000889</td>
      <td>1</td>
      <td>BABY CARE</td>
      <td>0</td>
      <td>2017-08-16</td>
      <td>1_BABY CARE</td>
      <td>46.8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3000890</td>
      <td>1</td>
      <td>BEAUTY</td>
      <td>2</td>
      <td>2017-08-16</td>
      <td>1_BEAUTY</td>
      <td>46.8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3000891</td>
      <td>1</td>
      <td>BEVERAGES</td>
      <td>20</td>
      <td>2017-08-16</td>
      <td>1_BEVERAGES</td>
      <td>46.8</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3000892</td>
      <td>1</td>
      <td>BOOKS</td>
      <td>0</td>
      <td>2017-08-16</td>
      <td>1_BOOKS</td>
      <td>46.8</td>
    </tr>
  </tbody>
</table>
</div>




```python
train1.head()
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
      <th>id</th>
      <th>store_nbr</th>
      <th>family</th>
      <th>sales</th>
      <th>onpromotion</th>
      <th>ds</th>
      <th>item</th>
      <th>oil</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>AUTOMOTIVE</td>
      <td>0.0</td>
      <td>0</td>
      <td>2013-01-01</td>
      <td>1_AUTOMOTIVE</td>
      <td>93.14</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>BABY CARE</td>
      <td>0.0</td>
      <td>0</td>
      <td>2013-01-01</td>
      <td>1_BABY CARE</td>
      <td>93.14</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>1</td>
      <td>BEAUTY</td>
      <td>0.0</td>
      <td>0</td>
      <td>2013-01-01</td>
      <td>1_BEAUTY</td>
      <td>93.14</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>1</td>
      <td>BEVERAGES</td>
      <td>0.0</td>
      <td>0</td>
      <td>2013-01-01</td>
      <td>1_BEVERAGES</td>
      <td>93.14</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>1</td>
      <td>BOOKS</td>
      <td>0.0</td>
      <td>0</td>
      <td>2013-01-01</td>
      <td>1_BOOKS</td>
      <td>93.14</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Create an index for each product
df = train1.pivot(index = 'ds', columns = 'item', values = 'sales').reset_index() #.rename({'date': 'ds'}, axis = 1)
promos = train1.pivot(index = 'ds', columns = 'item', values = 'onpromotion').reset_index() #.rename({'date': 'ds'}, axis = 1)
test_promos = test1.pivot(index = 'ds', columns = 'item', values = 'onpromotion').reset_index() #.rename({'date': 'ds'}, axis = 1)
```


```python
# Instantiate processing power for parallelism
num_cpus = cpu_count()
```


```python
# Get an index of all products
items = df.drop('ds', axis = 1).columns
```


```python
#!kaggle competitions submit -c store-sales-time-series-forecasting -f submission.csv -m "prophet_forecasting_at_scale"
```


```python
test.isnull().sum()
```




    id             0
    date           0
    store_nbr      0
    family         0
    onpromotion    0
    ds             0
    item           0
    dtype: int64




```python
model = Prophet(holidays = holidays[['ds', 'holiday']])
model.add_regressor('onpromotion') #, standardize = False)
model.add_regressor('oil' )#, standardize = False)
```




    <prophet.forecaster.Prophet at 0x7c37f7ace770>




```python
# Compile data
trainingdata = df[['ds', items[0]]].rename({items[0]: 'y'}, axis = 1)
trainingdata['y'] = trainingdata['y'].astype(float)

# cONVERT DATES
trainingdata['ds'] = pd.to_datetime(trainingdata['ds'])
promos['ds'] = pd.to_datetime(promos['ds'])

# Add regressors
trainingdata = trainingdata.merge(promos[['ds', items[0]]]).rename({items[0]: 'onpromotion'}, axis = 1).merge(oil, how = 'left', on = 'ds')

trainingdata['oil'] = trainingdata['oil'].fillna(method = 'bfill')
```


```python
trainingdata.head()
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
      <th>ds</th>
      <th>y</th>
      <th>onpromotion</th>
      <th>oil</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2013-01-01</td>
      <td>0.0</td>
      <td>0</td>
      <td>93.14</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2013-01-02</td>
      <td>3.0</td>
      <td>0</td>
      <td>93.14</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2013-01-03</td>
      <td>2.0</td>
      <td>0</td>
      <td>92.97</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2013-01-04</td>
      <td>2.0</td>
      <td>0</td>
      <td>93.12</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2013-01-05</td>
      <td>0.0</td>
      <td>0</td>
      <td>93.20</td>
    </tr>
  </tbody>
</table>
</div>




```python
model.fit(trainingdata)
```

    14:13:49 - cmdstanpy - INFO - Chain [1] start processing
    14:13:50 - cmdstanpy - INFO - Chain [1] done processing





    <prophet.forecaster.Prophet at 0x7c37f7ace770>




```python
fut = model.make_future_dataframe(periods = 16)
merge_content = test_promos[['ds', items[0]]].rename({items[0]: 'onpromotion'}, axis = 1).merge(oil, how = 'left', on = 'ds')
fut = fut.merge(merge_content)
fut['oil'] = fut['oil'].fillna(method = 'bfill')
```


```python
fut
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
      <th>ds</th>
      <th>onpromotion</th>
      <th>oil</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2017-08-16</td>
      <td>0</td>
      <td>46.80</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2017-08-17</td>
      <td>0</td>
      <td>47.07</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2017-08-18</td>
      <td>0</td>
      <td>48.59</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2017-08-19</td>
      <td>0</td>
      <td>47.39</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2017-08-20</td>
      <td>0</td>
      <td>47.39</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2017-08-21</td>
      <td>0</td>
      <td>47.39</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2017-08-22</td>
      <td>0</td>
      <td>47.65</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2017-08-23</td>
      <td>0</td>
      <td>48.45</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2017-08-24</td>
      <td>0</td>
      <td>47.24</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2017-08-25</td>
      <td>0</td>
      <td>47.65</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2017-08-26</td>
      <td>0</td>
      <td>46.40</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2017-08-27</td>
      <td>0</td>
      <td>46.40</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2017-08-28</td>
      <td>0</td>
      <td>46.40</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2017-08-29</td>
      <td>0</td>
      <td>46.46</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2017-08-30</td>
      <td>2</td>
      <td>45.96</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2017-08-31</td>
      <td>0</td>
      <td>47.26</td>
    </tr>
  </tbody>
</table>
</div>




```python
fut['ds'] = pd.to_datetime(fut['ds'])
preds = model.predict(fut)
```


```python
preds[['ds', 'onpromotion', 'oil', 'yhat']]
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
      <th>ds</th>
      <th>onpromotion</th>
      <th>oil</th>
      <th>yhat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2017-08-16</td>
      <td>0.000000</td>
      <td>-0.095392</td>
      <td>1.749961</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2017-08-17</td>
      <td>0.000000</td>
      <td>-0.094169</td>
      <td>1.512343</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2017-08-18</td>
      <td>0.000000</td>
      <td>-0.087285</td>
      <td>1.776728</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2017-08-19</td>
      <td>0.000000</td>
      <td>-0.092720</td>
      <td>2.870283</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2017-08-20</td>
      <td>0.000000</td>
      <td>-0.092720</td>
      <td>3.268697</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2017-08-21</td>
      <td>0.000000</td>
      <td>-0.092720</td>
      <td>2.059172</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2017-08-22</td>
      <td>0.000000</td>
      <td>-0.091543</td>
      <td>2.067662</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2017-08-23</td>
      <td>0.000000</td>
      <td>-0.087919</td>
      <td>2.086430</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2017-08-24</td>
      <td>0.000000</td>
      <td>-0.093400</td>
      <td>0.411933</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2017-08-25</td>
      <td>0.000000</td>
      <td>-0.091543</td>
      <td>2.089462</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2017-08-26</td>
      <td>0.000000</td>
      <td>-0.097204</td>
      <td>3.169663</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2017-08-27</td>
      <td>0.000000</td>
      <td>-0.097204</td>
      <td>3.550008</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2017-08-28</td>
      <td>0.000000</td>
      <td>-0.097204</td>
      <td>2.317490</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2017-08-29</td>
      <td>0.000000</td>
      <td>-0.096932</td>
      <td>2.297301</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2017-08-30</td>
      <td>7.158534</td>
      <td>-0.099197</td>
      <td>9.436443</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2017-08-31</td>
      <td>0.000000</td>
      <td>-0.093309</td>
      <td>2.004886</td>
    </tr>
  </tbody>
</table>
</div>




```python
out = test[test['item'] == items[0]].drop('item', axis = 1)
out['ds'] = pd.to_datetime(out['ds'])
out.head()
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
      <th>id</th>
      <th>date</th>
      <th>store_nbr</th>
      <th>family</th>
      <th>onpromotion</th>
      <th>ds</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>33</th>
      <td>3000921</td>
      <td>2017-08-16</td>
      <td>10</td>
      <td>AUTOMOTIVE</td>
      <td>0</td>
      <td>2017-08-16</td>
    </tr>
    <tr>
      <th>1815</th>
      <td>3002703</td>
      <td>2017-08-17</td>
      <td>10</td>
      <td>AUTOMOTIVE</td>
      <td>0</td>
      <td>2017-08-17</td>
    </tr>
    <tr>
      <th>3597</th>
      <td>3004485</td>
      <td>2017-08-18</td>
      <td>10</td>
      <td>AUTOMOTIVE</td>
      <td>0</td>
      <td>2017-08-18</td>
    </tr>
    <tr>
      <th>5379</th>
      <td>3006267</td>
      <td>2017-08-19</td>
      <td>10</td>
      <td>AUTOMOTIVE</td>
      <td>0</td>
      <td>2017-08-19</td>
    </tr>
    <tr>
      <th>7161</th>
      <td>3008049</td>
      <td>2017-08-20</td>
      <td>10</td>
      <td>AUTOMOTIVE</td>
      <td>0</td>
      <td>2017-08-20</td>
    </tr>
  </tbody>
</table>
</div>




```python
out.tail()
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
      <th>id</th>
      <th>date</th>
      <th>store_nbr</th>
      <th>family</th>
      <th>onpromotion</th>
      <th>ds</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>19635</th>
      <td>3020523</td>
      <td>2017-08-27</td>
      <td>10</td>
      <td>AUTOMOTIVE</td>
      <td>0</td>
      <td>2017-08-27</td>
    </tr>
    <tr>
      <th>21417</th>
      <td>3022305</td>
      <td>2017-08-28</td>
      <td>10</td>
      <td>AUTOMOTIVE</td>
      <td>0</td>
      <td>2017-08-28</td>
    </tr>
    <tr>
      <th>23199</th>
      <td>3024087</td>
      <td>2017-08-29</td>
      <td>10</td>
      <td>AUTOMOTIVE</td>
      <td>0</td>
      <td>2017-08-29</td>
    </tr>
    <tr>
      <th>24981</th>
      <td>3025869</td>
      <td>2017-08-30</td>
      <td>10</td>
      <td>AUTOMOTIVE</td>
      <td>2</td>
      <td>2017-08-30</td>
    </tr>
    <tr>
      <th>26763</th>
      <td>3027651</td>
      <td>2017-08-31</td>
      <td>10</td>
      <td>AUTOMOTIVE</td>
      <td>0</td>
      <td>2017-08-31</td>
    </tr>
  </tbody>
</table>
</div>




```python
# cONVERT DATES
df['ds'] = pd.to_datetime(df['ds'])
promos['ds'] = pd.to_datetime(promos['ds'])
test['ds'] = pd.to_datetime(test['ds'])
```


```python
# Create a function to perform modeling
def prophet_model(item):
    
    # Init model
    model = Prophet(holidays = holidays[['ds', 'holiday']])
    model.add_regressor('onpromotion') #, standardize = False)
    model.add_regressor('oil' )#, standardize = False)

    # Compile data
    trainingdata = df[['ds', item]].rename({item: 'y'}, axis = 1)
    trainingdata['y'] = trainingdata['y'].astype(float)

    # Add regressors
    trainingdata = trainingdata.merge(promos[['ds', item]], how = 'left').rename({item: 'onpromotion'}, axis = 1).merge(oil, how = 'left', on = 'ds')
    trainingdata['onpromotion'] = trainingdata['onpromotion'].fillna(0)
    trainingdata['oil'] = trainingdata['oil'].fillna(method = 'bfill')
    
    # Train
    model.fit(trainingdata)
    
    # Init predictions
    fut = model.make_future_dataframe(periods = 16)
    merge_content = test_promos[['ds', item]].rename({item: 'onpromotion'}, axis = 1).merge(oil, how = 'left', on = 'ds')
    fut = fut.merge(merge_content)
    fut['oil'] = fut['oil'].fillna(method = 'bfill')
    
    # Model
    preds = model.predict(fut)[['ds', 'yhat']] 
    preds['ds'] = pd.to_datetime(preds['ds'])
    
    # Output
    out = test[test['item'] == item].drop('item', axis = 1)    
    
    return out.merge(preds, how = 'left', on = 'ds').rename({'yhat': 'sales'}, axis = 1) #.drop(['ds', 'oil', 'onpromotion'], axis = 1)
```


```python
random.choice(items)
```




    '34_LADIESWEAR'




```python
prophet_model(random.choice(items))
```

    14:13:51 - cmdstanpy - INFO - Chain [1] start processing
    14:13:51 - cmdstanpy - INFO - Chain [1] done processing





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
      <th>id</th>
      <th>date</th>
      <th>store_nbr</th>
      <th>family</th>
      <th>onpromotion</th>
      <th>ds</th>
      <th>sales</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3001581</td>
      <td>2017-08-16</td>
      <td>29</td>
      <td>AUTOMOTIVE</td>
      <td>0</td>
      <td>2017-08-16</td>
      <td>6.083554</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3003363</td>
      <td>2017-08-17</td>
      <td>29</td>
      <td>AUTOMOTIVE</td>
      <td>0</td>
      <td>2017-08-17</td>
      <td>6.022175</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3005145</td>
      <td>2017-08-18</td>
      <td>29</td>
      <td>AUTOMOTIVE</td>
      <td>0</td>
      <td>2017-08-18</td>
      <td>6.568839</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3006927</td>
      <td>2017-08-19</td>
      <td>29</td>
      <td>AUTOMOTIVE</td>
      <td>0</td>
      <td>2017-08-19</td>
      <td>7.950376</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3008709</td>
      <td>2017-08-20</td>
      <td>29</td>
      <td>AUTOMOTIVE</td>
      <td>0</td>
      <td>2017-08-20</td>
      <td>8.325301</td>
    </tr>
    <tr>
      <th>5</th>
      <td>3010491</td>
      <td>2017-08-21</td>
      <td>29</td>
      <td>AUTOMOTIVE</td>
      <td>0</td>
      <td>2017-08-21</td>
      <td>6.421023</td>
    </tr>
    <tr>
      <th>6</th>
      <td>3012273</td>
      <td>2017-08-22</td>
      <td>29</td>
      <td>AUTOMOTIVE</td>
      <td>0</td>
      <td>2017-08-22</td>
      <td>6.901322</td>
    </tr>
    <tr>
      <th>7</th>
      <td>3014055</td>
      <td>2017-08-23</td>
      <td>29</td>
      <td>AUTOMOTIVE</td>
      <td>0</td>
      <td>2017-08-23</td>
      <td>6.409948</td>
    </tr>
    <tr>
      <th>8</th>
      <td>3015837</td>
      <td>2017-08-24</td>
      <td>29</td>
      <td>AUTOMOTIVE</td>
      <td>0</td>
      <td>2017-08-24</td>
      <td>5.036301</td>
    </tr>
    <tr>
      <th>9</th>
      <td>3017619</td>
      <td>2017-08-25</td>
      <td>29</td>
      <td>AUTOMOTIVE</td>
      <td>0</td>
      <td>2017-08-25</td>
      <td>6.838067</td>
    </tr>
    <tr>
      <th>10</th>
      <td>3019401</td>
      <td>2017-08-26</td>
      <td>29</td>
      <td>AUTOMOTIVE</td>
      <td>0</td>
      <td>2017-08-26</td>
      <td>8.204809</td>
    </tr>
    <tr>
      <th>11</th>
      <td>3021183</td>
      <td>2017-08-27</td>
      <td>29</td>
      <td>AUTOMOTIVE</td>
      <td>0</td>
      <td>2017-08-27</td>
      <td>8.558502</td>
    </tr>
    <tr>
      <th>12</th>
      <td>3022965</td>
      <td>2017-08-28</td>
      <td>29</td>
      <td>AUTOMOTIVE</td>
      <td>0</td>
      <td>2017-08-28</td>
      <td>6.625649</td>
    </tr>
    <tr>
      <th>13</th>
      <td>3024747</td>
      <td>2017-08-29</td>
      <td>29</td>
      <td>AUTOMOTIVE</td>
      <td>0</td>
      <td>2017-08-29</td>
      <td>7.066330</td>
    </tr>
    <tr>
      <th>14</th>
      <td>3026529</td>
      <td>2017-08-30</td>
      <td>29</td>
      <td>AUTOMOTIVE</td>
      <td>0</td>
      <td>2017-08-30</td>
      <td>6.506890</td>
    </tr>
    <tr>
      <th>15</th>
      <td>3028311</td>
      <td>2017-08-31</td>
      <td>29</td>
      <td>AUTOMOTIVE</td>
      <td>0</td>
      <td>2017-08-31</td>
      <td>6.418256</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Forecast
#start = time.time()

# Disable outputs
#logging.getLogger("cmdstanpy").disabled = True 

# Parallelization
#p = Pool(num_cpus)
#result = p.map(prophet_model, items)

#end = time.time()
```


```python
#print(f'Elapsed Modeling Time:  {round((start - end) / 60, 2)} minutes..')
```


```python
# Compile output
#res = pd.concat([i for i in result])
#out = res[['id', 'sales']]
```


```python
#out.shape
```


```python
test.shape
```




    (28512, 7)




```python
# Write file
out.to_csv('submission.csv', index = False)
```


```python
# Visualize outputs
sample = random.choice(items)
```


```python
sample
```




    '29_PET SUPPLIES'




```python
# Init model
model = Prophet(holidays = holidays[['ds', 'holiday']])
model.add_regressor('onpromotion') #, standardize = False)
model.add_regressor('oil' )#, standardize = False)

# Compile data
trainingdata = df[['ds', sample]].rename({sample: 'y'}, axis = 1)
trainingdata['y'] = trainingdata['y'].astype(float)

# Add regressors
trainingdata = trainingdata.merge(promos[['ds', sample]], how = 'left').rename({sample: 'onpromotion'}, axis = 1).merge(oil, how = 'left', on = 'ds')
trainingdata['onpromotion'] = trainingdata['onpromotion'].fillna(0)
trainingdata['oil'] = trainingdata['oil'].fillna(method = 'bfill')
    
# Train
model.fit(trainingdata)
    
# Init predictions
fut = model.make_future_dataframe(periods = 16)
merge_content = test_promos[['ds', sample]].rename({sample: 'onpromotion'}, axis = 1).merge(oil, how = 'left', on = 'ds')
fut = fut.merge(merge_content)
fut['oil'] = fut['oil'].fillna(method = 'bfill')
    
# Model
preds = model.predict(fut) #[['ds', 'yhat']] 
preds['ds'] = pd.to_datetime(preds['ds'])
    
# Output
out = test[test['item'] == sample].drop('item', axis = 1)    
out = out.merge(preds, how = 'left', on = 'ds').rename({'yhat': 'sales'}, axis = 1) #.drop(['ds', 'oil', 'onpromotion'], axis = 1)
```

    14:13:53 - cmdstanpy - INFO - Chain [1] start processing
    14:13:53 - cmdstanpy - INFO - Chain [1] done processing



```python
print(f"Examining Outputs for {sample}")
    
model.plot(preds)
```

    Examining Outputs for 29_PET SUPPLIES





    
![png](prophet-store-sales-forecasting_files/prophet-store-sales-forecasting_38_1.png)
    




    
![png](prophet-store-sales-forecasting_files/prophet-store-sales-forecasting_38_2.png)
    



```python
model.plot_components(preds)
```




    
![png](prophet-store-sales-forecasting_files/prophet-store-sales-forecasting_39_0.png)
    




    
![png](prophet-store-sales-forecasting_files/prophet-store-sales-forecasting_39_1.png)
    



```python
print('Showing Time Series for Model: ')
```

    Showing Time Series for Model: 



```python
out.head()
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
      <th>id</th>
      <th>date</th>
      <th>store_nbr</th>
      <th>family</th>
      <th>onpromotion_x</th>
      <th>ds</th>
      <th>trend</th>
      <th>yhat_lower</th>
      <th>yhat_upper</th>
      <th>trend_lower</th>
      <th>...</th>
      <th>weekly</th>
      <th>weekly_lower</th>
      <th>weekly_upper</th>
      <th>yearly</th>
      <th>yearly_lower</th>
      <th>yearly_upper</th>
      <th>multiplicative_terms</th>
      <th>multiplicative_terms_lower</th>
      <th>multiplicative_terms_upper</th>
      <th>sales</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3001607</td>
      <td>2017-08-16</td>
      <td>29</td>
      <td>PET SUPPLIES</td>
      <td>0</td>
      <td>2017-08-16</td>
      <td>7.514397</td>
      <td>4.404190</td>
      <td>8.789459</td>
      <td>7.514397</td>
      <td>...</td>
      <td>-0.199900</td>
      <td>-0.199900</td>
      <td>-0.199900</td>
      <td>-0.229431</td>
      <td>-0.229431</td>
      <td>-0.229431</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>6.625475</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3003389</td>
      <td>2017-08-17</td>
      <td>29</td>
      <td>PET SUPPLIES</td>
      <td>0</td>
      <td>2017-08-17</td>
      <td>7.520778</td>
      <td>4.404179</td>
      <td>8.655474</td>
      <td>7.520778</td>
      <td>...</td>
      <td>-0.343361</td>
      <td>-0.343361</td>
      <td>-0.343361</td>
      <td>-0.181378</td>
      <td>-0.181378</td>
      <td>-0.181378</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>6.541952</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3005171</td>
      <td>2017-08-18</td>
      <td>29</td>
      <td>PET SUPPLIES</td>
      <td>0</td>
      <td>2017-08-18</td>
      <td>7.527158</td>
      <td>4.113727</td>
      <td>8.530878</td>
      <td>7.527158</td>
      <td>...</td>
      <td>-0.558148</td>
      <td>-0.558148</td>
      <td>-0.558148</td>
      <td>-0.131434</td>
      <td>-0.131434</td>
      <td>-0.131434</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>6.414476</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3006953</td>
      <td>2017-08-19</td>
      <td>29</td>
      <td>PET SUPPLIES</td>
      <td>0</td>
      <td>2017-08-19</td>
      <td>7.533539</td>
      <td>5.316851</td>
      <td>9.693612</td>
      <td>7.533539</td>
      <td>...</td>
      <td>0.510069</td>
      <td>0.510069</td>
      <td>0.510069</td>
      <td>-0.080453</td>
      <td>-0.080453</td>
      <td>-0.080453</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7.515592</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3008735</td>
      <td>2017-08-20</td>
      <td>29</td>
      <td>PET SUPPLIES</td>
      <td>0</td>
      <td>2017-08-20</td>
      <td>7.539920</td>
      <td>5.291586</td>
      <td>9.929054</td>
      <td>7.539920</td>
      <td>...</td>
      <td>0.736900</td>
      <td>0.736900</td>
      <td>0.736900</td>
      <td>-0.029338</td>
      <td>-0.029338</td>
      <td>-0.029338</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7.799919</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 345 columns</p>
</div>




```python
out.shape
```




    (16, 345)




```python
fig, ax = plt.subplots()
ax.scatter(trainingdata['ds'], trainingdata['y'], color = 'dodgerblue', s = .1)
ax.scatter(out['ds'], out['sales'], color = 'red', s = .2)
plt.ylabel('Count of Item')
plt.xlabel('Date')
```




    Text(0.5, 0, 'Date')




    
![png](prophet-store-sales-forecasting_files/prophet-store-sales-forecasting_43_1.png)
    



```python

```
