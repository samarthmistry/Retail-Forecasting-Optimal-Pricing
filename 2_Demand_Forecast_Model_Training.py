################################################## 0: import libraries and define functions
import csv
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.externals import joblib

################################################## 1: define paths of input files and output files
## input files
df_sales_loc = "D:/samarth/Desktop/PriceOp/Project/aggregated_sales_data/"
processed_time_d_loc = "D:/samarth/Desktop/PriceOp/Project/publicparameters/processed_time_df.csv"
## output files
df_train_loc = "D:/samarth/Desktop/PriceOp/Project/train_data/"
modelDir = "D:/samarth/Desktop/PriceOp/Project/Models/"
df_model_performance_loc="D:/samarth/Desktop/PriceOp/Project/model_performance_data/"

## test whether the model exists or not
try:
    dirfilename = pd.read_csv(modelDir+'model_name.csv')
    model_exist = True
except:
    model_exist = False

## if model_exist, means the pipeline for the first has finished, means the training data are ready for retraining
if model_exist:
    ################################################## 2: read into aggregated sales data
    df_sales = pd.read_csv(df_sales_loc+'df_sales.csv')
    df_sales = df_sales.fillna(0)
    df_sales = df_sales.drop(["StoreID", "ProductID"], axis=1)
    df_sales = df_sales.rename(columns={'DepartmentID':'department_id', 'BrandID':'brand_id'})
    ## get the time the model is built
    model_time = df_sales['week_start'].max()

    ################################################## 3: feature engineering: build the features
    ## calculate relative price and discount for train data
    competing_group = ['week_start', 'store_id', 'department_id']
    df_train_price_sum = df_sales.groupby(competing_group).agg("sum")['price'].to_frame().reset_index(drop=False)
    df_train_price_sum = df_train_price_sum.rename(columns={'price':'price_sum'})

    df_train_price_count = df_sales.groupby(competing_group).agg("count")['price'].to_frame().reset_index(drop=False)
    df_train_price_count = df_train_price_count.rename(columns={'price':'count'})

    df_train = pd.merge(df_sales, df_train_price_sum, on=competing_group)
    df_train = pd.merge(df_train, df_train_price_count, on=competing_group)

    df_train['rl_price'] = df_train['price'] * df_train['count'] /  df_train['price_sum']
    df_train['discount'] = df_train['MSRP'] - df_train['price'] / df_train['MSRP']

    df_train = df_train.drop(["price_sum", "count"], axis=1)
    
    ################################################## 4: prepare the train data for modeling
    ## define categorical features, numerical features as well as label, which used in modeling
    features_categorical_train = ["department_id", "brand_id"]
    features_numerical_train = ["price", "AvgHouseholdIncome", "AvgTraffic", "rl_price", "discount"]
    label_train = ["sales"]

    for each in features_categorical_train:
        df_train[each] = df_train[each].astype("category")

    features_modeled_train = label_train + features_numerical_train + features_categorical_train

    df_train_modeling = df_train[features_modeled_train]

    X = df_train_modeling.iloc[:,1:].values  
    y = df_train_modeling.iloc[:,0].values

    ################################################## 5: train random forest regression model
    ## random forest
    ## train model
    rfModel = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=0)
    # Predict on train data
    rfModel.fit(X, y)
    # Predict on train data
    predictions = rfModel.predict(X)
    ## Evaluation of the model
    df_model_performance = pd.read_csv(df_model_performance_loc+'df_model_performance.csv')
    df_model_performance = df_model_performance.append([{'model_time': model_time, 'RMSE': np.sqrt(metrics.mean_squared_error(y, predictions)), 'R2': metrics.r2_score(y, predictions)}], ignore_index=True)
    df_model_performance.to_csv(df_model_performance_loc+'df_model_performance.csv', index=False)

    ################################################## 6: save model
    ## save model
    with open(processed_time_d_loc) as f:
        processed_time_d = csv.reader(f, delimiter=',')
        processed_time_d_list = list(processed_time_d)
    dirfilename = modelDir + "RandomForestRegression_until_" + processed_time_d_list[1][1]
    joblib.dump(rfModel, dirfilename)
    ## save Model file path information
    model_name = pd.read_csv(modelDir+'model_name.csv')
    model_name = model_name.append([{'model_name': dirfilename}], ignore_index=True)
    model_name.to_csv(modelDir+'model_name.csv', index=False)
    ## save the train data
    df_train.to_csv(df_train_loc+'df_train.csv', index=False)