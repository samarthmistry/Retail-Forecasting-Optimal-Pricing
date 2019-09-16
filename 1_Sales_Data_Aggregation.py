################################################## 0: import modules
import os
import numpy as np
from datetime import datetime, timedelta
import csv
import pandas as pd
from sklearn.cluster import KMeans
from pandas.io.json import json_normalize
from functools import reduce

################################################## 1: define paths
## input paths
sales_and_pc_raw_d_loc = "D:/samarth/Desktop/PriceOp/Project/rawdata/"
products_d_loc = "D:/samarth/Desktop/PriceOp/Project/publicparameters/products.csv"
stores_d_loc = "D:/samarth/Desktop/PriceOp/Project/publicparameters/stores.csv"
processed_time_d_loc = "D:/samarth/Desktop/PriceOp/Project/publicparameters/processed_time_df.csv"
## output paths
df_sales_loc = "D:/samarth/Desktop/PriceOp/Project/aggregated_sales_data/"
## input path if exists, output path if not exist
df_stores_loc = "D:/samarth/Desktop/PriceOp/Project/publicparameters/stores_processed"

################################################## 2: data aggregation
########################## 2.1 read in products, stores, sales, price_changes data
## get the start date and end date of the current sales cycle
with open(processed_time_d_loc) as f:
    processed_time_d = csv.reader(f, delimiter=',')
    processed_time_d_list = list(processed_time_d)
processed_time_d = [datetime.strptime(s, '%Y-%m-%d').date() for s in processed_time_d_list[1]]

################# 2.1.1 products data
## import file
df_products = pd.read_csv(products_d_loc)
## specify categorical fields and numerical 
header_products=['DepartmentID', 'BrandID', 'ProductID', 'MSRP', 'Cost']
fields_categorical_products=["DepartmentID","BrandID","ProductID"]
fields_numerical_products=list(set(header_products)-set(fields_categorical_products))
## convert
for each in fields_categorical_products:
    df_products[each] = df_products[each].astype('category')
for each in fields_numerical_products:
    df_products[each] = df_products[each].astype('float32')

################ 2.1.2 stores data
df_stores = pd.read_csv(stores_d_loc)
df_stores['StoreID'] = df_stores['StoreID'].astype('category')
# if the store data with group_val already exists, read in
try:
    df_stores_join = df_stores[['StoreID', 'AvgHouseholdIncome', 'AvgTraffic','group_val']]
    df_stores['group_val'] = df_stores['group_val'].astype('category')
# else: construct the group_val through clustering
except:
    ## split stores into control and treatment group according to store attributes
    ## (these attributes should be strongly related to store sales)
    df_stores['AvgHouseholdIncome_std'] = (df_stores['AvgHouseholdIncome']-df_stores['AvgHouseholdIncome'].mean())/df_stores['AvgHouseholdIncome'].std()
    df_stores['AvgTraffic_std'] = (df_stores['AvgTraffic']-df_stores['AvgTraffic'].mean())/df_stores['AvgTraffic'].std()
    store_group_col_names=['AvgHouseholdIncome_std', 'AvgTraffic_std']

    ## perform the clustering to cluster stores according to the attributes
    store_number=df_stores.shape[0]
    if store_number<=3:
        cluster_number=1
    else:
        cluster_number=3

    clusters = KMeans(n_clusters=cluster_number).fit(df_stores.loc[:,store_group_col_names].values)
    df_stores['cluster_val'] = clusters.labels_+1
    df_stores['rand']=np.random.random_sample(df_stores.shape[0])
    df_stores['rank'] = df_stores.groupby(by=['cluster_val'])['rand'].transform(lambda x: x.rank()).astype('int')

    for val in df_stores['cluster_val'].unique():
        tmp=df_stores[df_stores['cluster_val']==val]
        tmp['cume_dist']=[1-(tmp.loc[index, 'rank']/tmp['rank'].max()) for index in tmp.index]
        for index in tmp.index:
            df_stores.loc[index, 'cume_dist']=tmp.loc[index, 'cume_dist']

    for val in df_stores['cluster_val'].unique():
        tmp2=df_stores[df_stores['cluster_val']==val]  
        for index in tmp2.index:        
            if (tmp2['rank'].max()>=2 and tmp2.loc[index, 'cume_dist']>=0.5):
                tmp2.loc[index, 'group_val']='control'
            elif (tmp2['rank'].max()>=2 and tmp2.loc[index, 'cume_dist']<0.5):
                tmp2.loc[index, 'group_val']='treatment'
            else:
                tmp2.loc[index, 'group_val']='other'
        for index in tmp2.index:
            df_stores.loc[index, 'group_val']=tmp2.loc[index, 'group_val']
    
    df_stores['group_val'] = df_stores['group_val'].astype('category') 
    df_stores.to_csv(stores_d_loc, index=False) 
    df_stores_join=df_stores[['StoreID', 'AvgHouseholdIncome', 'AvgTraffic','group_val']]

################# 2.1.3 sales data
## parsing sales json file and construct df_sales

sales_dates=[(processed_time_d[0]+timedelta(i+1)).strftime('%Y_%m_%d') for i in range((processed_time_d[1]-processed_time_d[0]).days)]

sales_raw_file_name = []
for i in range(1,7):
    for date in sales_dates:
        sales_raw_file_name.append(sales_and_pc_raw_d_loc+'sales_store'+str(i)+'_'+str(date)+'_00_00_00.json')

def json_to_salesdf(row):
    _, row = row
    df_json = json_normalize(row.Products)
    SalesLogDateTime = pd.Series([row.SalesLogDateTime]*len(df_json), name='SalesLogDateTime')
    StoreID = pd.Series([row.StoreID]*len(df_json), name='StoreID')
    Subtotal = pd.Series([row.Subtotal]*len(df_json), name='Subtotal')
    Tax = pd.Series([row.Tax]*len(df_json), name='Tax')
    Total = pd.Series([row.Total]*len(df_json), name='Total')
    TransactionDateTime = pd.Series([row.TransactionDateTime]*len(df_json), name='TransactionDateTime')
    return pd.concat([SalesLogDateTime,StoreID,Subtotal,Tax,Total,TransactionDateTime,df_json],axis=1)

df_sales=pd.DataFrame()

for name in sales_raw_file_name:
    df = pd.read_json(name)
    transactions_as_df = json_normalize(df["Transactions"])
    transactions_as_df.columns = [f"{subcolumn}" for subcolumn in transactions_as_df.columns]
    df = df.drop("Transactions", axis=1).merge(transactions_as_df, right_index=True, left_index=True)
    df['TransactionDateTime'] = df['TransactionDateTime'].astype('str')
    df['TransactionDateTime'] = pd.to_datetime(df['TransactionDateTime'], format='%Y-%m-%d %H:%M:%S', yearfirst=True)
    df = [*map(json_to_salesdf, df.iterrows())]    #returns a list of dataframes
    df = reduce(lambda df,y:df.append(y), df)   #glues them together
    df_sales=df_sales.append(df,ignore_index=True)
    
df_sales = df_sales.reset_index(drop=True)
    
df_sales = df_sales.rename(index=str, columns={"ProductID": "product_id", "StoreID": "store_id", "TransactionDateTime": "date_date"})
header_sales =['product_id', 'store_id', 'date_date']
df_sales = df_sales[header_sales]
df_sales['product_id'] = df_sales['product_id'].astype('category')
df_sales['store_id'] = df_sales['store_id'].astype('category')

################# 2.1.4 price changes data
## parsing price change json file and construct df_price_change

price_change_dates=[(processed_time_d[0]+timedelta(i*7)).strftime('%Y_%m_%d') for i in range(int((processed_time_d[1]-processed_time_d[0]).days/7))]

price_change_raw_file_name = []
for i in range(1,7):
    for date in price_change_dates:
        price_change_raw_file_name.append(sales_and_pc_raw_d_loc+'pc_store'+str(i)+'_'+str(date)+'_00_00_00.json')

df_price_change=pd.DataFrame()

for name in price_change_raw_file_name:
    df = pd.read_json(name)
    priceupdate_as_df = json_normalize(df["PriceUpdates"])
    priceupdate_as_df.columns = [f"{subcolumn}" for subcolumn in priceupdate_as_df.columns]
    df = df.drop("PriceUpdates", axis=1).merge(priceupdate_as_df, right_index=True, left_index=True)
    df['PriceDate'] = df['PriceDate'].astype('str')
    df['PriceDate'] = pd.to_datetime(df['PriceDate'], format='%Y-%m-%d', yearfirst=True)
    df_price_change=df_price_change.append(df,ignore_index=True)
    
df_price_change = df_price_change.reset_index(drop=True)
df_price_change = df_price_change.rename(index=str, columns={"ProductID": "product_id", "StoreID": "store_id", "PriceDate": "week_start", "Price": "price"})
df_price_change['product_id'] = df_price_change['product_id'].astype('category')
df_price_change['store_id'] = df_price_change['store_id'].astype('category')

########################## 2.2 aggregate the sales data and join with store and product attributes
## create week_start for df_sales
df_sales_date_min=pd.to_datetime(processed_time_d[0])
df_sales['week_start']=df_sales['date_date'].apply(lambda x:df_sales_date_min+timedelta((x-df_sales_date_min).days/7*7))

## aggregate df_sales on weekly basis
df_sales=df_sales.groupby(['week_start','store_id','product_id']).agg("count")
df_sales=df_sales.rename(columns={'date_date':'sales'})
df_sales=df_sales.reset_index(drop=False)
## join the df_sales with df_price_change, warning: under this scenario, every week each product in each department in each store has an entry in df_price_change
df_sales=pd.merge(df_sales, df_price_change, on=["week_start","store_id","product_id"])
## join df_sales with df_products and df_stores to get products and stores attributes
df_sales=pd.merge(df_sales, df_products, left_on="product_id", right_on="ProductID")
df_sales=pd.merge(df_sales, df_stores_join, left_on="store_id", right_on="StoreID")
df_sales['week_start']=df_sales['week_start'].dt.date

################################################## 3: data export
df_sales_date = processed_time_d[0]
df_sales.to_csv(df_sales_loc+'week_start_'+df_sales_date.strftime('%Y-%m-%d')+'.csv', index=False)

if (os.path.exists('D:/samarth/Desktop/PriceOp/Project/aggregated_sales_data/df_sales.csv')):
    df_sales_cumulative = pd.read_csv(df_sales_loc+'df_sales.csv')
    df_sales_cumulative = pd.concat([df_sales_cumulative, df_sales], ignore_index=True)
    df_sales_cumulative.to_csv(df_sales_loc+'df_sales.csv', index=False)
else:
    df_sales.to_csv(df_sales_loc+'df_sales.csv', index=False)
