################################################## 0: import libraries and define functions
import numpy as np
import pandas as pd
import csv
from datetime import datetime, timedelta
from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder
from gurobipy import *
from functools import reduce
from itertools import groupby

def reduceByKey(func, iterable):
    """Reduce by key (equivalent to the Spark counterpart)
    1. Sort by key
    2. Group by key yielding (key, grouper)
    3. For each pair yield (key, reduce(func, last element of each grouper))
    """
    get_first = lambda p: p[0]
    get_second = lambda p: p[1]
    return map(
        lambda l: (l[0], reduce(func, map(get_second, l[1]))),
        groupby(sorted(iterable, key=get_first), get_first)
    )

## concatenate variables at particular index with string s in p
def concatenate_variables(p, index, s):
    p_tmp = np.array([str(pi) for pi in p])
    return p + [s.join(p_tmp[index])]

## extract elements from list based on index
def extract_from_list_based_on_index(p, index):
    return tuple([p[i] for i in index])

## input p (df_batch_scored_names,pred_demand) => ((competing_group_vars+'price_sum'),['product_id','price','obj','cost','msrp'])
def reduce_key_value_map_IPk(p):
    key = extract_from_list_based_on_index(p, index_reduce_key_vars)
    value = extract_from_list_based_on_index(p, index_reduce_value_vars)
    product_id, msrp, cost, price = value
    sales = max(0, round(p[-1]))
    obj = (price - cost) * sales
    return (key, [[product_id, price, obj, cost, msrp]])

################################################## 1: define paths of input files and output files
## input paths
df_train_loc = "D:/samarth/Desktop/PriceOp/Project/train_data/"
df_sales_loc = "D:/samarth/Desktop/PriceOp/Project/aggregated_sales_data/"
processed_time_d_loc = "D:/samarth/Desktop/PriceOp/Project/publicparameters/processed_time_df.csv"
modelDir = "D:/samarth/Desktop/PriceOp/Project/Models/"
## output paths
price_change_d_loc = "D:/samarth/Desktop/PriceOp/Project/medium_results/"
opt_results_d_loc = "D:/samarth/Desktop/PriceOp/Project/opt_results_data/"

################################################## 2: read into input data and construct df_test
## read into train data and sales data
df_train = pd.read_csv(df_train_loc+'df_train.csv')
df_sales = pd.read_csv(df_sales_loc+'df_sales.csv')
## get the start date and end date of the current sales cycle
with open(processed_time_d_loc) as f:
    processed_time_d = csv.reader(f, delimiter=',')
    processed_time_d_list = list(processed_time_d)
processed_time_d = [datetime.strptime(s, '%Y-%m-%d').date() for s in processed_time_d_list[1]]

## construct the df_test
## construct the df_test based on the lastest week's data
df_sales_date_max = datetime.strftime(processed_time_d[0], '%Y-%m-%d')
#print(df_sales.tail())
df_test = df_sales[df_sales.week_start == df_sales_date_max]
df_test = df_test.rename(index=str, columns={"week_start": "week_start_origin"})
## how many weeks to predict ahead: num_weeks_ahead
# Since we are using synthetic data (with the simulator), it is a must to keep num_weeks_ahead=0, because there are some
# dependencies with the current version of simulator
# Note: if working with the real data, we could change this parameter to provide suggested price several weeks ahead the
# next week
num_weeks_ahead = 0
week_start = datetime.strftime(datetime.strptime(df_sales_date_max, '%Y-%m-%d') + timedelta(7*(num_weeks_ahead+1)), '%Y-%m-%d')
df_test['week_start'] = [week_start] * df_test.shape[0]

## only do optimization for stores in the treatment group
df_test = df_test[df_test.group_val == 'treatment']
df_test = df_test.rename(columns={"DepartmentID": "department_id", "BrandID": "brand_id"})
df_test = df_test[['store_id', 'product_id', 'department_id', 'brand_id', 'MSRP', 'Cost', 'AvgHouseholdIncome', 
                   'AvgTraffic', 'week_start']]
df_test = df_test.drop_duplicates()

################################################## 3: load the model built last time
dirfilename_load = pd.read_csv(modelDir+'model_name.csv')
dirfilename_load = str(dirfilename_load.iloc[-1,0])
rfModel = joblib.load(dirfilename_load)

## Direct Integer Programming Optimization function (not using bound algorithm)
def IPk(p_tmp):
    global df_optimal
    instance_reduce_key = p_tmp[0]
    instance_reduce_value = p_tmp[1]
    instance_reduce_value = pd.DataFrame(instance_reduce_value, columns=['product_id', 'price', 'obj', 'cost', 'msrp'])
    instance_reduce_value = instance_reduce_value.sort_values(by=['product_id', 'price'], ascending=[True, True])
    prod_list = instance_reduce_value['product_id'].tolist()
    obj_list = instance_reduce_value['obj'].tolist()
    price_list = instance_reduce_value['price'].tolist()
    products_num = len(instance_reduce_value['product_id'].unique())
    ## get whether_valid_price, if equal to 1, valid, if equal to 0, invalid
    ## invalid ones needed to set into the constrains
    instance_reduce_value['whether_valid_price'] = instance_reduce_value.apply(
        lambda row: 1 if row['price'] >= row['cost'] and row['price'] <= row['msrp'] else 0, axis=1)
    for i in range(products_num):
        whether_valid_price_individual_product = instance_reduce_value['whether_valid_price'].iloc[
                                                 price_K * i:price_K * (i + 1) - 1]
        if all(whether_valid_price == 0 for whether_valid_price in whether_valid_price_individual_product):
            abs_diff_price = abs(
                instance_reduce_value['msrp'].iloc[price_K * i:price_K * (i + 1)] - instance_reduce_value[
                                                                                            'price'].iloc[
                                                                                        price_K * i:price_K * (
                                                                                        i + 1)])
            row_index = i * price_K + min(enumerate(abs_diff_price), key=lambda x: x[1])[0]
            instance_reduce_value.iloc[row_index, instance_reduce_value.columns.get_loc('whether_valid_price')] = 1
    
    whether_valid_price_list = instance_reduce_value['whether_valid_price'].tolist()
    
    model = Model()
    model.setParam("OutputFlag", 0)

    # Add variables to model
    vars = []
    for j in range(len(instance_reduce_value)):
        vars.append(model.addVar(vtype=GRB.BINARY))

    # Populate objective
    obj = LinExpr()
    for j in range(len(instance_reduce_value)):
        obj += obj_list[j]*vars[j]
    model.setObjective(obj, GRB.MAXIMIZE)        

    # Populate constr 1 matrix
    for i in range(products_num):
        expr1 = LinExpr()
        for j in range(len(instance_reduce_value)):
            if ((j >= price_K * i) & (j < price_K * (i + 1))):
                expr1 += vars[j]
        model.addConstr(expr1, GRB.EQUAL, 1)

    # Populate constr 2 matrix
    expr2 = LinExpr()
    for j in range(len(instance_reduce_value)):
        expr2 += price_list[j]*vars[j]
    model.addConstr(expr2, GRB.EQUAL, instance_reduce_key[3])
    
    # Populate constr 3 matrix
    expr3 = LinExpr()
    for j in range(len(instance_reduce_value)):
        if whether_valid_price_list[j] == 0:
            expr3 += vars[j]
    model.addConstr(expr3, GRB.EQUAL, 0)

    #model.write("File.lp")
    # Solve
    model.optimize()
    
    if model.Status == GRB.OPTIMAL:
        obj_val = model.objVal
        solution = []
        for v in model.getVars():
            solution.append(v.X)
        solution = [a*b for a,b in zip(solution,price_list)]
        solution = list(filter(lambda a: a != 0, solution))
        for k in range(products_num):
            df_optimal = df_optimal.append({'week_start' : instance_reduce_key[2],
                                            'department_id' : instance_reduce_key[1],
                                            'store_id' : instance_reduce_key[0],
                                            'price_sum' : instance_reduce_key[3],
                                            'product_id' : prod_list[price_K * k],
                                            'profit_obj_val' : obj_val,
                                            'price' : solution[k]},
                                            ignore_index=True)
    
    model.reset()
    #print("\n------------------------------------------------------------------------------\n")

################################################## 4: optimization
##define categorical features, which used in modeling
features_categorical_train_and_test = ["department_id", "brand_id"]
features_numerical_train_and_test = ["price", "AvgHouseholdIncome", "AvgTraffic", "rl_price", "discount"]
##feature index the categorical features
for each in features_categorical_train_and_test:
    df_test[each] = df_test[each].astype("category")

## step 4.1: input for whole prize optimization
## input 1: df (Spark DataFrame) here: M*CP
df = df_test
df = df.reset_index(drop=True)
## input 2: competing_group_vars
competing_group_vars = ['week_start', 'department_id', 'store_id']
## add an index number for competing groups
df_names = df.columns.tolist()
concatenate_index = [df.columns.get_loc(c) for c in df.columns if c in competing_group_vars]
df = [concatenate_variables(list(row), concatenate_index, '_') for index, row in df.iterrows()]
df = pd.DataFrame(df, columns=df_names+["competing_group_index_column_name"])
le = LabelEncoder()
df[['competing_group_index_column_name']] = le.fit_transform(df[['competing_group_index_column_name']])
df = df.rename(index=str, columns={"competing_group_index_column_name": "competing_group_index_column_name_StringIndexed"})
##feature index the categorical features
for each in features_categorical_train_and_test:
    df[each] = df[each].astype("category")
df['store_id'] = df['store_id'].astype("category")

price_K = 10

##calculate the range of price and price sum
min_cost_df = pd.DataFrame(df.groupby(competing_group_vars)['Cost'].agg(min))
min_cost_df = min_cost_df.rename(columns={'Cost':'min_cost'})
min_cost_df = min_cost_df.reset_index(drop=False)

max_msrp_df = pd.DataFrame(df.groupby(competing_group_vars)['MSRP'].agg(max))
max_msrp_df = max_msrp_df.rename(columns={'MSRP':'max_msrp'})
max_msrp_df = max_msrp_df.reset_index(drop=False)

count_df = pd.DataFrame(df.groupby(competing_group_vars)['MSRP'].agg('count'))
count_df = count_df.rename(columns={'MSRP':'count'})
count_df = count_df.reset_index(drop=False)

price_range_df=pd.merge(df, min_cost_df, on=competing_group_vars)
price_range_df=pd.merge(price_range_df, max_msrp_df, on=competing_group_vars)
price_range_df=pd.merge(price_range_df, count_df, on=competing_group_vars)

price_single_range_df = [[pr[-5]]+[pr[2]]+[pr[0]]+[pr[-1]]+[i] for index, pr in price_range_df.iterrows() for i in
                         np.linspace(pr[-3], pr[-2], price_K).tolist()]
price_single_range_df = pd.DataFrame(price_single_range_df , columns=competing_group_vars + ["count", "price"])

price_sum_range_df = [[pr[-5]]+[pr[2]]+[pr[0]]+[i] for index, pr in price_range_df.iterrows() for i in
                      np.linspace(pr[-1] * pr[-3], pr[-1] * pr[-2], (price_K - 1) * pr[-1] + 1).tolist()]
price_sum_range_df = pd.DataFrame(price_sum_range_df, columns=competing_group_vars + ["price_sum"])

price_single_sum_range_df=pd.merge(price_single_range_df, price_sum_range_df, on=competing_group_vars).drop_duplicates()
price_single_sum_range_df['department_id'] = price_single_sum_range_df['department_id'].astype("category")
price_single_sum_range_df['store_id'] = price_single_sum_range_df['store_id'].astype("category")

df_price_added = pd.merge(df, price_single_sum_range_df, on=competing_group_vars, how="outer").drop_duplicates()

df_price_added['rl_price'] = df_price_added['price'] * df_price_added['count'] /  df_price_added['price_sum']
df_price_added['discount'] = df_price_added['MSRP'] - df_price_added['price'] / df_price_added['MSRP']

features_modeled_test = features_numerical_train_and_test + features_categorical_train_and_test  ##no label, only features
df_price_added1 = df_price_added[features_modeled_test]

predictions = pd.DataFrame(rfModel.predict(df_price_added1), columns=['predictions'])

df_for_output = pd.concat([df_price_added,predictions],axis=1)

## 4.2.2: reduce the df_for_output to the competing group level

## calculate the objective function: (price-cost)*pred_demand
reduce_key_vars = ['store_id', 'department_id', 'week_start', 'price_sum']
reduce_value_vars = ['product_id', 'MSRP', 'Cost', 'price']
index_reduce_key_vars = [df_price_added.columns.get_loc(c) for c in df_price_added.columns if c in reduce_key_vars]
index_reduce_value_vars = [df_price_added.columns.get_loc(c) for c in df_price_added.columns if c in reduce_value_vars]

## reducebyKey: competing_vars + ['price_sum','product_id','price']
df_reduced = df_for_output.apply(reduce_key_value_map_IPk, axis=1)
df_reduced = list(reduceByKey(lambda x, y: x + y, df_reduced))


df_optimal_names = ['week_start', 'department_id', 'store_id', 'price_sum', 'product_id', 'price']
df_optimal = pd.DataFrame(columns = df_optimal_names)

df_reduced = list(map(IPk, df_reduced))

grouping_vars = ['week_start', 'store_id', 'department_id']
df_best_profit = pd.DataFrame(df_optimal.groupby(grouping_vars)['profit_obj_val'].agg(max))
df_best_profit = df_best_profit.reset_index(drop=False)

for each in ['store_id', 'department_id']:
    df_optimal[each] = df_optimal[each].astype("int64")

df_optimal2 = pd.merge(df_optimal, df_best_profit, on=grouping_vars+['profit_obj_val'])

for each in ['department_id', 'store_id']:
    df_optimal2[each] = df_optimal2[each].astype("category")

df_optimal3 = pd.merge(df_optimal2, df_for_output, on=['week_start', 'department_id', 'store_id', 
                                                       'price_sum', 'product_id', 'price'])

price_change_names = ['product_id', 'store_id', 'week_start', 'price']
price_change_df = df_optimal3[price_change_names]
price_change_df_name = price_change_d_loc+'suggested_prices_'+processed_time_d[1].strftime('%Y-%m-%d')+'.csv'
price_change_df.to_csv(price_change_df_name, index=False)

df_opt = df_optimal3[['week_start', 'store_id', 'product_id', 'MSRP', 'Cost', 'price', 'predictions']]
df_opt = df_opt.rename(index=str, columns={"price": "price_recommendation", "predictions": "demand_forecast"})
df_opt_name = opt_results_d_loc + 'recommendations_for_' + processed_time_d[1].strftime('%Y-%m-%d')+'.csv'
df_opt.to_csv(df_opt_name, index=False)

if (os.path.exists('D:/samarth/Desktop/PriceOp/Project/opt_results_data/df_recommendations.csv')):
    df_opt_cumulative = pd.read_csv(opt_results_d_loc+'df_recommendations.csv')
    df_opt_cumulative = pd.concat([df_opt_cumulative, df_opt], ignore_index=True)
    df_opt_cumulative.to_csv(opt_results_d_loc+'df_recommendations.csv', index=False)
else:
    df_opt.to_csv(opt_results_d_loc+'df_recommendations.csv', index=False)