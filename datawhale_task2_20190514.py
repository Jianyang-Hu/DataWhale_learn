# -*- coding: utf-8 -*-
# @version : Python3.6
# @Time    : 2019/5/14 22:02
# @Author  : Jianyang-Hu
# @contact : jianyang1993@163.com
# @File    : datawhale_task2_20190514.py
# @Software: PyCharm

# 导入库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

# 导入数据
data=pd.read_csv(r'E:\Python3.6\practice\DataWhale\First_task_20190511\task2_20190514.csv',encoding='gbk')
# print(data.head())
#样本及特征数量
# print(data.shape)
# 特征
# print(data.describe())

#特征频率
freq_rate = []
for col in data.columns:
    if (data[col].value_counts(dropna=False, normalize=True).max()) > 0.5:
        freq_rate.append(col)
# 删除status
freq_rate.remove('status')

#存储的特征频率过大，删除
data.drop(freq_rate, axis=1, inplace=True)
# print(data.describe())

# 时间处理
time_col = ['latest_query_time', 'loans_latest_time']
#待处理数据
unknown_col = ['loans_latest_day', 'latest_query_day', 'loans_long_time', 'first_transaction_day', 'first_transaction_time']

data['latest_query_time'] = pd.to_datetime(data['latest_query_time'])
data['latest_query_time_year'] = data['latest_query_time'].dt.year
data['latest_query_time_month'] = data['latest_query_time'].dt.month
data['latest_query_time_weekday'] = data['latest_query_time'].dt.weekday

data['loans_latest_time'] = pd.to_datetime(data['loans_latest_time'])
data['loans_latest_time_month'] = data['loans_latest_time'].dt.month
data['loans_latest_time_weekday'] = data['loans_latest_time'].dt.weekday
data = data.drop(['latest_query_time', 'loans_latest_time'], axis=1)

# 删除与业务无关数据
useless_col = ['custid', 'trade_no', 'id_name']
data.drop(useless_col, axis=1, inplace=True)

# 使用IV值筛选特征
def cal_WOE(dataset, col, target):
    # 对特征进行统计分组
    subdata = pd.DataFrame(dataset.groupby(col)[col].count())
    # 每个分组中响应客户的数量
    suby = pd.DataFrame(dataset.groupby(col)[target].sum())
    # subdata 与 suby 的拼接
    data = pd.DataFrame(pd.merge(subdata, suby, how='left', left_index=True, right_index=True))

    # 相关统计，总共的样本数量total，响应客户总数b_total，未响应客户数量g_total
    b_total = data[target].sum()
    total = data[col].sum()
    g_total = total - b_total

    # WOE公式
    data["bad"] = data.apply(lambda x:round(x[target]/b_total, 100), axis=1)
    data["good"] = data.apply(lambda x:round((x[col] - x[target])/g_total, 100), axis=1)
    data["WOE"] = data.apply(lambda x:np.log(x.bad / x.good), axis=1)
    return data.loc[:, ["bad", "good", "WOE"]]


def cal_IV(dataset):
    dataset["IV"] = dataset.apply(lambda x:(x["bad"] - x["good"]) * x["WOE"], axis=1)
    IV = sum(dataset["IV"])
    return IV





