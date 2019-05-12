# -*- coding: utf-8 -*-
# @version : Python3.6
# @Time    : 2019/5/11 22:53
# @Author  : Jianyang-Hu
# @contact : jianyang1993@163.com
# @File    : datawhale_firsttask_20190511.py
# @Software: PyCharm

import pandas as pd
from pandas import DataFrame

# import data set
data = pd.read_csv(r'E:\Python3.6\practice\DataWhale\First_task_20190511\datawhale_firsttask_finance_data.csv',encoding='gbk')
print(data.head())

# observing data characteristics
print(data.describe())

# Analyze field types, divided into object, int64 and float64
print(data.dtypes)

# View missing conditions   -- dtypes: float64(70), int64(13), object(7)
print(data.info())


#calculate the missing rate and unique rate
data_loss=DataFrame(data.isnull().sum(),columns=['loss_num'])
data_unique=DataFrame(data.nunique(),columns=['unique_num'])
data_loss_unique=pd.merge(data_loss,data_unique,right_index=True,left_index=True)
data_loss_unique['loss_rate']=round(data_loss_unique['loss_num']/data.shape[0],4)
data_loss_unique['unique_rate']=round(data_loss_unique['unique_num']/data.shape[0],4)
print(data_loss_unique.head().T)
print(data_loss_unique.sort_values(by='unique_rate',ascending=False))



#drop   Unnamed: 0、custid、id_name、source、trade_no、bank_card_no .loss_rate>0.8&unique_rate=1
columns_drop=['Unnamed: 0','custid','source','student_feature','bank_card_no','trade_no','id_name']
data.drop(columns=columns_drop,axis=1,inplace=True)

# 将reg_preference_for_trad 进行编码转化为数值
map_dict={'一线城市':1,'二线城市':2,'三线城市':3,'境外':4,'其他':0}
data['reg_preference_for_trad']=data['reg_preference_for_trad'].map(map_dict)
print(data.head())


# 数据标准化/归一化
for i in data.columns:
    if data[i].dtype==object:
            print(i)
    else:
        data[i]=(data[i]-data[i].min())/(data[i].max()-data[i].min())
print(data.head())


# 数据清洗完成，切分数据,训练预测
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(data,data['status'],test_size=0.3,random_state=2018)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

#

# SVC
# from sklearn.svm import SVC
# svc = SVC(kernel="rbf", C=1)
# svc.fit(X_train, y_train)
# svc.score(X_test, y_test)




# 随机森林
from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier(n_estimators=500, min_samples_split=2, bootstrap=True, max_depth=4, max_features=6)
RF.fit(X_train, y_train)
RF.score(X_test, y_test)






