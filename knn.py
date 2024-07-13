import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
data=pd.read_csv('Classified Data',index_col=0)
df=pd.DataFrame(data)
# print(df.info())
# print(df.head())
scaler=StandardScaler()
scaler.fit(df.drop('TARGET CLASS',axis=1))
scaled=scaler.transform(df.drop('TARGET CLASS',axis=1))
# print(scaled)
df_feat=pd.DataFrame(scaled,columns=df.columns[:-1])
# print(df_feat.head())
X=df_feat
y=df['TARGET CLASS']
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=101)
print(x_test)