import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix

data=pd.read_csv('Classified Data',index_col=0)
df=pd.DataFrame(data)
print(df.info())
print(df.head())
scaler=StandardScaler()
scaler.fit(df.drop('TARGET CLASS',axis=1))
scaled=scaler.transform(df.drop('TARGET CLASS',axis=1))
print(scaled)
df_feat=pd.DataFrame(scaled,columns=df.columns[:-1])
print(df_feat.head())
X=df_feat
y=df['TARGET CLASS']
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=101)
print(x_test)

knn=KNeighborsClassifier(n_neighbors=1) #small datasets
knn.fit(x_train,y_train)
pred=knn.predict(x_test)
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))

err_rate=[]
for i in range(1,40):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train,y_train)
    pred_i=knn.predict(x_test)
    err_rate.append(np.mean(pred_i!=y_test))
print(err_rate)

knn=KNeighborsClassifier(n_neighbors=17) #large datasets
knn.fit(x_train,y_train)
pred=knn.predict(x_test)
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))