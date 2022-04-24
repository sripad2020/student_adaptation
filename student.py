import pandas as pd
data=pd.read_csv('students_adaptability_level_online_education.csv')
print(data.columns)
from sklearn.preprocessing import LabelEncoder
lab=LabelEncoder()
data['adapted']=lab.fit_transform(data['Adaptivity Level'])
data['network']=lab.fit_transform(data['Network Type'])
data['finance_condition']=lab.fit_transform(data['Financial Condition'])
data['student']=lab.fit_transform(data['IT Student'])
data['insti_type']=lab.fit_transform(data['Institution Type'])
data['lms']=lab.fit_transform(data['Self Lms'])
data['load']=lab.fit_transform(data['Load-shedding'])
data['net_tpe']=lab.fit_transform(data['Internet Type'])
print(data['adapted'])
print(data['network'])
print(data['finance_condition'])
print(data['student'])
print(data['insti_type'])
print(data['lms'])
print(data['load'])
print(data['net_tpe'])
from sklearn.model_selection import train_test_split
x=data[['network','net_tpe','lms','insti_type','student','finance_condition','load']]
y=data['adapted']
x_tr,x_te,y_tr,y_te=train_test_split(x,y,test_size=0.35)
import keras
from keras.models import Sequential
from keras.layers import Dense
import keras.optimizers,keras.activations,keras.losses,keras.metrics
model=Sequential()
model.add(Dense(input_dim=x.shape[1],activation=keras.activations.softmax,units=2))
model.add(Dense(input_dim=x.shape[1],activation=keras.activations.softmax,units=2))
model.add(Dense(units=1,activation=keras.activations.softmax))
model.compile(loss=keras.losses.categorical_crossentropy,metrics='accuracy',optimizer='adam')
model.fit(x_tr,y_tr,batch_size=13,epochs=13)
pred=model.predict([[2,0,0,1,0,1,1]])
print(pred)