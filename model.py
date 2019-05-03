'''
Programmer ==> D!m@r

'''
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

#read csv
dataname = "data1.csv"
data =pd.read_csv(dataname,delimiter=';')
nextdata = data
data = data.shift(periods=1, fill_value=0)
x =data[['Open','Close','Low','High']]
data['dif']=nextdata['High']-nextdata['Low']
y = data['dif']

#smoothing the data
x = x.rolling(10).mean()
y = y.rolling(10).mean()
x=x[11:]
y=y[11:]
x=x.values
y=y.values

#data scaling with sklearn
x = preprocessing.scale(x)
y = preprocessing.scale(y)

#split data to x_train y_train x_test y_test
x_train = x[:int(0.9*len(x))]
y_train=y[:int(0.9*len(x))]

x_test=x[int(0.9*len(x)):]
y_test=y[int(0.9*len(x)):]

#model
print('Training started...')
clf = MLPRegressor(solver='adam', alpha=1e-5,max_iter=100,learning_rate='adaptive',
                     hidden_layer_sizes=(128,128,64), random_state=1)
clf.fit(x_train, y_train)                         
score = clf.score(x_test,y_test)
print('Accuracy : ',round(score,3),'%')

#make predictions
predictions = clf.predict(x_test)


#plot predictions
plt.plot(predictions,'y',label='Predictions')
plt.plot(y_test,label='True Data')
plt.ylabel('Real Data/Predictions')
plt.legend()
plt.show()
