# Import the libraries
import numpy as np
import pandas as pd
import tensorflow as tf

#print(tf.__version__)

# Part 1 - Data Preprocessing

# Importing the database
dataset = pd.read_csv('D:/Deep Learning A-Z/Course Databases/Part 1 - Artificial Neural Networks (ANN)/Churn_Modelling.csv')
#print(dataset)
X = dataset.iloc[:, 3:-1].values # inputs emitting the first three columns
#print(X)
y = dataset.iloc[:, -1].values # output
# print(y)

# Encoding Categorical data
# 'Gender' Column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:,2] = le.fit_transform(X[:,2])
#print(X[0:7])

# Geography column
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1])],remainder = 'passthrough')
X = np.array(ct.fit_transform(X))
# print(X)

# Splitting the data into training and testing data sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.2,random_state = 0)
# print(X_train)
# print(y_train)
# print(X_test)
# print(y_test)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
#print(len(X_train))
X_test = sc.fit_transform(X_test)

# Building the ANN

# Initializing the ANN
ann = tf.keras.models.Sequential()

# adding input and first hidden layers
ann.add(tf.keras.layers.Dense(units=6,activation='relu'))
# add a second hidden layer
ann.add(tf.keras.layers.Dense(units=6,activation='relu'))
# adding output layer
ann.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))

#Training the Ann

# compiling the ann
import keras.losses as lss
ann.compile(optimizer = 'adam',loss = lss.binary_crossentropy, metrics =['accuracy'] )


# training
ann.fit(X_train,y_train,batch_size = 32, epochs = 100)

# testing one sample
# Geography: France, credit score 600, gender male, age 40, tenure 3, balance 60k,
# num products 2, active member yes, salary 50k
result=ann.predict(sc.transform([[1,0,0,600,1,40,3,60000,2,1,1,50000]]))>0.5
print(result)
