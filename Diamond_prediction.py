# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 11:59:09 2022

@author: Farhan
credit: kong.kah.chun
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

#1. Read CSV file
filepath = r"C:\Users\ACER\Desktop\SHRDC\Git\AI-05 repo 2(actual)\Dataset\diamonds.csv"
data_diamond = pd.read_csv(filepath)

#%%
#2. The column which is not useful as a feature will be remove
data_diamond = data_diamond.drop('Unnamed: 0', axis=1) #no header

#3. Split the data into features and label
diamond_features = data_diamond.copy()
diamond_label = diamond_features.pop('price')

#4. Check the split data
print("------------------Features--------------------")
print(diamond_features.head())
print("------------------Label-----------------------")
print(diamond_label.head())

#%%
#5. Ordinal encode categorical features
#orders matter
cut_categories = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
color_categories = ['J','I','H','G','F','E','D']
clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']
ordinal_encoder = OrdinalEncoder(categories=[cut_categories,color_categories,
                                             clarity_categories])
diamond_features[['cut','color','clarity']] = ordinal_encoder.fit_transform(diamond_features[['cut','color','clarity']])

#Check the transformed Features
print("------------------Transformed Features-----------------------")
print(diamond_features.head())

#6. Split the features and labels into train-validation-test sets (60:20:20 split)
SEED = 12345
x_train, x_iter, y_train, y_iter = train_test_split(diamond_features,diamond_label,
                                                    test_size=0.4,random_state=SEED)
x_val, x_test, y_val, y_test = train_test_split(x_iter,y_iter,test_size=0.5,
                                                random_state=SEED)

#7. Perform feature scaling, using training data for fitting
scaler = StandardScaler()
scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

#### Data preparation is completed ####

#%%
#8. Create a feedforward neural network using TensorFlow Keras
number_input = x_train.shape[-1]
model = tf.keras.Sequential()
model.add(tf.keras.layers.InputLayer(input_shape = number_input))
model.add(tf.keras.layers.Dense(128,activation='elu')) 
model.add(tf.keras.layers.Dense(64,activation='elu')) 
model.add(tf.keras.layers.Dense(32,activation='elu')) 
model.add(tf.keras.layers.Dropout(0.3)) 
model.add(tf.keras.layers.Dense(1)) #output layer

#9. Compile the model
model.compile(optimizer='adam',loss='mse',metrics=['mae','mse'])

#%%
#10. Train and evaluation of model
#Define callback functions: EarlyStopping and Tensorboard
base_log_path = r"C:\Users\ACER\Desktop\SHRDC\Deep learning\TensorBoard\p2_log" 
log_path = os.path.join(base_log_path, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_path)
es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=2)
EPOCHS = 100
BATCH_SIZE = 64
history = model.fit(x_train,y_train,validation_data=(x_val, y_val), batch_size=BATCH_SIZE,
                   epochs=EPOCHS, callbacks=[tb_callback, es_callback])

#%%
#11. Evaluate with test data for wild testing
test_result = model.evaluate(x_test,y_test,batch_size=BATCH_SIZE)
print(f"Test loss = {test_result[0]}")
print(f"Test MAE = {test_result[1]}")
print(f"Test MSE = {test_result[2]}")

#%%
#12. Plot a graph of prediction vs label on test data
predictions = np.squeeze(model.predict(x_test))
labels = np.squeeze(y_test)
plt.plot(predictions,labels,".")
plt.xlabel("Predictions")
plt.ylabel("Labels")
plt.title("Graph of Predictions vs Labels with Test Data")
save_path = r"C:\Users\ACER\Desktop\SHRDC\Git\AI-05 repo 2(actual)\Image"
plt.savefig(os.path.join(save_path,"Result.png"),bbox_inches='tight')
plt.show()

