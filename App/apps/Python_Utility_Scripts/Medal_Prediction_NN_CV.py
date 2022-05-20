import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, InputLayer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from joblib import load
import matplotlib.pyplot as plt



def MSE(preds,labels):
    return np.mean((preds-labels)**2.0)

def run():

    print(tf.config.list_physical_devices(device_type=None))
    physical_devices = tf.config.list_physical_devices('GPU')
    print(physical_devices)
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


    #create each of the 5 models to test as indpenedent objects
    model1 = Sequential()
    model1.add(InputLayer(input_shape=(X_train.shape[1],)))
    model1.add(Dense(32,activation='relu'))
    model1.add(Dense(1,activation='linear'))

    model2 = Sequential()
    model2.add(InputLayer(input_shape=(X_train.shape[1],)))
    model2.add(Dense(64,activation='relu'))
    model2.add(Dense(32,activation='relu'))
    model2.add(Dense(1,activation='linear'))

    model3 = Sequential()
    model3.add(InputLayer(input_shape=(X_train.shape[1],)))
    model3.add(Dense(128,activation='relu'))
    model3.add(Dense(64,activation='relu'))
    model3.add(Dense(32,activation='relu'))
    model3.add(Dense(1,activation='linear'))

    model4 = Sequential()
    model4.add(InputLayer(input_shape=(X_train.shape[1],)))
    model4.add(Dense(256,activation='relu'))
    model4.add(Dense(128,activation='relu'))
    model4.add(Dense(64,activation='relu'))
    model4.add(Dense(32,activation='relu'))
    model4.add(Dense(1,activation='linear'))

    model5 = Sequential()
    model5.add(InputLayer(input_shape=(X_train.shape[1],)))
    model5.add(Dense(256,activation='relu'))
    model5.add(Dropout(.15))
    model5.add(Dense(128,activation='relu'))
    model5.add(Dropout(.10))
    model5.add(Dense(64,activation='relu'))
    model5.add(Dense(32,activation='relu'))
    model5.add(Dense(1,activation='linear'))

    #train each model and write results
    cv_models = [model1, model2, model3, model4, model5]
    cv_model_histories = {}
    for i in range(0,len(cv_models)):
        print(f'----- Training Model {i} -----')
        model_i = cv_models[i]
        model_i.compile(optimizer='Adam',loss='MeanSquaredError',metrics=['MeanSquaredError','MeanAbsoluteError'])
        model_i_history = model_i.fit(X_train,y_train,epochs=15,batch_size=256,validation_data=(X_test,y_test),verbose=0)
        model_identifier_string = 'Model '+ str(i)
        cv_model_histories[model_identifier_string] = model_i_history
        history_i_frame = pd.DataFrame(model_i_history.history)
        nn_history_str = 'Trained_NN_Model_'+str(i)+'_History.csv'
        nn_model_history_path = '../../Data/Modeling Results/'
        history_i_frame.to_csv(nn_model_history_path+nn_history_str, index=False)
        nn_model_is_str = 'Trained_NN_Model_'+str(i)
        nn_save_location_path = '../../Models/Neural Networks/'
        model_i.save(nn_save_location_path+nn_model_is_str)
    print('All Models Successfully Trained!')

    #plot train
    fig = plt.figure(figsize=(12,7))
    for i in range(0,len(cv_models)):
        model_str = 'Model ' + str(i)
        history_i = cv_model_histories[model_str]
        acc_i = history_i.history['mean_squared_error']
        epochs = range(1,len(acc_i)+1)
        plt.plot(epochs,acc_i,label=model_str)
        
    plt.legend()
    plt.title(f'Training Performance of Varying Neural Network Models on Data')
    plt.ylabel('Mean Squared Error')
    plt.xlabel('Epoch')
    plt.show()

    #plot test
    fig = plt.figure(figsize=(12,7))
    for i in range(0,len(cv_models)):
        model_str = 'Model ' + str(i)
        history_i = cv_model_histories[model_str]
        val_acc_i = history_i.history['val_mean_squared_error']
        epochs = range(1,len(val_acc_i)+1)
        plt.plot(epochs,val_acc_i,label=model_str)
    plt.legend()
    plt.title(f'Test Performance of Varying Neural Network Models on Data')
    plt.ylabel('Mean Squared Error')
    plt.xlabel('Epoch')
    plt.show()
    
    fig = plt.figure(figsize=(12,7))
    for i in range(0,len(cv_models)):
        model_str = 'Model ' + str(i)
        history_i = cv_model_histories[model_str]
        val_acc_i = history_i.history['val_mean_squared_error']
        epochs = range(1,len(val_acc_i)+1)
        plt.plot(epochs,val_acc_i,label=model_str)
    plt.legend()
    plt.title(f'Test Performance of Varying Neural Network Models on Data')
    plt.ylabel('Mean Squared Error')
    plt.xlabel('Epoch')
    plt.show()

if __name__ == "__main__":
    run()