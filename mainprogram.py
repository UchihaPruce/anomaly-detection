import matplotlib.pyplot as plt
import numpy as np
import time
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from sidekick import sidekick
import warnings

np.random.seed(1234)

# Hyper-parameters
sequence_length = 50   #the sequence at which it  must iterate 
random_data_dup = 10  # each sample randomly duplicated between 0 and 9 times, see dropin function
epochs = 1 #One Epoch is when an ENTIRE dataset is passed forward and backward through the neural network only ONCE. 
batch_size = 50 # one epoch is too big to feed to the computer at once we divide it in several smaller batches
path_to_dataset = 'testset.txt' # the data is passed


def dropin(X, y):
    """ The name suggests the inverse of dropout, i.e. adding more samples. See Data Augmentation section at
    http://simaaron.github.io/Estimating-rainfall-from-weather-radar-readings-using-recurrent-neural-networks/
    :param X: Each row is a training sequence
    :param y: Tne target we train and will later predict
    :return: new augmented X, y
    """
    print("X shape:", X.shape)
    print("y shape:",y.shape)
    X_hat = []
    y_hat = []
    for i in range(0, len(X)):
        for j in range(0, np.random.random_integers(0,20)):
            X_hat.append(X[i, :])
            y_hat.append(y[i])
    return np.asarray(X_hat), np.asarray(y_hat) #Changes the normal array to numpy array


def z_norm(result):
    """ here we get mean,standard deviation and returns the same """
    result_mean = result.mean() #.mean() is a predifined func which finds the mean
    result_std = result.std()   # .std() is a predefined func which finds the standard deviation
    result -= result_mean    # result = result- result mean
    result /= result_std   # result = result/result std
    return result, result_mean

def get_split_prep_data(train_start, train_end,
                          test_start, test_end):
    """we pass the the params which are train start, train end and test start, test end
        from the run model function it basically loads the dataset and passes the value
        to savitzky_golay function aka sidekick class"""
    data = np.loadtxt(path_to_dataset) #loads the testset.txt
    data = sidekick(data[:, 1], 11, 3) # smoothed version
    print("Length of Data", len(data)) #len(data ) finds the length of data 

    # train data
    print("Creating train data...") 

    result = [] #creates a result array ...all the results are stored in this array
    """The result is appended from the train data ...based on the sequence length"""
    for index in range(train_start, train_end - sequence_length):
        result.append(data[index: index + sequence_length])
    result = np.array(result)  # shape (samples, sequence_length)
    result, result_mean = z_norm(result) # calls upon normal function

    print("Mean of train data : ", result_mean)
    print("Train data shape  : ", result.shape)

    train = result[train_start:train_end, :]
    np.random.shuffle(train)  # shuffles in-place
    X_train = train[:, :-1]
    y_train = train[:, -1]
    X_train, y_train = dropin(X_train, y_train) # calls the dropin func

    # test data
    print("Creating test data...")

    result = []
    """Does the same as above but for the test set"""
    for index in range(test_start, test_end - sequence_length):
        result.append(data[index: index + sequence_length])
    result = np.array(result)  # shape (samples, sequence_length)
    result, result_mean = z_norm(result) # calls upon normal function

    print("Mean of test data : ", result_mean)
    print("Test data shape  : ", result.shape)

    X_test = result[:, :-1]
    y_test = result[:, -1]

    print("Shape X_train", np.shape(X_train))
    print("Shape X_test", np.shape(X_test))

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1)) #reshapes accordingly to the parameters for the TRAINING set
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1)) #reshapes accordingly to the parameters for the TESTING set

    return X_train, y_train, X_test, y_test


def build_model(): # one input, three layers and one output
    model = Sequential()
    layers = {'input': 1, 'hidden1': 64, 'hidden2': 256, 'hidden3': 100, 'output': 1}
    model.add(LSTM(
            input_length=sequence_length - 1,
            input_dim=layers['input'],
            output_dim=layers['hidden1'],
            return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
            layers['hidden2'],
            return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
            layers['hidden3'],
            return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(
            output_dim=layers['output']))
    model.add(Activation("linear"))

    start = time.time() #to give the execution time ...this part of the code is used 
    model.compile(loss="mse", optimizer="rmsprop") # the model is compiled with mean squared error (mse)
    print("Compilation Time : ", time.time() - start) # gives the compilation time for every model iteration
    return model


def run_network(model=None, data=None):
    global_start_time = time.time()
    epochs = 1 #One Epoch is when an ENTIRE dataset is passed forward and backward through the neural network only ONCE. 

    if data is None:
        print('Loading data... ')
        X_train, y_train, X_test, y_test = get_split_prep_data(
                0, 3000, 3000, 12000)
        """passes the start of tbe train set,end of the training set,test start and test end """
    else:
        X_train, y_train, X_test, y_test = data

    print('\nData Loaded. Compiling...\n')

    if model is None:
        model = build_model() #build model function is called

    try:
        print("Training")
        model.fit(
                X_train, y_train,
                batch_size=512, nb_epoch=epochs, validation_split=0.05) # fits the model based on the batch size and epoch, the x of train and y of train 
        print("Predicting")
        predicted = model.predict(X_test)
        """fit(self, x, y, batch_size=32, nb_epoch=10, verbose=1, callbacks=[], validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None)
           batch_size: integer. Number of samples per gradient update.
           nb_epoch: integer, the number of times to iterate over the training data arrays.
           verbose: 0, 1, or 2. Verbosity mode. 0 = silent, 1 = verbose, 2 = one log line per epoch."""
        print("shape of predicted", np.shape(predicted), "size", predicted.size)
        print("Reshaping predicted")
        predicted = np.reshape(predicted, (predicted.size,))
    except KeyboardInterrupt:
        print("prediction exception")
        print('Training duration (s) : ', time.time() - global_start_time)
        return model, y_test, 0

    try:
        plt.figure(1) #plots the actual signal with anomalies , predicted signal aka smoothed signal and the squared error which is the one which denotes the error or the anomaly
        plt.subplot(311) 
        plt.title("Actual Signal with Anomalies")
        plt.plot(y_test[:len(y_test)], 'b')
        plt.subplot(312)
        plt.title("Predicted Signal")
        plt.plot(predicted[:len(y_test)], 'g')
        plt.subplot(313)
        plt.title("Squared Error")
        mse = ((y_test - predicted) ** 2)
        plt.plot(mse, 'r')
        plt.show()
    except Exception as e:
        print("plotting exception")
        print(str(e))
    print('Training duration (s) : ', time.time() - global_start_time)

    return model, y_test, predicted

run_network()

