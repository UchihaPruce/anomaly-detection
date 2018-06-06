# anomaly-detector
a anomaly detector using lstm
One way is as follows: Use LSTMs to build a prediction model, 
i.e. given current and past values, predict next few steps in the time-series. 
Then, error in prediction gives an indication of anomaly (LSTM-AD [1]). 
For example, if prediction error is high, then it indicates anomaly.
Note: This assumes that the normal time-series is predictable to some extent.

Another way is to directly use LSTM as a classifier with two classes: normal and anomalous.

Prediction model based approach is better when anomalous instances are not easily available whereas a classifier based approach 
is more suitable when there are sufficient labeled instances of both normal and anomalous instances.

READ THIS PAPER TO KNOW MORE ABOUT LSTM
https://www.elen.ucl.ac.be/Proceedings/esann/esannpdf/es2015-56.pdf


sequence_length = 50   #the sequence at which it  must iterate 
random_data_dup = 10  # each sample randomly duplicated between 0 and 9 times, see dropin function
epochs = 1 #One Epoch is when an ENTIRE dataset is passed forward and backward through the neural network only ONCE. 
batch_size = 50 # one epoch is too big to feed to the computer at once we divide it in several smaller batches 


FUNCTIONS EXPLAINATION
1.) DROPIN FUCTION 
The name suggests the inverse of dropout, i.e. adding more samples. See Data Augmentation section at
    http://simaaron.github.io/Estimating-rainfall-from-weather-radar-readings-using-recurrent-neural-networks/
    :param X: Each row is a training sequence
    :param y: Tne target we train and will later predict
    :return: new augmented X, y
Two arrays are declared...the content is appended and then they are converted to numpy array with the help of the function np.asarray(parameters)
this is what they do in the dropin function


2.) z_norm(PARAMETER)
Here we get mean,standard deviation and returns the same
the mean is found using the .mean() function 
the standard deviation is found using the .std() function
here we dont find have the upper bound and the lower bound like in other models (MAD OR DBSCAN)
we just find the standard deviation and compare each value ...to find anomaly 


3.) get_split_prep_data(parameters)
The parameters are train_start, train_end,test_start, test_end
we pass it from the run network function 
we pass the the parameters which are train start, train end and test start, test end
from the run model function it basically loads the dataset and passes the value
to savitzky_golay function aka sidekick class
load the dataset which is the testset.txt


the sidekick class is called ...
the sidekick class is nothing but Savitzky-Golay algorithm ...research more on it to understand that class
Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
The Savitzky-Golay filter removes high frequency noise from data.
It has the advantage of preserving the original shape and
features of the signal better than other types of filtering
approaches, such as moving averages techniques.



Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures.


np.int() - converts normal int to numpy int (https://docs.scipy.org/doc/numpy-1.13.0/user/basics.types.html)
np.mat() - converts normal arrat to numpy matrix (https://docs.scipy.org/doc/numpy/reference/generated/numpy.mat.html)
np.abs() - gives the absolute value (https://docs.scipy.org/doc/numpy/reference/generated/numpy.absolute.html)
np.convolve(a, v, mode='full') - Returns the discrete, linear convolution of two one-dimensional sequences. (https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.convolve.html)
np.concatenate((a1, a2, ...), axis=0, out=None) - Join a sequence of arrays along an existing axis.  (https://docs.scipy.org/doc/numpy/reference/generated/numpy.concatenate.html)
PLEASE READ NUMPY DOCUMENTAION FOR EACH FUNCTION 

THE CONDITIONS OF WINDOW SIZE
window_size and order have to be of type int
window_size size must be a positive odd number
window_size should have a large polynomials order	   
PRECOMPUTE COEFFICIENTS
PAD THE SIGNAL AT THE EXTREME WIDTH
VALUES TAKEN FROM THE SIGNAL ITSELF



TO understand more about The Savitzky-Golay algorithm please do the math and the formulae


TRAIN DATA IS CREATED 
the results array is created 
IN this function the mean of the train data, the shape of the train data is found ...compare the output and the program coding to understand
the same way the test data, the shape of the train data is found

Xtrain is shaped and xtest is shaped 
the reshape function is called upon
the get_split_prep_data() function returns X_train, y_train, X_test, y_test
if i missed anything in this function then please go google it to get its definition


4.)build_model()
Please do your research on Long short term memory(LSTM) and RNN(recurral neural network) before you try to understand this function 
IN this function one input one output layer and three hidden layer is used 
the model is sequential() ..
You can create a Sequential model by passing a list of layer instances to the constructor:
AND that is what we do in the function 
Read this documentation to understand Sequential model https://keras.io/getting-started/sequential-model-guide/
THEN with the help of the .add function we add the layers 

then the remaining part of the code is for the execution time and the compilation time
we then return the model

5.)run_network(model=None, data=None):
i told you what a epoch is before ..if you read the whole thing you would have known 
so first we call upon the get split prep data function
passes the start of the train set,end of the training set,test start and test end
the model value is none which is a temporary parameter, is given so the build model is called upon ...read the 4th function to understand about it
model.fit() - fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None)
read this documentation https://keras.io/models/model/
model.predict() - predict(self, x, batch_size=None, verbose=0, steps=None)
read this documentation https://keras.io/models/model/ make sure to scroll down
then we find the predicted shape 
an reshape the predicted 
then we plot the graphs


PLOTTING THE GRAPHS 
READ THE MATPLOTLIB DOCUMENTATION BEFORE READING THIS PART 
https://matplotlib.org/api/pyplot_api.html
read more on that if that does not suffice 
so our objective is to plot the actual signal with anomalies , predicted signal aka smoothed signals(savitzky_golay),the squared error which is the one which denotes the error or the anomaly
plt.title() - plots the title of the graph 
plt.plot(x axis, yaxis ) _ Its upto you to denote the x axis and y axis 
plt.subplot - https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplot.html read this documentation
adjust that according to your needs
then finally return model, y_test, predicted
This is IT ....do a lot of research try changing the program according to your needs and change it to multivariate
see the program output to see how the program is executed(the order at which it is executed, then track the program to make its changes)
or wait 

INSTEAD OF USING MATPLOTLIB TRY USING BOKEHJS for plotting these graphs ...cause we can zoom through them 
and try learning to use jupyter notebook..cause its very handy if you know how to use bokehjs
the program function call is 
1st)run_network():
    2nd)get_split_prep_data():
	    3rd) sidekick class
		4TH)z_norm()
		5TH)dropin()
		6th)z_norm()
	7th)build_model()
and thats the order at which it is called.... 
and thats the end of it
