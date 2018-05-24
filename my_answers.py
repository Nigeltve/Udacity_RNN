import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import SGD
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    
    X = [series[i:i+window_size] for i in range(0, len(series) - window_size)]
    
    y = [series[window_size + i] for i in range(0, len(series) - window_size)]
    
    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential()
    
    model.add(LSTM(5, input_shape = (window_size,1)))
    model.add(Dense(1))
                   
    return model


### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    import string
    chars = sorted(list(set(text)))
    punctuation = ['!', ',', '.', ':', ';', '?']
    
    letters = list(string.ascii_letters) #keeps  letters upper or lower
    white_space = [' ']
    all_letter = white_space + letters + punctuation # this is all the leters that need to be kept
    
    perged_chars = []
    for x in chars:
        # makes the purge list
        if x in all_letter:
            pass
        else:
            perged_chars.append(x)
            
    for x in perged_chars:
        text = text.replace(x, ' ') # goes through the purge list and gets rid of them from text
    
    text = str.lower(text) # returns everything as lowercase with punction
    return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = [text[i:i+window_size] for i in range(0, len(text) - window_size, step_size)] # same as step one but it goes the step size
    outputs = [text[window_size + i] for i in range(0, len(text) - window_size, step_size)] # same as step one but it goes the step size

    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    model = Sequential()
    
    model.add(LSTM(200, input_shape = (window_size,num_chars)))
    model.add(Dense(num_chars))
    model.add(Activation('softmax'))
    
    return model
