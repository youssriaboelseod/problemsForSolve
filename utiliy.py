import numpy as np
import pandas as pd
def one_hot_encode(arr, n_labels):
        '''
            LSTM expects an input that is one-hot encoded_id meaning that each character 
            is converted into an integer (via our created dictionary) 
            and then converted into a column vector where only it's corresponding integer index will have the value of 1 and the rest of the            vector will be filled with 0's. Since we're one-hot encoding the data
        '''
        # Initialize the the encoded_id array
        one_hot = np.zeros((np.multiply(*arr.shape), n_labels), dtype=np.float32)

        # Fill the appropriate elements with ones
        one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.

        # Finally reshape it to get back to the original array
        one_hot = one_hot.reshape((*arr.shape, n_labels))

        return one_hot
    # check that the function works as expected
def get_batches(arr, batch_size, seq_length):
    '''Create a generator that returns batches of size
    batch_size x seq_length from arr.
    
    Arguments
    ---------
    arr: Array you want to make batches from
    batch_size: Batch size, the number of sequences per batch
    seq_length: Number of encoded_id chars in a sequence
    '''
    batch_size_total = batch_size * seq_length
    # total number of batches we can make
    n_batches = len(arr)//batch_size_total
    
    # Keep only enough characters to make full batches
    arr = arr[:n_batches * batch_size_total]
    # Reshape into batch_size rows
    arr = arr.reshape((batch_size, -1))
    
    # iterate through the array, one sequence at a time
    for n in range(0, arr.shape[1], seq_length):
        # The features
        x = arr[:, n:n+seq_length]
        # The targets, shifted by one
        y = np.zeros_like(x)
        try:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n+seq_length]
        except IndexError:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]
        yield x, y
    