
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

#from local librarary
from utiliy import get_batches
#haracter_Level_RNN_Exercise.html
print ("the login the file train successfuly")
class TexTrainer():
    def train(net, data, epochs, batch_size, seq_length, lr, clip=5, val_frac=0.1, print_every=10):
        ''' Training a network 
        
            Arguments
            ---------
            
            net: CharRNN network
            data: text data to train the network
            epochs: Number of epochs to train
            batch_size: Number of mini-sequences per mini-batch, aka batch size
            seq_length: Number of character steps per mini-batch
            lr: learning rate
            clip: gradient clipping
            val_frac: Fraction of data to hold out for validation
            print_every: Number of steps for printing training and validation loss
        
        '''
        print ("data",data[:100])
        print ("batch_size",batch_size)
        print ("seq_length",seq_length)
        print ("epochs",epochs)

        net.train()
        
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        # create training and validation data
        val_idx = int(len(data)*(1-val_frac))
        data, val_data = data[:val_idx], data[val_idx:]
        train_on_gpu = torch.cuda.is_available()
        if(train_on_gpu):
            net.cuda()
        
        counter = 0
        n_chars = len(net.chars)
        print ("get batches",get_batches(data, batch_size, seq_length))
        for e in range(epochs):
            # initialize hidden state
            h = net.init_hidden(batch_size)
            print ("init_hiding for batch",h)
            for x, y in get_batches(data, batch_size, seq_length):
                print ("x",x,"y",y)
                counter += 1
                
                # One-hot encode our data and make them Torch tensors
                x = one_hot_encode(x, n_chars)
                inputs, targets = torch.from_numpy(x), torch.from_numpy(y)
                
                if(train_on_gpu):
                    inputs, targets = inputs.cuda(), targets.cuda()

                # Creating new variables for the hidden state, otherwise
                # we'd backprop through the entire training history
                h = tuple([each.data for each in h])

                # zero accumulated gradients
                net.zero_grad()
                
                # get the output from the model
                output, h = net(inputs, h)
                
                # calculate the loss and perform backprop
                loss = criterion(output, targets.view(batch_size*seq_length))
                loss.backward()
                # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                nn.utils.clip_grad_norm_(net.parameters(), clip)
                opt.step()
                
                # loss stats
                if counter % print_every == 0:
                    # Get validation loss
                    val_h = net.init_hidden(batch_size)
                    val_losses = []
                    net.eval()
                    for x, y in get_batches(val_data, batch_size, seq_length):
                        # One-hot encode our data and make them Torch tensors
                        x = one_hot_encode(x, n_chars)
                        x, y = torch.from_numpy(x), torch.from_numpy(y)
                        
                        # Creating new variables for the hidden state, otherwise
                        # we'd backprop through the entire training history
                        val_h = tuple([each.data for each in val_h])
                        
                        inputs, targets = x, y
                        if(train_on_gpu):
                            inputs, targets = inputs.cuda(), targets.cuda()

                        output, val_h = net(inputs, val_h)
                        val_loss = criterion(output, targets.view(batch_size*seq_length))
                    
                        val_losses.append(val_loss.item())
                    
                    net.train() # reset to train mode after iterationg through validation data
                    
                    print("Epoch: {}/{}...".format(e+1, epochs),
                        "Step: {}...".format(counter),
                        "Loss: {:.4f}...".format(loss.item()),
                        "Val Loss: {:.4f}".format(np.mean(val_losses)))
