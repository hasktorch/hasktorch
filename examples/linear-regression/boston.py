from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn as nn
from sklearn.datasets import load_boston
import numpy as np
import torch.optim as optim

lr = 0.01        # Learning rate
epoch_num = 10000

boston = load_boston()

features = torch.from_numpy(np.array(boston.data)).float() 
features = Variable(features)                   

labels = torch.from_numpy(np.array(boston.target)).float() 
labels = Variable(labels)                          
labels = np.reshape(labels, (506, 1))
linear_regression = nn.Linear(13, 1)

criterion = nn.MSELoss()
optimizer = optim.Adam(linear_regression.parameters(), lr=lr)

for ep in range(epoch_num):
   
    linear_regression.zero_grad()

    # Forward pass
    output = linear_regression(features)
    loss = criterion(output, labels)       
    if not ep%500:
        print('Epoch: {} - loss: {}'.format(ep, loss.data[0]))

    # Backward pass and updates
    loss.backward()                         
    optimizer.step()                        
