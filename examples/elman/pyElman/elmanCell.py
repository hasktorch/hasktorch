import torch.nn as nn
import torch


def print_details(rnncell):

    print(rnncell.weight_ih)
    print(rnncell.bias_ih)
    print(rnncell.weight_hh)
    print(rnncell.bias_hh)
    print("***")


# initializing the RNN
rnn = nn.RNNCell(2, 2)

inp = torch.randn(1, 2, requires_grad=True)
init_hidden = torch.randn(1, 2, requires_grad=True)
out = torch.randn(1, 2, requires_grad=True)

print(inp)
print(init_hidden)
print(out)

# print the initial state of the cell
print_details(rnn)

hidden_t1 = rnn(inp, init_hidden)

# arbitrary loss function
loss = nn.MSELoss()

# loss for our first timestep
t1_loss = loss(hidden_t1, out)

print(t1_loss)
"""
# backprop the loss using pytorch's handy auto-backprop mechanism
rnn.zero_grad()
t1_loss.backward()

# and for the update, we'll use simple gradient descent
# i.e: updatedWeights = weights - learning_rate * gradient
# and a ridiculously large learning rate so that the change is noticeable
learningRate = 100

rnn.weight_ih = nn.Parameter(rnn.weight_ih - (learningRate * rnn.weight_ih.grad))
rnn.weight_hh = nn.Parameter(rnn.weight_hh - (learningRate * rnn.weight_hh.grad))
rnn.bias_ih = nn.Parameter(rnn.bias_ih - (learningRate * rnn.bias_ih.grad))
rnn.bias_hh = nn.Parameter(rnn.bias_hh - (learningRate * rnn.bias_hh.grad))

# print the updated weights of the cell
print_details(rnn)
"""
