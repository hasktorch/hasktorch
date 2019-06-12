import torch.nn as nn
import torch

def read_params(filename):

    with open(filename, "r+") as f:
        s = f.read()
        tensors = s.split("***")

        params = tensors[0]
        values = tensors[1]
        pred   = tensors[2]
        loss   = tensors[3]
        grads  = tensors[4]

    wts = []
    for param in params.split("\n"):
        if param != "":
            wts.append(eval(param))

    vals = []
    for value in values.split("\n"):
        if value != "":
            vals.append(eval(value))

    predicted = eval(pred.rstrip("\n").lstrip("\n"))

    loss = eval(loss.lstrip("\n").rstrip("\n"))

    gradients = eval(grads.rstrip("\n").lstrip("\n"))

    return wts, vals, predicted, loss, gradients


def print_details(rnncell):

    print(rnncell.weight_ih)
    print(rnncell.bias_ih)
    print(rnncell.weight_hh)
    print(rnncell.bias_hh)
    print("***")


def init_rnn(rnn, params):

    rnn.weight_ih = nn.Parameter(torch.DoubleTensor(params[0]))
    rnn.bias_ih   = nn.Parameter(torch.DoubleTensor(params[1]))
    rnn.weight_hh = nn.Parameter(torch.DoubleTensor(params[2]))
    rnn.bias_hh   = nn.Parameter(torch.DoubleTensor(params[3]))
    return rnn


# reading in the Haskell-calculated values
wts, vals, pred, loss_, grads = read_params("out")

# initializing the RNN
rnn = nn.RNNCell(2, 2, nonlinearity='tanh')

inp = torch.DoubleTensor(vals[0])
init_hidden = torch.DoubleTensor(vals[1])
out = torch.DoubleTensor(vals[2])

# print(vals[0])
# print(inp)
# print("-------------")
# print(vals[1])
# print(init_hidden)
# print("-------------")
# print(vals[2])
# print(out)
# print("-------------")


# initialize the RNN with the haskell-initialized parameters
rnn = init_rnn(rnn, wts)


# for w in wts:
#     print(w)
# print("--------")

# print the initial state of the cell
#print_details(rnn)


hidden_t1 = rnn(inp, init_hidden)

hidden_ = torch.matmul(rnn.weight_ih, inp.t())

print(rnn.bias_ih.t())
print(hidden_)
print(hidden_t1)
"""
# arbitrary loss function
loss = nn.MSELoss()

# loss for our first timestep
t1_loss = loss(hidden_t1, out)

print(loss_)
print(t1_loss)
"""
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
