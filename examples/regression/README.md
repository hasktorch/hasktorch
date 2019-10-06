# Simple Linear Regression Example

Implements a linear regression as a single linear layer. Input samples are 
generated from sampling a normal random distribution and calculating a
ground truth dataset.

Stochastic gradient descent is applied to infer the ground truth weight and
bias parameters.

# Running the Example

Setup environment variables (run this from the top-level hasktorch project 
directory where the `setenv` file is):

```
source setenv
```

Building and running:

```
stack run regression
```
