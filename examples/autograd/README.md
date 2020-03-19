# Basic data flow through autograd

This example contains a very simple multi layer model meant to understand how autograd is used in Hasktorch to calculate gradients. The weights are taken to be equal to 1, and no activation has been applied to allow the user to conveniently verify the gradients for themselves. In the absence of the activation being applied and when the weights are equal to 1, the gradients for the respective weights will be equal to the value of input feature they are connected to. 

For Hasktorch, the autograd function takes the loss (or the function to find the derivative of, in our case the unactivated output) along with a list of Independent Tensors with respect to which the derivative is to found. But for the normal operation, the tensors should be dependent tensors, as would be clear in the code. 

# Running the Example

Setup environment variables (run this from the top-level hasktorch project 
directory where the `setenv` file is):

```
source setenv
```

Building and running:

```
stack run autograd
``` 
