# Examples

## Dynamic Tensor Examples

These examples do not attempt to type-check tensor dimensions.

- gaussian_process - basic gaussian process implementation
- minimal-text-example - "hello" string test of `rnn` modules, uses `rnn` as dependency`
- optimizers - experimental implementation of optimizers - gradient descent, gradient descent w/ momentum, adam, applied to optimization test functions
- regression - linear regression
- rnn - prototype implementations of Elman, LSTM, and GRU layers
- serialization - test serialization / deserialization of model state
- xor_mlp - an XOR multilayer perceptron
- vae - variational autoencoder

## Typed Tensor Examples

Some examples demonstrate typed tensor functionality. 

- static-xor-mlp - an XOR multilayer perceptron
- static-cnn - a convolutional neural network
- static-transformer - transformer with attention implementation

## Running the XOR MLP Example

The following steps should run the xor mlp example, assumes hasktorch has only been cloned but dependencies have not been pulled yet.

Starting at the top-level directory of the project, go to the deps directory and run the `get-deps.sh` shell script to retrieve project dependencies:

```
pushd deps
# For CPU
./get-deps.sh
# For CUDA-9
# ./get-deps.sh -a cu90
# For CUDA-10
# ./get-deps.sh -a cu100
popd
```

Set shared library environment variables:

```
source setenv
```

Build all examples:

```
stack build examples
```

Run the xor example:

```
stack run xor_mlp
```
