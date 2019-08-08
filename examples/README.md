# Examples

Most examples use untyped dimensions for now with the exception of `cnn`. More examples to be added prior to the 0.2 release.

- cnn - a convolutional neural network implemented with typed dimensions
- elman - elman RNN
- gaussian_process - basic gaussian process implementation
- regression - simple linear regression
- xor_mlp - a toy XOR multilayer perceptron
- vae - variational autoencoder

## Running the XOR MLP Example

The following steps should run the xor mlp example, assuming hasktorch has only been cloned but dependencies have not been pulled yet.

Start at the top-level directory of the project.
```
```
# Download libtorch-binary and other shared library dependencies
pushd deps
# For CPU
./get-deps.sh
# For CUDA-9
# ./get-deps.sh -a cu90
# For CUDA-10
# ./get-deps.sh -a cu100
popd

# Set shared library environment variables
source setenv

stack build examples

stack exec xor_mlp
```
