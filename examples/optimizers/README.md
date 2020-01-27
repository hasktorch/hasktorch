# Optimizers Example

Implementations of optimizers run on simple test functions. This implementation is for untyped tensors.

- `Main.hs` - main training loop execution
- `TestFunction.hs` - differentiable implementations of [test functions](https://en.wikipedia.org/wiki/Test_functions_for_optimization)

Optimizers are represented by types encapsulating the state of the optimizer (momentum, moments, etc. or a dummy constructor in the case of a stateless optimizer such as gradient descent).

The `step` function can be called for any optimizer, and is defined as:

```
class Optimizer o where
    step :: 
        LearningRate -- ^ learning rate (Tensor alias)
        -> [Gradient] -- ^ gradients (Tensor alias)
        -> [Tensor] -- ^ model parameters
        -> o -- ^ optimizer state
        -> ([Tensor], o) -- ^ new model parameters and optimizer state

class Optimizer o where
    step :: LearningRate -> [Gradient] -> [Tensor] -> o -> ([Tensor], o)
```

The `step` function is responsible for computing gradients, returning a tuple containing the new set of parameters and an updated version of the state of the optimizer.

## Test Functions and OptimizersTest Functions

3 Test functions are currently implemented - convex quadratic, rosenbrock, and ackley function.

Optimization over convex quadratic and rosenbrock should converge, while optimization over the ackley function does not.

# Running the Example

Setup environment variables (run this from the top-level hasktorch project 
directory where the `setenv` file is):

```
source setenv
```

Building and running:

```
stack run optimizers
```
