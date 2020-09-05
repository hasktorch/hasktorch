# Optimizers Example

Implementations of optimizers run on simple test functions. This implementation is for untyped tensors.

- `Main.hs` - main training loop execution
- `TestFunction.hs` - differentiable implementations of [test functions](https://en.wikipedia.org/wiki/Test_functions_for_optimization)

Optimizers are represented by libtorch's c++ codes.

The `step` function can be called for any optimizer, and is defined as:

```
class Optimizer option where
  type ToOptStateRef option :: *
  initOptimizer :: Parameterized d => option -> d -> IO (OptimizerState option (ToOptStateRef option) d)
  step :: Parameterized d => OptimizerState option (ToOptStateRef option) d -> (d -> IO Tensor) -> IO Tensor
  getParams :: Parameterized d => OptimizerState option (ToOptStateRef option) d -> IO d
```

The `step` function returns loss, and it updates parameters of OptimizerState.

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
stack run optimizers-cpp
```
