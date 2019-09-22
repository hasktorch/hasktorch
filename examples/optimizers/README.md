# Optimizers Example Example Example Example

Implementations of optimizers run on simple test functions. This implementation is for untyped tensors.

- `Main.hs` - main training loop execution
- `Optimizers.hs` - optimizer implementations and typeclass definition (candidate implementation for a library optimizers implementation)
- `TestFunction.hs` - differentiable implementations of [test functions](https://en.wikipedia.org/wiki/Test_functions_for_optimization<Paste>)

Optimizers are represented by types encapsulating the state of the optimizer (momentum, moments, etc. or a dummy constructor in the case of a stateless optimizer such as gradient descent).

The `step` function can be called for any optimizer, and is defined as:

```
class Optimizer a where
    step :: 
        Tensor -- ^ learning rate
        -> a -- ^ optimizer state
        -> [Parameter] -- ^ model parameters
        -> [Tensor] -- ^ gradient
        -> ([Tensor], a) -- ^ new model parameters and optimizer state
```

The `step` function is responsible for computing gradients, returning a tuple containing the new set of parameters and an updated version of the state of the optimizer.
