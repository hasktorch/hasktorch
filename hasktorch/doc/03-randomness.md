---
title: Randomness
---

```haskell top
import Control.Monad.State
import Torch
import Torch.Internal.Managed.Type.Context (manual_seed_L)
```

```haskell top hide
import Inliterate.Import (AskInliterate)
instance AskInliterate Bool
instance AskInliterate Tensor
```

# Randomness

Create a randomly initialized matrix:

```haskell do
w <- randIO' [2, 2]
```

```haskell eval
w
```

Note that random initialization without specifying the random number
generator (RNG) state as above is necessarily impure, because we
expect a different result each time `randIO'` is called with the same
arguments.

```haskell do
w' <- randIO' [2, 2]
```

```haskell eval
w'
```

To make a computation deterministic, the RNG can be explicitly seeded:

```haskell do
manual_seed_L 42
w1 <- randIO' [2, 2]
```

```haskell do
manual_seed_L 42
w2 <- randIO' [2, 2]
```

```haskell eval
w1 == w2
```

## Pure vs. Impure

Hasktorch also includes pure variants of the random initialization
functions that accept the RNG state as an argument and return the
updated RNG state with the result.  For example, the `rand'` function
in `Torch.Random`:

```haskell
rand' :: [Int] -> Generator -> (Tensor, Generator)
```

where `Generator` is the type representing the RNG state.

By convention, random initialization functions with the `IO` suffix
thread the RNG state implicitly through the `IO` context, while the
analogous functions without the `IO` suffix expect a `Generator`
argument and return a new generator along with the result.

To use the "pure" style, we first need to initialize a `Generator`,
and then explicitly thread it through our computation:

```haskell do
rng0 <- mkGenerator (Device CPU 0) 31415
```

```haskell do
let (x1, rng1) = rand' [2, 2] rng0
    (x2, rng2) = rand' [2, 2] rng1
```

```haskell eval
(x1, x2)
```

The benefit of this approach is the clear separation between
deterministic and nondeterministic parts of the computation, in this
example manifested in the explicit threading of the `Generator` value.

Note that we can use the `State` monad to make this threading
implicit while still making use of the type system to enforce the
boundary between deterministic and nondeterministic code:

```haskell do
let randomPair = do x1 <- state $ rand' [2, 2]
                    x2 <- state $ rand' [2, 2]
                    pure (x1, x2)
```

We can then pass in a `Generator` to get a pair of random tensors like this:

```haskell eval
runState randomPair rng0
```
