---
title: Randomness
---

```haskell top hide
import Torch
import Inliterate.Import (AskInliterate)
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

Note that since random initialization returns a different result each
time, unlike other tensor constructors, is monadic reflecting the
context of an underlying random number generator (RNG) changing state.

Hasktorch includes variations of random tensor initializers in which the
RNG object is threaded explicitly rather than implicitly. See the
`Torch.Random` module functions for details and variations. Samplers for
which the RNG is not explicit such as `randIO'` example above use the
`IO` suffix.
