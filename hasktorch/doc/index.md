---
title: Hasktorch
---

Welcome to Hasktorch!

Hasktorch is a Haskell library for scientific computing and
differentiable programming. It is built on [libtorch][] — the same
C++ core that powers PyTorch — so it executes on the same kernels,
the same automatic differentiation tape, and the same GPUs, while the
surface you program against is Haskell: pure [functional
programming][olah-nn-types-fp], an expressive type system, and
compile-time checking of the things deep-learning code most often
gets wrong at runtime.

The library is layered, and this tutorial follows the layers. Each
one is optional: you can be productive with the untyped API alone,
and every further part buys more static guarantees for more type-level
machinery.

## Part I — The untyped API

The dynamically-shaped layer, closest to PyTorch: if you have used
`torch`, this is the same mental model with Haskell syntax. Shapes
live in values, mistakes surface at runtime.

1. [Getting Started](01-getting-started.html) — installation and first steps
2. [Tensors](02-tensors.html) — creation, values, shapes, devices
3. [Randomness](03-randomness.html) — random tensors and effects
4. [Automatic Differentiation](04-automatic-differentiation.html) — independent tensors and gradients
5. [Differentiable Programs](05-differentiable-programs.html) — models as records of parameters
6. [Linear Regression](06-linear-regression.html) — a first complete training loop

## Part II — The typed API

Shapes, dtypes, and devices move into the types, so a mismatched
matrix multiplication or a misplaced batch dimension is a compile
error. The guarantee compounds: if the program compiles, the forward
pass, loss, gradients, and optimizer steps all agree.

7. [Typed Tensors](07-typed-tensors.html) — shape-indexed tensors and typed autograd
8. [Named Tensors and Lenses](08-named-tensors.html) — dimensions with *meaning*: records as axes, fields as lenses
9. [Indexing and Slicing](09-indexing.html) — PyTorch's `[1, :, 1:3:2]` in both APIs, bounds-checked in the typed one
10. [Lenses](10-lenses.html) — slicing lenses and whole-model traversals (e.g. convert a model to `Half` in one line), in both APIs

## Part III — Semantics: writing the math, running the tensors

The layer this tutorial builds towards: element-level formulas and
architectural structure expressed directly, with the tensor execution
derived from them.

11. [Graded and Staged Tensor Programs](11-graded-and-staged.html) — why `Tensor` is not a `Monad` and what it is instead; element formulas that run both as their own specification and as vectorized ATen calls
12. [Moving Dimensions](12-dimensions.html) — `dimUp`/`dimDown`: dimensions migrate between tensor axes and Haskell structure, up to tree-shaped data
13. [Attention, Equation by Equation](13-attention.html) — a trainable decoder block in sixty lines, one definition per equation
14. [Networks as Arrows](14-arrows.html) — models compose with `>>>`, `&&&`, and `proc` notation; ResNet skips and layer stacks as folds

## Part IV — The runtime underneath

15. [TorchScript and the JIT](15-torchscript-and-jit.html) — tracing models from Haskell, and measured reality about fusion

Chapters 1–6 assume no Haskell type-level programming at all.
Chapters 7–10 use type-level naturals and records. Chapters 11–15 are
where the research-flavoured ideas live — each grounded in code that
is compiled, executed, and tested in CI, with the rendered output on
these very pages.

Looking for a reference? See the [API docs][api-docs] hosted on
[hasktorch.org][hasktorch-org].

[api-docs]: http://hasktorch.org/docs.html
[getting-started]: 01-getting-started.html
[hasktorch-org]: http://hasktorch.org/
[libtorch]: https://pytorch.org/cppdocs/installing.html
[olah-nn-types-fp]: https://colah.github.io/posts/2015-09-NN-Types-FP/
