---
title: Typed Tensors
---

# Typed Tensors

Typed tensors provide an alternative API for which tensor
characteristics are encoded in the type of the tensor. The Hasktorch
library is layered such that \[ TODO \]

Using typed tensors increases the expressiveness of program invariants
that can be automatically checked by GHC at the cost of needing to be
more explicit in writing code and also requiring working with
Haskell's type-level machinery. Type-level Haskell programming has
arisen through progressive compiler iteration leading to mechanisms
that have a higher degree of complexity compared to value-level
Haskell code or other languages such as Idris which were designed with
type-level computations from inception. In spite of these compromises,
Haskell offers enough capability to express powerful type-level
representations of models that is matched by few other languages used
for production applications.

Things we can do in typed Hasktorch:

- specify, check, and infer tensor shapes at compile time

- specify, check, and infer tensor data types at compile time

- specify, check, and infer tensor compute devices at compile time

We can encode all Hasktorch tensor shapes on the type level using
type-level lists and natural numbers:

```haskell
type EmptyShape = '[]
type OneDimensionalShape (a :: Nat) = '[a]
type TwoDimensionalShape (a :: Nat) (b :: Nat) = '[a, b]
...
```

Tensor data types and compute device types are lifted to the type
level using the `DataKinds` language extension:

```haskell
type BooleanCPUTensor (shape :: [Nat]) = Tensor '(CPU,  0) 'Bool  shape
type IntCUDATensor    (shape :: [Nat]) = Tensor '(CUDA, 1) 'Int64 shape
```

Devices are represented as tuples consisting of a `DeviceType` (here
`CPU` for the CPU and `CUDA` for a CUDA device, respectively) and a
device id (here the `Nat`s `0` and `1`, respectively).

It is a common misconception that specifying tensor properties at
compile time is only possible if all tensor properties are constants
and are statically known. If this were the case, then we could only
write functions over fully specified tensors, say,

```haskell
boring :: BooleanCPUTensor '[] -> BooleanCPUTensor '[]
boring = id
```

Fortunately, Haskell has the ability to reason about type variables.
This feature is called parametric polymorphism. Consider this simple
example of a function:

```haskell
tensorNoOp
  :: forall (shape :: [Nat]) (dtype :: DType) (device :: (DeviceType, Nat))
   . Tensor device dtype shape
  -> Tensor device dtype shape
tensorNoOp = id
```

Here, `shape`, `dtype`, and `device` are type variables that have been
constrained to be of kind shape, data type, and device, respectively.
The universal quantifier `forall` implies that this function is
well-defined for all inhabitants of the types that are compatible with
the type variables and their constraints.

The `tensorNoOp` function may seem trivial, and that is because its
type is very strongly constrained: Given any typed tensor, it must
return a tensor of the same shape and with the same data type and on
the same device. Besides by means of the identity, `id`, there are not
many ways in which this function can be implemented.

There is a connection between a type signature of a function and
mathematical proofs. We can say that the fact that a function exists
is witnessed by its implementation. The implementation, `tensorNoOp =
id`, is the proof of the theorem stated by `tensorNoOp`'s type
signature. And here we have a proof that we can run this function on
any device and for any data type and tensor shape.

This is the essence of typed Hasktorch. As soon as the compiler gives
its OK, we hold a proof that our program will run without shape or
CUDA errors.
