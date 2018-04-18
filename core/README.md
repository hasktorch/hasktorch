# hasktorch-core

`hasktorch-core` includes reexports of the high-level interface to basic tensor creation and
math operations and manages allocation/deallocation via foreign pointers.

## Package Structure

From hasktorch-core.cabal:

    exposed-modules:
      -- support modules
        Torch.Core.Exceptions
      , Torch.Core.Random
      , Torch.Core.LogAdd

      -- CPU modules
      , Torch
      , Torch.Storage
      , Torch.Dynamic

      -- CPU modules
      , Torch.Cuda
      , Torch.Cuda.Storage
      , Torch.Cuda.Dynamic


<!--
## Basic Implementation Concepts: Foreign Pointer Abstractions

Raw tensors used in `raw/` are used as `Ptr a` where the type of the target of
the pointer `a` corresponds to the Haskell representation of C TH Structures.
For example `Ptr CTHDoubleTensor` is a pointer to `CTHDoubleTensor`, the Double
instantiation of the `THTensor` struct.

The `core` modules provide a higher level representations that hides the raw C
pointer and structs. Currently there are both static and dynamically typed
interfaces, where statically typed tensors represent tensor dimensions at the
type level (example from `Double.hs`):

```
newtype TensorDoubleStatic (d :: [Nat]) = TDS {
  tdsTensor :: ForeignPtr CTHDoubleTensor
  } deriving (Show)

type TDS = TensorDoubleStatic
```

while dynamically typed tensors represent dimensions at the value level (from
`Types.hs`):

```
data TensorDouble = TensorDouble {
  tdTensor :: !(ForeignPtr CTHDoubleTensor),
  tdDim :: !(TensorDim Word)
  } deriving (Eq, Show)
```

Both implementations are a work-in-progress, however the statically typed
implementation is recommended as a default.

In order to preserve pure functional semantics, accessors such as `tdsTensor`
and `tdTensor` are only intended to be used by the API implementation and not by
end users at the application level.

When constructing these tensor types, the internal tensor is initialized with a
pointer to the TH library's deallocator function `TH[Type]Tensor_free` as the
finalizer used by `newForeignPtr`. The `raw/` modules provide C pointers
functions to functions as via functions having a `p_` prefix. So for example
`p_THDoubleTensor_free` is used as the finalizer for Double tensors.

-->

## Basic Implementation Concepts: Preserving Immutability in non-IO FFI

In the C API and in `raw`, operations mutate memory storage pointed to by raw C
pointers and therefore take place in the IO. For example
`c_THDoubleTensor_sigmoid` is applies a sigmoid transformation to values in a
tensor:

```
-- |c_THDoubleTensor_sigmoid : r_ t -> void
foreign import ccall "THTensorMath.h THDoubleTensor_sigmoid"
  c_THDoubleTensor_sigmoid :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

```

Hasktorch experimentally adopts and extends the pytorch naming convention with the
following:
- As in pytorch: a function suffixed with `_` will mutate the first tensor, inplace,
  _and return that tensor as its output_. This means that `sigmoid_ :: t -> IO t` will
  mutate its input.

- A function (usually found in typeclasses) that is prefixed with `_` is a direct
  call into C and up to two of the first arguments may be mutated. This means that
  `_sigmoid` is simply the direct call to C.

- As in pytorch: a function not prefixed or suffixed with `_` is a pure function and
  calling it will construct a new tensor as an output. In essence `sigmoid :: t -> IO t`
  is what you would expect in a happy haskell file.

Unsafe operations are used to provide a functional API if and only if the
operation is functional module allocation / deallocation. Just as native types
do not treat memory allocation as IO, operations that do not perform
mutation but do perform allocation are presented as pure functions. In many of
these cases, if the return value is a tensor, it is allocated within the
function and returned with populated values.

While pure functions are desireable in haskell-land, pragmatic deep learning research
often promotes the mutation of tensors for performance reasons. Hasktorch currently
doesn't handle the correct usage of this (with `ST`), and future contributions here would
be welcome.
