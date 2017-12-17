# torch-core

Higher-level interface to basic tensor operations, including memory/resource
management via foreign pointers.

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
  tdsTensor :: !(ForeignPtr CTHDoubleTensor)
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

In order to preserve pure functional semantics, accessors such as `tdsTensor`
and `tdTensor` are only intended to be used by the API implementation and not by
end users at the application level.

When constructing these tensor types, the internal tensor is initialized with a
pointer to the TH library's deallocator function `TH[Type]Tensor_free` as the
finalizer used by `newForeignPtr`. The `raw/` modules provide C pointers
functions to functions as via functions having a `p_` prefix. So for example
`p_THDoubleTensor_free` is used as the finalizer for Double tensors.

## Basic Implementation Concepts: Preserving Immutability in non-IO FFI

In the C API and in `raw`, operations mutate memory storage pointed to by raw C
pointers and therefore take place in the IO. For example
`c_THDoubleTensor_sigmoid` is applies a sigmoid transformation to values in a
tensor:

```
-- |c_THDoubleTensor_sigmoid : r_ t -> void
foreign import ccall "THTensorMath.h THDoubleTensor_sigmoid"
  c_THDoubleTensor_sigmoid :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

```

A common pattern in the TH API is for the first argument to be a tensor pointer
storing the result (`r_`), which is modified as a post-condition to the
procedure.

While it is possible to work with such an API directly, it quickly becomes
cumbersome, as many mathematical operations are pure functions, it is only due
to the underlying representation and FFI API that such operations must be done
monadically.

... (TODO - rules around use of unsafe operations)
