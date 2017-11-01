# torch-raw

Comprehensive Raw bindings to core TorcH (TH) C library. This library includes
hundreds of functions with varied math and tensor operations.

The `generic/` subdirectory corresponds to templated `generic/` directory in the
`TH` library.

Each tensor type-templated module in `TH/generic` is written to separate Haskell
modules, one for each type. Thus the convention for both files and function
prefixes follows `TH[Tensor-Type][ModuleName]`.

Convention for function names is that prepending function names with `c_`
correspond to raw bindings to functions, while prepending function names `p_`
corresponds bindings to function pointers.

Some functions will have exit conditions for handling errors (handled by
`THError(...)`), for example, if tensor dimensions are mismatched for a certain
function. These conditions are not checked or captured by these raw bindings.
Instead, preconditions are expected to be handled by the higher-level APIs
either via the type representation or by conditional data checks.

For more background, refer to references on TH internals:

- [Torch Internals Writeup by Adam Paszke](https://apaszke.github.io/torch-internals.html) 
- [A Tour of PyTorch Internals (Part I)](http://pytorch.org/2017/05/11/Internals.html)
- [Haskell/Torch binding using backpack](http://blog.ezyang.com/2017/08/backpack-for-deep-learning/).
