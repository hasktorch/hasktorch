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

For more background, refer to references on TH internals:

- [Torch Internals Writeup by Adam Paszke](https://apaszke.github.io/torch-internals.html) 
- [A Tour of PyTorch Internals (Part I)](http://pytorch.org/2017/05/11/Internals.html)
- [Haskell/Torch binding using backpack](http://blog.ezyang.com/2017/08/backpack-for-deep-learning/).
