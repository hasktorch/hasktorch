
# Note on `native_functions_modified.yaml`

This version introduces a minor change of removing 2 legacy functions from the spec. The reaosn is that parsing the dispatch field is problematic - normally it's a record but in these cases it's a string. Without a tag field, the parse is ill-defined. This could be fixed by preprocessing the `yaml` file, but it doesn't seem worth doing given these two methods are deprecated.

```
$ diff native_functions.yaml native_functions_modified.yaml
2178,2183c2178,2183
< # legacy method
< - func: _dimI(Tensor self) -> int64_t
<   variants: method
<   dispatch: sparse_dim_sparse
<   requires_tensor: True
<   device_guard: False
---
> # # legacy method
> # - func: _dimI(Tensor self) -> int64_t
> #   variants: method
> #   dispatch: sparse_dim_sparse
> #   requires_tensor: True
> #   device_guard: False
2194,2199c2194,2199
< # legacy method
< - func: _dimV(Tensor self) -> int64_t
<   variants: method
<   dispatch: dense_dim_sparse
<   requires_tensor: True
<   device_guard: False
---
> # # legacy method
> # - func: _dimV(Tensor self) -> int64_t
> #   variants: method
> #   dispatch: dense_dim_sparse
> #   requires_tensor: True
> #   device_guard: False
```