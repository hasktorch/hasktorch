# Note on `native_functions.yaml`

If for some reason you'd want to use that file instead of Declarations.yaml, be aware that two of the functions listed in there(_dimI, _dimV) are problematic. This is because their dispatch field, which is usually a record, is just a string. Fortunately those are deprecated, so it is ok to filter them out.
