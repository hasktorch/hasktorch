See `examples/README.md` for additional info.

## Getting dependencies

`deps/` holds several external dependencies that are retrieved using the `deps/get-deps.sh` script.

This should be run prior to building

## Running code generation

Code generation is used to build low-level FFI functions.

Note that the code is already generated in this repo under `ffi`, running this is only needed if changes are being made to the code generation process.

To run:

```
stack build codegen
stack exec codegen-exe
```

To get CLI options:

```
stack exec codegen-exe -- --help
```
