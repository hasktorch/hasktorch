# Hasktorch 0.2 Libtorch FFI

Work on ffi bindings into the c++ libtorch library in preparation for 0.2 which targets the pytorch's post 1.0libtorch backend.

General approach is to use generated `Declarations.yaml` spec instead of header parsing for code generation.

## Getting dependencies

`deps/` holds several external dependencies that are retrieved using the `deps/get-deps.sh` script.

This should be run prior to building

## XOR MLP Example

The following steps should run the xor mlp example:

```
# Download libtorch-binary and other shared library dependencies
pushd deps
./get-deps.sh
popd

# Set shared library environment variables
source setenv

stack build examples

stack exec xor_mlp
```

## Running code generation

Corde generation is used to build low-level FFI functions.

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

## Contributions

Contributions/PRs are welcome. 
