# Hasktorch 0.2 Pre-Release

Hasktorch is a library for tensors and neural networks in Haskell. It is an independent open source community project which leverages the core C++ libraries shared by PyTorch. 

This project is in early development and should only be used by contributing developers. Expect substantial changes to the library API as it evolves. Contributions and PRs are welcome.

Currently we are prepping development and migration for a major 2nd release (please excuse sparsity of docs in the meantime). If you're interested in details or contributing please get in touch (see contributing).

## Project Structure

- `codegen/` - code generation, parses `Declarations.yaml` spec from pytorch and produces `ffi/` contents
- `deps/` - submodules for dependencies - libtorch, mklml, pytorch
- `examples/` - high level example models (xor mlp, typed cnn)
- `ffi/`- low level FFI bindings to libtorch
- `hasktorch/` - higher level user-facing library, calls into `ffi/`, used by `examples/`
- `inline-c/` - submodule to inline-cpp fork used for C++ FFI
- `spec/` - specification files used for `codegen/`

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

## Contributing

We welcome new contributors. 

Contact Austin Huang or Sam Stites for access to the [hasktorch slack channel][slack]. You can send an email to [hasktorch@gmail.com][email] or on twitter as [@austinvhuang][austin-twitter] and [@SamStites][sam-twitter].

[email]:mailto:hasktorch@gmail.com
[austin-twitter]:https://twitter.com/austinvhuang
[sam-twitter]:https://twitter.com/samstites
[slack]:https://hasktorch.slack.com 
[gitter-dh]:https://gitter.im/dataHaskell/Lobby

## Developer Information

See [the wiki](https://github.com/hasktorch/ffi-experimental/wiki) for developer information.
