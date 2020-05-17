# Hasktorch 0.2 Pre-Release

Hasktorch is a library for tensors and neural networks in Haskell. It is an independent open source community project which leverages the core C++ libraries shared by PyTorch.

This project is in active development, so expect changes to the library API as it evolves. We encourage new users to join the slack for questions/discussions and contributious/PR are encouraged (see Contributing). Currently we are prepping development and migration for a major 2nd release.

## Project Structure

Basic functionality:

- `deps/` - submodules and downloads for build dependencies - libtorch, mklml, pytorch
- `examples/` - high level example models (xor mlp, typed cnn)
- `experimental/` - experimental projects or tips (jupyterlab)
- `hasktorch/` - higher level user-facing library, calls into `ffi/`, used by `examples/`

Internals (for contributing developers):

- `codegen/` - code generation, parses `Declarations.yaml` spec from pytorch and produces `ffi/` contents
- `inline-c/` - submodule to inline-cpp fork used for C++ FFI
- `libtorch-ffi/`- low level FFI bindings to libtorch
- `spec/` - specification files used for `codegen/`


## Getting Started

### On OSX or Ubuntu-like OSes'

The following steps run a toy linear regression example, assuming the hasktorch repository has just been cloned.

Starting at the top-level directory of the project, go to the `deps/` (dependencies) directory and run the `get-deps.sh` shell script to retrieve project dependencies with the following commands:

```
pushd deps           # Change to deps directory and save the current directory.
./get-deps.sh        # Run the shell script to retrieve the dependency. 
popd                 # Go back to the root directory of the project.
```

If you are using CUDA-9, replace `./get-deps.sh` with `./get-deps.sh -a cu92`. Likewise for CUDA-10, replace `./get-deps.sh` with `./get-deps.sh -a cu101`.

These downloads include various pytorch shared libraries. Note `get-deps.sh` only has to be run once when the repo is initially cloned.

Next, set shell environment to reference the shared library locations:


```
source setenv
```

Note `source setenv` should be run from the top-level directory of the repo.

### via nix-shell

Always the artifacts of hasktorch's master branch are uploaded to cachix.
If you setup cachix before using nix-shell, nix-shell will be faster.

```
nix-env -i cachix
cachix use hasktorch
nix-shell ./hasktorch/shell.nix
```

Will get you into a development environment for hasktorch using the CPU backend.
On NixOS you may also pass in a `cudaVersion` argument of `9` or `10` to provision a CUDA environment:

```
nix-shell ./hasktorch/shell.nix --arg cudaVersion 9 # or 10
```

If you are running cabal or stack to develop hasktorch, there is a shell hook to tell you which `extra-lib-dirs` and `extra-include-dirs` fields to include in your stack.yaml or cabal.project.local. This hook will also explain how to create a cabal.project.freeze file.

### Building examples

Finally, try building and running the linear regression example:

```
stack run regression
```

For additional examples, see the `examples/` directory.

### Set up development environement in VS Code.
If you want to develop the project in VS Code and get Haskell Tooling support,
you will need to install [HIE(Haskell IDE Enginer)](https://github.com/haskell/haskell-ide-engine).
Since this project uses the resolver version `lts-14.7`, so you will need to 
install and use the corresponding version of HIE which is `hie-8.6.5`.

And then install the [Haskell Language Server plugin](https://marketplace.visualstudio.com/items?itemName=alanz.vscode-hie-server). If you encounter the `hie executable missing, please make sure it is installed, see github.com/haskell/haskell-ide-engine` when starting VSCode,
first make sure that when you run:
```
which hie
```
It should give you an output.
And the path of the `hie` executable in the `settings.json` by adding:
```
"languageServerHaskell.hieExecutablePath": "~/.local/bin/hie-8.6.5",
``` 


## Using as a library in a project via `nix`

See the example project in `examples/library-example` for a `default.nix` that can be dropped alongside a .cabal file.


## Contributing

We welcome new contributors.

Contact Austin Huang or Sam Stites for access to the [hasktorch slack channel][slack]. You can send an email to [hasktorch@gmail.com][email] or on twitter as [@austinvhuang][austin-twitter] and [@SamStites][sam-twitter].

[email]:mailto:hasktorch@gmail.com
[austin-twitter]:https://twitter.com/austinvhuang
[sam-twitter]:https://twitter.com/samstites
[slack]:https://hasktorch.slack.com
[gitter-dh]:https://gitter.im/dataHaskell/Lobby

## Developer Information

See [the wiki](https://github.com/hasktorch/hasktorch/wiki) for developer information.
