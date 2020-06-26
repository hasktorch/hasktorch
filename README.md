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

### Stack with Nix

It is also possible to compile hasktorch with Stack while getting system
dependencies with Nix.

First, make sure both Stack and Nix are installed, and then optionally enable
the hasktorch Cachix, as described above.  After that, just run
`stack --nix build` to build.

As long as you pass the `--nix` flag to Stack, Stack will use Nix to get into
an environment with all required system dependencies (mostly just `libtorch`)
before running builds, tests, etc.

Note that if you are running `stack` with Nix, you want to make sure you have
_not_ run the `deps/get-deps.sh` script.  In particular, the `deps/libtorch/` and
`deps/mklml/` directories must not exist.

### Building examples

Finally, try building and running the linear regression example:

```
stack run regression
```

For additional examples, see the `examples/` directory.

### Set up development environment

[ghcide](https://github.com/digital-asset/ghcide) is currently the preferred language server backend, and is provided by the Nix shell environment. If you are not using Nix, the `ghcide` binary can be installed using `stack` or `cabal`; see instructions [here](https://github.com/digital-asset/ghcide#with-cabal-or-stack).

You will then want to install an integration for your preferred editor. `ghcide` supports [VS Code](https://github.com/digital-asset/ghcide#using-with-vs-code), [Atom](https://github.com/digital-asset/ghcide#using-with-atom), [Sublime Text](https://github.com/digital-asset/ghcide#using-with-sublime-text), [Emacs](https://github.com/digital-asset/ghcide#using-with-emacs), [Vim](https://github.com/digital-asset/ghcide#using-with-vimneovim), and [others](https://github.com/digital-asset/ghcide).


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
