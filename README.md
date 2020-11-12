# Hasktorch 0.2 Pre-Release

Hasktorch is a library for tensors and neural networks in Haskell.
It is an independent open source community project which leverages the core C++ libraries shared by PyTorch.

This project is in active development, so expect changes to the library API as it evolves.
We would like to invite new users to [join our Hasktorch slack space](#contributing) for questions and discussions. [Contributions/PR are encouraged](#contributing).

Currently we are developing the second major release of Hasktorch (0.2). Note the 1st release, Hasktorch 0.1, on hackage is outdated and _should not_ be used.

## Documentation

The documentation is divided into several sections:

- [Introductory Videos](#introductory-videos)
- [Getting Started](#getting-started)
- [Known Issues](#known-issues)
- [Contributing](#contributing)
- [Notes for Library Developers](#notes-for-library-developers)

## Introductory Videos

- [High-level MuniHac talk](https://www.youtube.com/watch?v=Qu6RIO02m1U&t=360) by [@austinvhuang][austin-twitter]
- [Hands-on live-coding demo](https://www.youtube.com/watch?v=ZnYa99QoznE&t=1689) by [@tscholak][torsten-twitter]
- [Low-level FFI talk](https://www.youtube.com/watch?v=qWpD8t_Aodk&feature=youtu.be) by [@junjihashimoto][junji-twitter]


## Getting Started

The following steps will get you started.
They assume the hasktorch repository has just been cloned.

* [linux+cabal+cpu](#linuxcabalcpu)
* [linux+cabal+cuda11](#linuxcabalcuda11)
* [macos+cabal+cpu](#macoscabalcpu)
* [nixos+cabal+cpu](#nixoscabalcpu)
* [nixos+cabal+cuda11](#nixoscabalcuda11)


### linux+cabal+cpu

Starting from the top-level directory of the project, run:

```sh
$ pushd deps     # Change to the deps directory and save the current directory.
$ ./get-deps.sh  # Run the shell script to retrieve the libtorch dependencies.
$ popd           # Go back to the root directory of the project.
$ source setenv  # Set the shell environment to reference the shared library locations.
```

To build and test the Hasktorch library, run:

```sh
$ cabal build hasktorch  # Build the Hasktorch library.
$ cabal test hasktorch   # Build and run the Hasktorch library test suite.
```

To build and test the example executables shipped with hasktorch, run:

```sh
$ cabal build examples  # Build the Hasktorch examples.
$ cabal test examples   # Build and run the Hasktorch example test suites.
```

To run the MNIST CNN example, run:

```sh
$ cd examples                   # Change to the examples directory.
$ ./datasets/download-mnist.sh  # Download the MNIST dataset.
$ mv mnist data                 # Move the MNIST dataset to the data directory.
$ export DEVICE=cpu             # Set device to CPU for the MNIST CNN example.
$ cabal run static-mnist-cnn    # Run the MNIST CNN example.
```


### linux+cabal+cuda11

Starting from the top-level directory of the project, run:

```sh
$ pushd deps             # Change to the deps directory and save the current directory.
$ ./get-deps.sh -a cu11  # Run the shell script to retrieve the libtorch dependencies.
$ popd                   # Go back to the root directory of the project.
$ source setenv          # Set the shell environment to reference the shared library locations.
```

To build and test the Hasktorch library, run:

```sh
$ cabal build hasktorch  # Build the Hasktorch library.
$ cabal test hasktorch   # Build and run the Hasktorch library test suite.
```

To build and test the example executables shipped with hasktorch, run:

```sh
$ cabal build examples  # Build the Hasktorch examples.
$ cabal test examples   # Build and run the Hasktorch example test suites.
```

To run the MNIST CNN example, run:

```sh
$ cd examples                   # Change to the examples directory.
$ ./datasets/download-mnist.sh  # Download the MNIST dataset.
$ mv mnist data                 # Move the MNIST dataset to the data directory.
$ export DEVICE="cuda:0"        # Set device to CUDA for the MNIST CNN example.
$ cabal run static-mnist-cnn    # Run the MNIST CNN example.
```


### macos+cabal+cpu

Starting from the top-level directory of the project, run:

```sh
$ pushd deps     # Change to the deps directory and save the current directory.
$ ./get-deps.sh  # Run the shell script to retrieve the libtorch dependencies.
$ popd           # Go back to the root directory of the project.
$ source setenv  # Set the shell environment to reference the shared library locations.
```

To build and test the Hasktorch library, run:

```sh
$ cabal build hasktorch  # Build the Hasktorch library.
$ cabal test hasktorch   # Build and run the Hasktorch library test suite.
```

To build and test the example executables shipped with hasktorch, run:

```sh
$ cabal build examples  # Build the Hasktorch examples.
$ cabal test examples   # Build and run the Hasktorch example test suites.
```

To run the MNIST CNN example, run:

```sh
$ cd examples                   # Change to the examples directory.
$ ./datasets/download-mnist.sh  # Download the MNIST dataset.
$ mv mnist data                 # Move the MNIST dataset to the data directory.
$ export DEVICE=cpu             # Set device to CPU for the MNIST CNN example.
$ cabal run static-mnist-cnn    # Run the MNIST CNN example.
```


### nixos+cabal+cpu

(Optional) Install and set up Cachix:

```sh
$ nix-env -iA cachix -f https://cachix.org/api/v1/install  # (Optional) Install Cachix.
$ cachix use iohk                                          # (Optional) Use IOHK's cache.
$ cachix use hasktorch                                     # (Optional) Use hasktorch's cache.
```

Starting from the top-level directory of the project, run:

```sh
$ nix-shell  # Enter the nix shell environment for Hasktorch.
```

To build and test the Hasktorch library, run:

```sh
$ cabal build hasktorch  # Build the Hasktorch library.
$ cabal test hasktorch   # Build and run the Hasktorch library test suite.
```

To build and test the example executables shipped with hasktorch, run:

```sh
$ cabal build examples  # Build the Hasktorch examples.
$ cabal test examples   # Build and run the Hasktorch example test suites.
```

To run the MNIST CNN example, run:

```sh
$ cd examples                   # Change to the examples directory.
$ ./datasets/download-mnist.sh  # Download the MNIST dataset.
$ mv mnist data                 # Move the MNIST dataset to the data directory.
$ export DEVICE=cpu             # Set device to CPU for the MNIST CNN example.
$ cabal run static-mnist-cnn    # Run the MNIST CNN example.
```


### nixos+cabal+cuda11

(Optional) Install and set up Cachix:

```sh
$ nix-env -iA cachix -f https://cachix.org/api/v1/install  # (Optional) Install Cachix.
$ cachix use iohk                                          # (Optional) Use IOHK's cache.
$ cachix use hasktorch                                     # (Optional) Use hasktorch's cache.
```

Starting from the top-level directory of the project, run:

```sh
$ nix-shell --arg cudaSupport true --argstr cudaMajorVersion 11  # Enter the nix shell environment for Hasktorch.
```

To build and test the Hasktorch library, run:

```sh
$ cabal build hasktorch  # Build the Hasktorch library.
$ cabal test hasktorch   # Build and run the Hasktorch library test suite.
```

To build and test the example executables shipped with hasktorch, run:

```sh
$ cabal build examples  # Build the Hasktorch examples.
$ cabal test examples   # Build and run the Hasktorch example test suites.
```

To run the MNIST CNN example, run:

```sh
$ cd examples                   # Change to the examples directory.
$ ./datasets/download-mnist.sh  # Download the MNIST dataset.
$ mv mnist data                 # Move the MNIST dataset to the data directory.
$ export DEVICE="cuda:0"        # Set device to CUDA for the MNIST CNN example.
$ cabal run static-mnist-cnn    # Run the MNIST CNN example.
```


## Known Issues

### Tensors Cannot Be Moved to CUDA

In rare cases, you may see errors like

```
cannot move tensor to "CUDA:0"
```

although you have CUDA capable hardware in your machine and
have followed the getting-started instructions for CUDA support.

If that happens, check if `/run/opengl-driver/lib` exists.
If not, make sure your CUDA drivers are installed correctly.


### Weird Behaviour When Switching from CPU-Only to CUDA-Enabled Nix Shell

If you have run `cabal` in a CPU-only Hasktorch Nix shell before,
you may need to:

* Clean the `dist-newstyle` folder using `cabal clean`.
* Delete the `.ghc.environment*` file in the Hasktorch root folder.

Otherwise, at best, you will not be able to move tensors to CUDA,
and, at worst, you will see weird linker errors like

```
gcc: error: hasktorch/dist-newstyle/build/x86_64-linux/ghc-8.8.3/libtorch-ffi-1.5.0.0/build/Torch/Internal/Unmanaged/Autograd.dyn_o: No such file or directory
`cc' failed in phase `Linker'. (Exit code: 1)
```


## Contributing

We welcome new contributors.

Contact Austin Huang or Sam Stites for access to the [hasktorch slack channel][slack].
You can send an email to [hasktorch@gmail.com][email] or on twitter as [@austinvhuang][austin-twitter],
[@SamStites][sam-twitter], or [@tscholak][torsten-twitter].

[email]:mailto:hasktorch@gmail.com
[austin-twitter]:https://twitter.com/austinvhuang
[sam-twitter]:https://twitter.com/samstites
[torsten-twitter]:https://twitter.com/tscholak
[junji-twitter]:https://twitter.com/junjihashimoto3
[slack]:https://hasktorch.slack.com
[gitter-dh]:https://gitter.im/dataHaskell/Lobby

## Notes for library developers

See [the wiki](https://github.com/hasktorch/hasktorch/wiki) for developer notes.

### Project Folder Structure

Basic functionality:

- `deps/` - submodules and downloads for build dependencies (libtorch, mklml, pytorch) -- you can ignore this if you are on Nix
- `examples/` - high level example models (xor mlp, typed cnn, etc.)
- `experimental/` - experimental projects or tips
- `hasktorch/` - higher level user-facing library, calls into `ffi/`, used by `examples/`

Internals (for contributing developers):

- `codegen/` - code generation, parses `Declarations.yaml` spec from pytorch and produces `ffi/` contents
- `inline-c/` - submodule to inline-cpp fork used for C++ FFI
- `libtorch-ffi/`- low level FFI bindings to libtorch
- `spec/` - specification files used for `codegen/`

