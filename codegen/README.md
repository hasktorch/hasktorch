# hasktorch-codegen

Parse source files from the [TH][th] library and generate low-level bindings in
Haskell.

**Warning - parsers are only "good enough" to process their intended C source
inputs. They are not intended for general purpose use.**

## Code Generation

The [`raw/`][raw] modules are generated using the scripts in
[`codegen/`][codegen]. Since the outputs are already part of the repo, you
should not need to run [`codegen/`][codegen] programs to use hasktorch.

However, if you are contributing to hasktorch itself, you may want to
modify/re-generate the code generation processes. Currently there are three main
operations:

- `stack exec codegen-generic` - Builds generic modules (one version per tensor type).
- `stack exec codegen-concrete` - Builds non-generic modules.

All of these programs write `.hs` files into the [`output/`][output] directory
as a staging area (rather than overwriting [`raw/`][raw] directly).

For details on the TH library's pseudo-templating preprocessor mechanism for
underlying the generic modules, see [Adam Paszke's
writeup](https://apaszke.github.io/torch-internals.html).

[th]: https://github.com/pytorch/pytorch/tree/master/torch/lib/TH

<!-- project directory links -->

[codegen]: ./codegen/
[core]: ./core/
[examples]: ./examples/
[output]: ./output/
[raw]: ./raw/
[vendor]: ./vendor/

