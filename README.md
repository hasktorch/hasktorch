# Hasktorch 0.2 Libtorch 1.0 FFI

Work on ffi bindings into the c++ libtorch library in preparation for 0.2 which targets the pytorch's post 1.0libtorch backend.

General approach is to use generated `Declarations.yaml` spec instead of header parsing for code generation.

## getting dependencies

`deps/` holds several external dependencies that are retrieved using the `deps/get-deps.sh` script.

This should be run prior to building

## yaml -> binding codegen (WIP)

To run:

```
stack build codegen
stack exec codegen-exe
```

To get CLI options:

```
stack exec codegen-exe -- --help
```

## ffi testing (WIP)

To run without repl:

```
#Download libtorch-binary
pushd deps
./get-deps.sh
popd

#Set environment variable(LD_LIBRARY_PATH)
source setenv

#Build and test with stack
stack test

#Build and test with cabal
#setup cabal.project.freeze using stackage, extra-include-dirs and extra-lib-dirs for cabal
./setup-cabal.sh
cabal new-test all
```

To run (currently works in the repl):

```
stack ghci --ghc-options='-fobject-code'  ffi
...
Prelude Main> main
Hello torch!
 0.2401  0.0901  0.9807
 0.9168  0.3757  0.4029
[ Variable[CPUFloatType]{2,3} ]
 1  1
 1  1
[ Variable[CPUFloatType]{2,2} ]
 0.1232  1.5721
 0.5392  0.2395
[ Variable[CPUFloatType]{2,2} ]
 1.1232  2.5721
 1.5392  1.2395
[ Variable[CPUFloatType]{2,2} ]
Prelude Main>
```

## Contributions

Contributions/PRs are welcome. 
