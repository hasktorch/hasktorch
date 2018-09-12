   let prelude = https://raw.githubusercontent.com/dhall-lang/dhall-to-cabal/master/dhall/prelude.dhall
in let types = https://raw.githubusercontent.com/dhall-lang/dhall-to-cabal/master/dhall/types.dhall
in let common = ../dhall/common.dhall
in let packages = common.packages
in let cabalvars = common.cabalvars
in let partials = ../dhall/backpack/partials.dhall
in let mixins = ../dhall/backpack/mixins.dhall
in let provides = ../dhall/backpack/provides.dhall
in let fn = ../dhall/common/functions.dhall
in let indef-support =
    λ(isth : Bool)
    → let lib = fn.showlib isth
    in
      [ { rename = "Torch.Sig.Index.Tensor"    , to = "Torch.FFI.${lib}.Long.Tensor"  }
      , { rename = "Torch.Sig.Index.TensorFree", to = "Torch.FFI.${lib}.Long.FreeTensor"  }
      , { rename = "Torch.Sig.Mask.Tensor"     , to = "Torch.FFI.${lib}.Byte.Tensor"  }
      , { rename = "Torch.Sig.Mask.TensorFree" , to = "Torch.FFI.${lib}.Byte.FreeTensor"  }
      , { rename = "Torch.Sig.Mask.MathReduce" , to = "Torch.FFI.${lib}.Byte.TensorMath"  }
      ]
in let indef-unsigned-reexports =
    [ fn.renameNoop "Torch.Indef.Index"
    , fn.renameNoop "Torch.Indef.Mask"
    , fn.renameNoop "Torch.Indef.Types"
    , fn.renameNoop "Torch.Indef.Storage"
    , fn.renameNoop "Torch.Indef.Storage.Copy"
    , fn.renameNoop "Torch.Indef.Static.Tensor"
    , fn.renameNoop "Torch.Indef.Static.Tensor.Copy"
    , fn.renameNoop "Torch.Indef.Static.Tensor.Index"
    , fn.renameNoop "Torch.Indef.Static.Tensor.Masked"
    , fn.renameNoop "Torch.Indef.Static.Tensor.Math"
    , fn.renameNoop "Torch.Indef.Static.Tensor.Math.Compare"
    , fn.renameNoop "Torch.Indef.Static.Tensor.Math.CompareT"
    , fn.renameNoop "Torch.Indef.Static.Tensor.Math.Pairwise"
    , fn.renameNoop "Torch.Indef.Static.Tensor.Math.Pointwise"
    , fn.renameNoop "Torch.Indef.Static.Tensor.Math.Reduce"
    , fn.renameNoop "Torch.Indef.Static.Tensor.Math.Scan"
    , fn.renameNoop "Torch.Indef.Static.Tensor.Mode"
    , fn.renameNoop "Torch.Indef.Static.Tensor.ScatterGather"
    , fn.renameNoop "Torch.Indef.Static.Tensor.Sort"
    , fn.renameNoop "Torch.Indef.Static.Tensor.TopK"
    , fn.renameNoop "Torch.Indef.Dynamic.Tensor"
    , fn.renameNoop "Torch.Indef.Dynamic.Tensor.Copy"
    , fn.renameNoop "Torch.Indef.Dynamic.Tensor.Index"
    , fn.renameNoop "Torch.Indef.Dynamic.Tensor.Masked"
    , fn.renameNoop "Torch.Indef.Dynamic.Tensor.Math"
    , fn.renameNoop "Torch.Indef.Dynamic.Tensor.Math.Compare"
    , fn.renameNoop "Torch.Indef.Dynamic.Tensor.Math.CompareT"
    , fn.renameNoop "Torch.Indef.Dynamic.Tensor.Math.Pairwise"
    , fn.renameNoop "Torch.Indef.Dynamic.Tensor.Math.Pointwise"
    , fn.renameNoop "Torch.Indef.Dynamic.Tensor.Math.Reduce"
    , fn.renameNoop "Torch.Indef.Dynamic.Tensor.Math.Scan"
    , fn.renameNoop "Torch.Indef.Dynamic.Tensor.Mode"
    , fn.renameNoop "Torch.Indef.Dynamic.Tensor.ScatterGather"
    , fn.renameNoop "Torch.Indef.Dynamic.Tensor.Sort"
    , fn.renameNoop "Torch.Indef.Dynamic.Tensor.TopK"
    ]

in let indef-signed-reexports =
    indef-unsigned-reexports #
     [ fn.renameNoop "Torch.Indef.Static.Tensor.Math.Pointwise.Signed"
     , fn.renameNoop "Torch.Indef.Dynamic.Tensor.Math.Pointwise.Signed"
     ]

in let indef-floating-reexports =
    indef-signed-reexports #
     [ fn.renameNoop "Torch.Indef.Dynamic.NN"
     , fn.renameNoop "Torch.Indef.Dynamic.Tensor.Math.Blas"
     , fn.renameNoop "Torch.Indef.Dynamic.Tensor.Math.Floating"
     , fn.renameNoop "Torch.Indef.Dynamic.Tensor.Math.Lapack"
     , fn.renameNoop "Torch.Indef.Dynamic.Tensor.Math.Pointwise.Floating"
     , fn.renameNoop "Torch.Indef.Dynamic.Tensor.Math.Reduce.Floating"
     , fn.renameNoop "Torch.Indef.Static.NN"
     , fn.renameNoop "Torch.Indef.Static.NN.Activation"
     , fn.renameNoop "Torch.Indef.Static.NN.Backprop"
     , fn.renameNoop "Torch.Indef.Static.NN.Conv1d"
     , fn.renameNoop "Torch.Indef.Static.NN.Conv2d"
     , fn.renameNoop "Torch.Indef.Static.NN.Criterion"
     , fn.renameNoop "Torch.Indef.Static.NN.Layers"
     , fn.renameNoop "Torch.Indef.Static.NN.Linear"
     , fn.renameNoop "Torch.Indef.Static.NN.Math"
     , fn.renameNoop "Torch.Indef.Static.NN.Padding"
     , fn.renameNoop "Torch.Indef.Static.NN.Pooling"
     , fn.renameNoop "Torch.Indef.Static.NN.Sampling"
     , fn.renameNoop "Torch.Indef.Static.Tensor.Math.Blas"
     , fn.renameNoop "Torch.Indef.Static.Tensor.Math.Floating"
     , fn.renameNoop "Torch.Indef.Static.Tensor.Math.Lapack"
     , fn.renameNoop "Torch.Indef.Static.Tensor.Math.Pointwise.Floating"
     , fn.renameNoop "Torch.Indef.Static.Tensor.Math.Reduce.Floating"
     , fn.renameNoop "Torch.Indef.Static.Tensor.Random.TH"
     , fn.renameNoop "Torch.Indef.Static.Tensor.Random.THC"
     , fn.renameNoop "Torch.Indef.Static.Tensor.Math.Random.TH"
     , fn.renameNoop "Torch.Indef.Dynamic.Tensor.Random.TH"
     , fn.renameNoop "Torch.Indef.Dynamic.Tensor.Random.THC"
     , fn.renameNoop "Torch.Indef.Dynamic.Tensor.Math.Random.TH"
     , fn.renameNoop "Torch.Undefined.Tensor.Math.Random.TH"
     , fn.renameNoop "Torch.Undefined.Tensor.Random.TH"
     , fn.renameNoop "Torch.Undefined.Tensor.Random.THC"
     ]

in let baseexports =
    λ(isth : Bool)  →
    λ(ttype : Text) →
    [ fn.renameNoop "Torch."++(if isth then "" else "Cuda.")++"${ttype}"
      , fn.renameNoop "Torch."++(if isth then "" else "Cuda.")++"${ttype}.Dynamic"
      , fn.renameNoop "Torch."++(if isth then "" else "Cuda.")++"${ttype}.Storage"
      ]
in let nnexports =
    λ(isth : Bool)  →
    λ(ttype : Text) →
      [ fn.renameNoop "Torch"++(if isth then "" else "Cuda.")++"${ttype}.NN"
      , fn.renameNoop "Torch"++(if isth then "" else "Cuda.")++"${ttype}.NN.Activation"
      , fn.renameNoop "Torch"++(if isth then "" else "Cuda.")++"${ttype}.NN.Backprop"
      , fn.renameNoop "Torch"++(if isth then "" else "Cuda.")++"${ttype}.NN.Conv1d"
      , fn.renameNoop "Torch"++(if isth then "" else "Cuda.")++"${ttype}.NN.Conv2d"
      , fn.renameNoop "Torch"++(if isth then "" else "Cuda.")++"${ttype}.NN.Criterion"
      , fn.renameNoop "Torch"++(if isth then "" else "Cuda.")++"${ttype}.NN.Layers"
      , fn.renameNoop "Torch"++(if isth then "" else "Cuda.")++"${ttype}.NN.Linear"
      , fn.renameNoop "Torch"++(if isth then "" else "Cuda.")++"${ttype}.NN.Math"
      , fn.renameNoop "Torch"++(if isth then "" else "Cuda.")++"${ttype}.NN.Padding"
      , fn.renameNoop "Torch"++(if isth then "" else "Cuda.")++"${ttype}.NN.Pooling"
      , fn.renameNoop "Torch"++(if isth then "" else "Cuda.")++"${ttype}.NN.Sampling"
      ]
in common.Package
   // { name = "hasktorch-core"
      , flags = [ common.flags.cuda, common.flags.lite ]
      , description = "core tensor abstractions wrapping raw TH bindings"
      , synopsis = "Torch for tensors and neural networks in Haskell"
      , executables =
          [ { name = "memcheck"
            , executable =
              λ(config : types.Config)
              → prelude.defaults.Executable
               // { main-is = "Memcheck.hs"
                  , build-depends =
                      [ packages.base
                      , packages.hasktorch-core
                      ]
                    , default-language = cabalvars.default-language
                    , hs-source-dirs = [ "exe" ]
                    } }
          ]
      , test-suites =
          [ { name = "spec"
            , test-suite =
              λ(config : types.Config)
              → prelude.defaults.TestSuite
               // { type = < exitcode-stdio = { main-is = "Spec.hs" } | detailed : { module : Text } >
                  , build-depends =
                      [ packages.QuickCheck
                      , packages.backprop
                      , packages.base
                      , packages.dimensions
                      , packages.ghc-typelits-natnormalise
                      , packages.hasktorch-core
                      , packages.hspec
                      , packages.singletons
                      , packages.text
                      , packages.mtl
                      , packages.microlens-platform
                      , packages.monad-loops
                      , packages.time
                      , packages.transformers
                      ]
                    , default-extensions = cabalvars.default-extensions
                    , default-language = cabalvars.default-language
                    , hs-source-dirs = [ "tests" ]
                    , other-modules =
                      [ "Orphans"
                      , "MemorySpec"
                      , "RawLapackSVDSpec"
                      , "GarbageCollectionSpec"
                      , "Torch.Prelude.Extras"
                      , "Torch.Core.LogAddSpec"
                      , "Torch.Core.RandomSpec"
                      , "Torch.Static.TensorSpec"
                      , "Torch.Static.NN.AbsSpec"
                      , "Torch.Static.NN.LinearSpec"
                      ]
                    }
            }
          ]
      , library =
        [ λ ( config : types.Config )
          → let
            cpu-lite-exports =
                [ fn.renameNoop "Torch.Types.Numeric" ]
                # baseexports True "Long"
                # baseexports True "Double"
                # nnexports   True "Double"
          in let
            cpu-full-exports = cpu-lite-exports #
                ( baseexports True "Byte"
                # baseexports True "Char"
                # baseexports True "Short"
                # baseexports True "Int"
                # baseexports True "Float"
                )
          in let
            gpu-lite-exports = cpu-lite-exports #
                ( baseexports False "Long"
                # baseexports False "Double"
                # nnexports   False "Double"
                )
          in let
            gpu-full-exports = cpu-full-exports #
                ( baseexports False "Byte"
                # baseexports False "Char"
                # baseexports False "Short"
                # baseexports False "Int"
                # baseexports False "Float"
                )
          in common.Library
          // { hs-source-dirs = [ "utils" ]
            , build-depends =
              [ packages.base
              , fn.anyver "hasktorch-core-cpu"
              , fn.anyver "hasktorch-core-gpu"
              , packages.hasktorch-types-th
              , packages.containers
              , packages.deepseq
              , packages.dimensions
              , packages.hasktorch-raw-th
              , packages.managed
              , packages.microlens
              , packages.numeric-limits
              , packages.safe-exceptions
              , packages.singletons
              , packages.text
              , packages.typelits-witnesses
              ]
            , exposed-modules =
              [ "Torch.Core.Exceptions"
              , "Torch.Core.Random"
              , "Torch.Core.LogAdd"
              ]
            , reexported-modules =
              if               config.flag "lite" && False == config.flag "cuda" then cpu-lite-exports
              else if False == config.flag "lite" && False == config.flag "cuda" then cpu-full-exports
              else if          config.flag "lite" &&          config.flag "cuda" then gpu-lite-exports
              else
                gpu-full-exports
            }
        ] : Optional (types.Config → types.Library)

      , sub-libraries =
        [ { name = "hasktorch-core-cpu"
          , library =
            λ (config : types.Config)
            → common.Library
            //
              { hs-source-dirs = [ "utils", "src" ]
              , build-depends =
                [ packages.base
                , packages.hasktorch-types-th
                , packages.containers
                , packages.deepseq
                , packages.dimensions
                , packages.hasktorch-raw-th
                , packages.hasktorch-types-th
                , packages.managed
                , packages.microlens
                , packages.numeric-limits
                , packages.safe-exceptions
                , packages.singletons
                , packages.text
                , packages.typelits-witnesses

                , packages.hasktorch-indef-floating
                , packages.hasktorch-indef-signed
                ] # (if config.flag "lite" then [] else [packages.hasktorch-indef-unsigned])
              , other-modules =
                [ "Torch.Core.Random"
                ]
              , reexported-modules = nnexports True "Double"
              , exposed-modules =
                  baseexports True "Long"
                # baseexports True "Double"
                # (if config.flag "lite" then [] else
                    baseexports True "Byte"
                  # baseexports True "Char"
                  # baseexports True "Short"
                  # baseexports True "Int"
                  # baseexports True "Float"
                  )
              , mixins =
                [ { package = "hasktorch-indef-signed"
                  , renaming =
                    { provides = prelude.types.ModuleRenaming.renaming (provides.signed True "Long")
                    , requires = prelude.types.ModuleRenaming.renaming (mixins.signed True "Long" # indef-support True)
                    } }
                , { package = "hasktorch-indef-floating"
                  , renaming =
                    { provides = prelude.types.ModuleRenaming.renaming (provides.floating True "Double")
                    , requires = prelude.types.ModuleRenaming.renaming (mixins.floating True "Double" # indef-support True)
                    } }
                ] # if config.flag "lite" then [] else
                  [ { package = "hasktorch-indef-unsigned"
                    , renaming =
                      { provides = prelude.types.ModuleRenaming.renaming (provides.unsigned True "Byte")
                      , requires = prelude.types.ModuleRenaming.renaming (mixins.unsigned True "Byte" # indef-support True)
                      } }
                  , { package = "hasktorch-indef-unsigned"
                    , renaming =
                      { provides = prelude.types.ModuleRenaming.renaming (provides.unsigned True "Char")
                      , requires = prelude.types.ModuleRenaming.renaming (mixins.unsigned True "Char" # indef-support True)
                      } }
                  , { package = "hasktorch-indef-unsigned"
                    , renaming =
                      { provides = prelude.types.ModuleRenaming.renaming (provides.unsigned True "Short")
                      , requires = prelude.types.ModuleRenaming.renaming (mixins.unsigned True "Short" # indef-support True)
                      } }
                  , { package = "hasktorch-indef-unsigned"
                    , renaming =
                      { provides = prelude.types.ModuleRenaming.renaming (provides.signed True "Int")
                      , requires = prelude.types.ModuleRenaming.renaming (mixins.signed True "Int" # indef-support True)
                      } }
                  , { package = "hasktorch-indef-unsigned"
                    , renaming =
                      { provides = prelude.types.ModuleRenaming.renaming (provides.floating True "Float")
                      , requires = prelude.types.ModuleRenaming.renaming (mixins.floating True "Float" # indef-support True)
                      } }
                  ] }
        , { name = "hasktorch-core-gpu"
          , library =
            λ (config : types.Config)
            → common.Library
            //
              { hs-source-dirs = [ "utils", "src" ]
              , build-depends =
                [ packages.base
                , packages.containers
                , packages.deepseq
                , packages.dimensions
                , packages.hasktorch-raw-th
                , packages.hasktorch-types-th
                , packages.hasktorch-raw-thc
                , packages.hasktorch-types-thc
                , packages.managed
                , packages.microlens
                , packages.numeric-limits
                , packages.safe-exceptions
                , packages.singletons
                , packages.text
                , packages.typelits-witnesses

                , packages.hasktorch-indef-floating
                , packages.hasktorch-indef-signed
                ] # (if config.flag "lite" then [] else [packages.hasktorch-indef-unsigned])
              , reexported-modules = nnexports False "Double"
              , exposed-modules =
                  baseexports False "Long"
                # baseexports False "Double"
                # (if config.flag "lite" then [] else
                    baseexports False "Byte"
                  # baseexports False "Char"
                  # baseexports False "Short"
                  # baseexports False "Int"
                  # baseexports False "Float"
                  )
              , cpp-options = [ "-DHASKTORCH_INTERNAL_CUDA" ] # (if config.flag "cuda" then ["-DCUDA"] else [])
              , mixins =
                [ { package = "hasktorch-indef-signed"
                  , renaming =
                    { provides = prelude.types.ModuleRenaming.renaming (provides.signed False "Long")
                    , requires = prelude.types.ModuleRenaming.renaming (mixins.signed False "Long" # indef-support False)
                    } }
                , { package = "hasktorch-indef-floating"
                  , renaming =
                    { provides = prelude.types.ModuleRenaming.renaming (provides.floating False "Double")
                    , requires = prelude.types.ModuleRenaming.renaming (mixins.floating False "Double" # indef-support False)
                    } }
                ] # if config.flag "lite" then [] else
                  [ { package = "hasktorch-indef-unsigned"
                    , renaming =
                      { provides = prelude.types.ModuleRenaming.renaming (provides.unsigned False "Byte")
                      , requires = prelude.types.ModuleRenaming.renaming (mixins.unsigned False "Byte" # indef-support False)
                      } }
                  , { package = "hasktorch-indef-unsigned"
                    , renaming =
                      { provides = prelude.types.ModuleRenaming.renaming (provides.unsigned False "Char")
                      , requires = prelude.types.ModuleRenaming.renaming (mixins.unsigned False "Char" # indef-support False)
                      } }
                  , { package = "hasktorch-indef-unsigned"
                    , renaming =
                      { provides = prelude.types.ModuleRenaming.renaming (provides.unsigned False "Short")
                      , requires = prelude.types.ModuleRenaming.renaming (mixins.unsigned False "Short" # indef-support False)
                      } }
                  , { package = "hasktorch-indef-unsigned"
                    , renaming =
                      { provides = prelude.types.ModuleRenaming.renaming (provides.signed False "Int")
                      , requires = prelude.types.ModuleRenaming.renaming (mixins.signed False "Int" # indef-support False)
                      } }
                  , { package = "hasktorch-indef-unsigned"
                    , renaming =
                      { provides = prelude.types.ModuleRenaming.renaming (provides.floating False "Float")
                      , requires = prelude.types.ModuleRenaming.renaming (mixins.floating False "Float" # indef-support False)
                      } }
                  ] }
        , { name = "hasktorch-indef-unsigned"
          , library =
            λ (config : types.Config)
            → common.Library
            //
              { build-depends =
                [ packages.base
                , packages.hasktorch-partial
                , packages.hasktorch-indef
                ]
              , reexported-modules = indef-unsigned-reexports
              , mixins =
                [ { package = "hasktorch-indef"
                  , renaming =
                    { provides = prelude.types.ModuleRenaming.default {=}
                    , requires = prelude.types.ModuleRenaming.renaming
                      [ { rename = "Torch.Sig.NN"                             , to = "Torch.Undefined.NN" }
                      , { rename = "Torch.Sig.Types.NN"                       , to = "Torch.Undefined.Types.NN" }
                      , { rename = "Torch.Sig.Tensor.Math.Blas"               , to = "Torch.Undefined.Tensor.Math.Blas" }
                      , { rename = "Torch.Sig.Tensor.Math.Floating"           , to = "Torch.Undefined.Tensor.Math.Floating" }
                      , { rename = "Torch.Sig.Tensor.Math.Lapack"             , to = "Torch.Undefined.Tensor.Math.Lapack" }
                      , { rename = "Torch.Sig.Tensor.Math.Pointwise.Signed"   , to = "Torch.Undefined.Tensor.Math.Pointwise.Signed" }
                      , { rename = "Torch.Sig.Tensor.Math.Pointwise.Floating" , to = "Torch.Undefined.Tensor.Math.Pointwise.Floating" }
                      , { rename = "Torch.Sig.Tensor.Math.Reduce.Floating"    , to = "Torch.Undefined.Tensor.Math.Reduce.Floating" }
                      , { rename = "Torch.Sig.Tensor.Math.Random.TH"          , to = "Torch.Undefined.Tensor.Math.Random.TH" }
                      , { rename = "Torch.Sig.Tensor.Random.TH"               , to = "Torch.Undefined.Tensor.Random.TH" }
                      , { rename = "Torch.Sig.Tensor.Random.THC"              , to = "Torch.Undefined.Tensor.Random.THC" }
                      ]
                    } }
                 ]
              }
          }
        , { name = "hasktorch-indef-signed"
          , library =
            λ (config : types.Config)
            → common.Library
            //
              { build-depends =
                [ packages.base
                , packages.hasktorch-partial
                , packages.hasktorch-indef
                ]
              , reexported-modules = indef-signed-reexports
              , mixins =
                [ { package = "hasktorch-indef"
                  , renaming =
                    { provides = prelude.types.ModuleRenaming.default {=}
                    , requires = prelude.types.ModuleRenaming.renaming
                      [ { rename = "Torch.Sig.NN"                             , to = "Torch.Undefined.NN" }
                      , { rename = "Torch.Sig.Types.NN"                       , to = "Torch.Undefined.Types.NN" }
                      , { rename = "Torch.Sig.Tensor.Math.Blas"               , to = "Torch.Undefined.Tensor.Math.Blas" }
                      , { rename = "Torch.Sig.Tensor.Math.Floating"           , to = "Torch.Undefined.Tensor.Math.Floating" }
                      , { rename = "Torch.Sig.Tensor.Math.Lapack"             , to = "Torch.Undefined.Tensor.Math.Lapack" }
                      , { rename = "Torch.Sig.Tensor.Math.Pointwise.Floating" , to = "Torch.Undefined.Tensor.Math.Pointwise.Floating" }
                      , { rename = "Torch.Sig.Tensor.Math.Reduce.Floating"    , to = "Torch.Undefined.Tensor.Math.Reduce.Floating" }
                      , { rename = "Torch.Sig.Tensor.Math.Random.TH"          , to = "Torch.Undefined.Tensor.Math.Random.TH" }
                      , { rename = "Torch.Sig.Tensor.Random.TH"               , to = "Torch.Undefined.Tensor.Random.TH" }
                      , { rename = "Torch.Sig.Tensor.Random.THC"              , to = "Torch.Undefined.Tensor.Random.THC" }
                      ]
                    } }
                 ]
              }
          }
        , { name = "hasktorch-indef-floating"
          , library =
            λ (config : types.Config)
            → common.Library
            //
              { build-depends =
                [ packages.base
                , packages.hasktorch-indef
                ]
              , reexported-modules = indef-floating-reexports
              }
          }
        ]
      }
