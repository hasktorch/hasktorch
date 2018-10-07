   let prelude = ../../dhall-to-cabal/dhall/prelude.dhall
in let types = ../../dhall-to-cabal/dhall/types.dhall
in let common = ../dhall/common.dhall
in let packages = common.packages
in let cabalvars = common.cabalvars
in let partials = ../dhall/backpack/partials.dhall
in let mixins = ../dhall/backpack/mixins.dhall
in let provides = ../dhall/backpack/provides.dhall
in let sub-libraries = ../dhall/backpack/sublibraries.dhall
in let fn = ../dhall/common/functions.dhall
in common.Package
   // { name = "hasktorch-indef"
      , flags = [ common.flags.cuda ]
      , synopsis = "Torch for tensors and neural networks in Haskell"
      , description = "Core tensor abstractions wrapping raw TH bindings"
      , test-suites =
          [ { name = "spec-double-th"
            , test-suite =
              λ(config : types.Config)
              →
              let isth = True
              in prelude.defaults.TestSuite
               // { type = < exitcode-stdio = { main-is = "Spec.hs" } | detailed : { module : Text } >
                  , build-depends =
                      [ packages.QuickCheck
                      , packages.base
                      , packages.dimensions
                      , fn.anyver "hasktorch-indef-floating"
                      , packages.hspec
                      , packages.singletons
                      , packages.text
                      , packages.hasktorch-raw-th
                      , packages.hasktorch-types-th
                      ]
                    , default-extensions = cabalvars.default-extensions
                    , default-language = cabalvars.default-language
                    , hs-source-dirs = [ "tests" ]
                    , other-modules =
                      [ "Torch.Indef.StorageSpec"
                      , "Torch.Indef.Dynamic.TensorSpec"
                      ]
                    , mixins =
                      [ { package = "hasktorch-indef-floating"
                        , renaming =
                          { provides = prelude.types.ModuleRenaming.default {=}
                          , requires = prelude.types.ModuleRenaming.renaming
                            ( mixins.floatingbase isth "Double"
                            # mixins.randombase isth "Double" False
                            )
                          } } ]
                    }
            }
          ]
      , sub-libraries =
        [ sub-libraries.floating
        ]

      , library =
        [ λ(config : types.Config)
        → let basedeps =
          [ packages.base
          , packages.backprop
          , packages.containers
          , packages.deepseq
          , packages.dimensions
          , packages.hasktorch-signatures
          , packages.hasktorch-signatures-support
          , packages.hasktorch-types-th
          , packages.hasktorch-raw-th
          , packages.managed
          , packages.mtl
          -- , packages.numeric-limits -- for show?
          , packages.safe-exceptions
          , packages.singletons
          , packages.ghc-typelits-natnormalise -- replace with thorin?
          , packages.transformers
          , packages.text
          -- , packages.typelits-witness -- old dep?
          ]
        in common.Library
          // { hs-source-dirs = [ "src" ]
             , build-depends =
               if config.flag "cuda" == False
               then  basedeps
               else (basedeps #
                 [ packages.hasktorch-types-thc
                 , packages.cuda
                 ])
             , cpp-options =
               if config.flag "cuda"
               then [ "-DCUDA", "-DHASKTORCH_INTERNAL_CUDA" ]
               else [] : List Text
             , other-modules = [ "Torch.Indef.Internal" ]
             , exposed-modules = fn.getnames sub-libraries.allindef-reexports
               -- FIXME: not sure where to go with commented-out classes, they are TH-only but there are some THC-only things as well
               -- [ "Torch.Class.Blas"
               -- , "Torch.Class.Lapack"
               -- , "Torch.Class.Tensor.Conv"
               -- , "Torch.Class.Vector"

               -- , "Torch.Indef.Index"
               -- , "Torch.Indef.Mask"

               -- ==================================================== --
               -- Dynamic Tensor modules
               -- ==================================================== --
               -- , "Torch.Indef.Dynamic.Tensor"
               -- , "Torch.Indef.Dynamic.Tensor.Copy"
               -- , "Torch.Indef.Dynamic.Tensor.Index"
               -- , "Torch.Indef.Dynamic.Tensor.Masked"
               -- , "Torch.Indef.Dynamic.Tensor.Math"
               -- , "Torch.Indef.Dynamic.Tensor.Math.Compare"
               -- , "Torch.Indef.Dynamic.Tensor.Math.CompareT"
               -- , "Torch.Indef.Dynamic.Tensor.Math.Pairwise"
               -- , "Torch.Indef.Dynamic.Tensor.Math.Pointwise"
               -- , "Torch.Indef.Dynamic.Tensor.Math.Reduce"
               -- , "Torch.Indef.Dynamic.Tensor.Math.Random.TH"
               -- , "Torch.Indef.Dynamic.Tensor.Math.Scan"
               -- , "Torch.Indef.Dynamic.Tensor.Mode"
               -- , "Torch.Indef.Dynamic.Tensor.Random.TH"
               -- , "Torch.Indef.Dynamic.Tensor.Random.THC"
               -- , "Torch.Indef.Dynamic.Tensor.ScatterGather"
               -- , "Torch.Indef.Dynamic.Tensor.Sort"
               -- , "Torch.Indef.Dynamic.Tensor.TopK"

               -- , "Torch.Indef.Dynamic.Tensor.Math.Pointwise.Signed"

               -- , "Torch.Indef.Dynamic.NN"
               -- , "Torch.Indef.Dynamic.NN.Activation"
               -- , "Torch.Indef.Dynamic.NN.Criterion"
               -- , "Torch.Indef.Dynamic.NN.Pooling"
               -- , "Torch.Indef.Dynamic.Tensor.Math.Blas"
               -- , "Torch.Indef.Dynamic.Tensor.Math.Floating"
               -- , "Torch.Indef.Dynamic.Tensor.Math.Lapack"
               -- , "Torch.Indef.Dynamic.Tensor.Math.Pointwise.Floating"
               -- , "Torch.Indef.Dynamic.Tensor.Math.Reduce.Floating"

               -- -- ==================================================== --
               -- -- Static Tensor modules
               -- -- ==================================================== --
               -- , "Torch.Indef.Static.Tensor"
               -- , "Torch.Indef.Static.Tensor.Copy"
               -- , "Torch.Indef.Static.Tensor.Index"
               -- , "Torch.Indef.Static.Tensor.Masked"
               -- , "Torch.Indef.Static.Tensor.Math"
               -- , "Torch.Indef.Static.Tensor.Math.Compare"
               -- , "Torch.Indef.Static.Tensor.Math.CompareT"
               -- , "Torch.Indef.Static.Tensor.Math.Pairwise"
               -- , "Torch.Indef.Static.Tensor.Math.Pointwise"
               -- , "Torch.Indef.Static.Tensor.Math.Random.TH"
               -- , "Torch.Indef.Static.Tensor.Math.Reduce"
               -- , "Torch.Indef.Static.Tensor.Math.Scan"
               -- , "Torch.Indef.Static.Tensor.Random.TH"
               -- , "Torch.Indef.Static.Tensor.Random.THC"
               -- , "Torch.Indef.Static.Tensor.Mode"
               -- , "Torch.Indef.Static.Tensor.ScatterGather"
               -- , "Torch.Indef.Static.Tensor.Sort"
               -- , "Torch.Indef.Static.Tensor.TopK"

               -- , "Torch.Indef.Static.Tensor.Math.Pointwise.Signed"

               -- , "Torch.Indef.Static.NN"
               -- , "Torch.Indef.Static.NN.Activation"
               -- , "Torch.Indef.Static.NN.Backprop"
               -- , "Torch.Indef.Static.NN.Conv1d"
               -- , "Torch.Indef.Static.NN.Conv2d"
               -- , "Torch.Indef.Static.NN.Conv3d"
               -- , "Torch.Indef.Static.NN.Criterion"
               -- , "Torch.Indef.Static.NN.Layers"
               -- , "Torch.Indef.Static.NN.Linear"
               -- , "Torch.Indef.Static.NN.Math"
               -- , "Torch.Indef.Static.NN.Padding"
               -- , "Torch.Indef.Static.NN.Pooling"
               -- , "Torch.Indef.Static.NN.Sampling"
               -- , "Torch.Indef.Static.Tensor.Math.Blas"
               -- , "Torch.Indef.Static.Tensor.Math.Floating"
               -- , "Torch.Indef.Static.Tensor.Math.Lapack"
               -- , "Torch.Indef.Static.Tensor.Math.Pointwise.Floating"
               -- , "Torch.Indef.Static.Tensor.Math.Reduce.Floating"
               -- ]

                }
          ] : Optional (types.Config → types.Library)
      }
