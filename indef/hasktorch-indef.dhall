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
      , synopsis = "Core Hasktorch abstractions wrapping FFI bindings"
      , description
        = "The hasktorch-indef package constitutes the main user API for hasktorch. It uses backpack signatures to generically glue low-level FFI bindings to a high-level interface."
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
                      , packages.hasktorch-ffi-th
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
          , packages.hasktorch-ffi-th
          , packages.managed
          , packages.mtl
          , packages.safe-exceptions
          , packages.singletons
          , packages.ghc-typelits-natnormalise -- replace with thorin?
          , packages.transformers
          , packages.text
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
             }
          ] : Optional (types.Config → types.Library)
      }
