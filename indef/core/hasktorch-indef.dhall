   let prelude = ../dhall/dhall-to-cabal/dhall/prelude.dhall sha256:01509b3c6e9eaae4150a6e0ced124c2db191bf6046534a9d4973b7f05afd1d0a
in let types = ../dhall/dhall-to-cabal/dhall/types.dhall sha256:cfd7597246781e8d4c6dfa5f0eabba75f14dc3f3deb7527973909b37c93f42f5
in let fn = ../dhall/common/functions.dhall sha256:45e8bee44c93da6f4c47a3fdacc558b00858461325b807d4afc8bf0965716c33
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
                      , packages.backprop
                      , packages.base
                      , packages.dimensions
                      , packages.ghc-typelits-natnormalise
                      , fn.anyver "hasktorch-indef-floating"
                      , packages.hasktorch-ffi-th
                      , packages.hasktorch-types-th
                      , packages.hspec
                      , packages.mtl
                      , packages.singletons
                      , packages.text
                      , packages.transformers
                      ]
                    , default-extensions = cabalvars.default-extensions config
                    , default-language = cabalvars.default-language
                    , hs-source-dirs = [ "tests" ]
                    , other-modules =
                      [ "Torch.Indef.StorageSpec"
                      , "Torch.Indef.Dynamic.TensorSpec"
                      , "Torch.Indef.Static.TensorSpec"
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
          , packages.vector
          ]
        in common.Library config //
          { hs-source-dirs = [ "src" ]
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
          , default-extensions = cabalvars.default-extensions config
          , exposed-modules = fn.getnames sub-libraries.allindef-reexports
          }
        ] : Optional (types.Config → types.Library)
      }
