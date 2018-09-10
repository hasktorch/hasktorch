   let prelude = https://raw.githubusercontent.com/dhall-lang/dhall-to-cabal/master/dhall/prelude.dhall
in let types = https://raw.githubusercontent.com/dhall-lang/dhall-to-cabal/master/dhall/types.dhall
in let common = ../../dhall/common.dhall

in common.Package
   // { name = "hasktorch-raw-tests"
      , description = "core test libraries"
      , synopsis = "Testing library for raw TH and THC bindings"
      , library =
          [   λ(config : types.Config)
            →   prelude.defaults.Library
             // { build-depends =
                    [ common.packages.base
                    , common.packages.text
                    , common.packages.hasktorch-types-th
                    , common.packages.hspec
                    , common.packages.QuickCheck
                    ]
                , default-language = common.cabalvars.default-language
                , hs-source-dirs = [ "src" ]
                , other-modules = [ "Paths_hasktorch_raw_tests" ]
                , exposed-modules =
                    [ "Torch.FFI.Tests", "Torch.FFI.TestsNN" ]
                }
          ] : Optional (types.Config → types.Library)
      }
