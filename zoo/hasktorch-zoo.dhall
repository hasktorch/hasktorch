   let prelude = ../dhall/dhall-to-cabal/dhall/prelude.dhall sha256:01509b3c6e9eaae4150a6e0ced124c2db191bf6046534a9d4973b7f05afd1d0a
in let types   = ../dhall/dhall-to-cabal/dhall/types.dhall sha256:cfd7597246781e8d4c6dfa5f0eabba75f14dc3f3deb7527973909b37c93f42f5
in let common  = ../dhall/common.dhall
in let packages = common.packages
in let cabalvars = common.cabalvars
in let partials = ../dhall/backpack/partials.dhall
in let mixins = ../dhall/backpack/mixins.dhall
in let provides = ../dhall/backpack/provides.dhall
in let sublibraries = ../dhall/backpack/sublibraries.dhall
in let fn = ../dhall/common/functions.dhall
in common.Package //
  { name = "hasktorch-zoo"
  , flags = [ common.flags.cuda, common.flags.gd ]
  , synopsis = "Neural architectures in hasktorch"
  , description = "Neural architectures, data loading packages, initializations, and common tensor abstractions in hasktorch."
  , library =
    [ \( config : types.Config )
      -> common.Library config //
        { hs-source-dirs = [ "src" ]
        , build-depends =
          [ packages.base
          , packages.backprop
          , packages.dimensions
          , packages.hashable
          , packages.hasktorch
          , packages.microlens
          , packages.singletons
          , packages.generic-lens
          , packages.ghc-typelits-natnormalise
          , packages.vector

          -- data loader dependencies
          , packages.directory
          , packages.filepath
          , packages.deepseq
          , packages.mwc-random
          , packages.primitive
          , packages.safe-exceptions

          -- training iterator dependencies
          , packages.mtl
          , packages.transformers
          , if config.flag "gd" then packages.gd else packages.JuicyPixels
          ]
        , exposed-modules =
          [ "Torch.Data.Loaders.Internal"
          , "Torch.Data.Loaders.RGBVector"
          , "Torch.Data.Loaders.Cifar10"
          -- , "Torch.Data.Loaders.MNIST"
          , "Torch.Data.Loaders.Logging"

          , "Torch.Data.Metrics"
          , "Torch.Data.OneHot"
          , "Torch.Models.Vision.LeNet"
          , "Torch.Initialization"
          ]
        , cpp-options
          = (if config.flag "cuda" then [ "-DCUDA" ] else [] : List Text)
          # (if config.flag "gd" then [ "-DUSE_GD" ] else [] : List Text)
        }
    ] : Optional (types.Config → types.Library)
  }

-- benchmark bench
--   type: exitcode-stdio-1.0
--   hs-source-dirs: bench
--   default-language: Haskell2010
--   main-is: Main.hs
--   build-depends:
--       base
--     , hasktorch-zoo
--     , hasktorch
--     , random-shuffle
--     , transformers
--     , list-t
--     , criterion
--     , vector
--     , primitive
--     , JuicyPixels
--   other-modules:
--     ImageLoading
--   if flag(cuda)
--     cpp-options: -DCUDA
