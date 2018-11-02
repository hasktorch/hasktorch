   let prelude = ./dhall-to-cabal/dhall/prelude.dhall sha256:01509b3c6e9eaae4150a6e0ced124c2db191bf6046534a9d4973b7f05afd1d0a
in let types = ./dhall-to-cabal/dhall/types.dhall sha256:cfd7597246781e8d4c6dfa5f0eabba75f14dc3f3deb7527973909b37c93f42f5
in let fn = ./common/functions.dhall sha256:45e8bee44c93da6f4c47a3fdacc558b00858461325b807d4afc8bf0965716c33
in let common  = ../dhall/common.dhall
in let partials     = ../dhall/backpack/partials.dhall
in let mixins       = ../dhall/backpack/mixins.dhall
in let provides     = ../dhall/backpack/provides.dhall
in let sublibraries = ../dhall/backpack/sublibraries.dhall
in let packages = common.packages
in let cabalvars = common.cabalvars

in let base-depends =
  [ packages.base
  , packages.hasktorch
  , packages.microlens
  , packages.dimensions
  , packages.safe-exceptions
  , packages.singletons
  ]

in let exe-builder
  =  \(experimental : Bool)
  -> \(cancuda : Bool)
  -> \(name : Text)
  -> \(main : Text)
  -> \(extradeps : List types.Dependency)
  -> { name = name
     , executable =
       \(config : types.Config)
       ->  prelude.defaults.Executable //
         { main-is = main
         , default-language = cabalvars.default-language
         , hs-source-dirs = [ (if experimental then ("experimental/" ++ name) else name) ]
         , cpp-options = if (cancuda && config.flag "cuda") then [ "-DCUDA" ] else [] : List Text
         , default-extensions =
           if config.impl (prelude.types.Compilers.GHC {=}) (prelude.orLaterVersion (prelude.v "8.6"))
           then [prelude.types.Extensions.StarIsType False]
           else [] : List types.Extensions
         , build-depends
           = base-depends
           # extradeps
           # (if (cancuda && config.flag "cuda") then [ packages.cuda ] else [] : List types.Dependency)
         }
     }

in let exe-defaults
  =  \(experimental : Bool)
  -> \(cancuda : Bool)
  -> \(name : Text)
  -> \(main : Text)
  -> exe-builder experimental cancuda name main ([] : List types.Dependency)

in let lenet-experiment =
  \(config : types.Config)
  ->  prelude.defaults.Executable //
    { default-language = cabalvars.default-language
    , hs-source-dirs = [ "lenet-cifar10" ]
    , cpp-options
      = (if config.flag "cuda"  then [ "-DCUDA"  ] else [] : List Text)
      # (if config.flag "debug" then [ "-DDEBUG" ] else [] : List Text)
    , default-extensions =
      if config.impl (prelude.types.Compilers.GHC {=}) (prelude.orLaterVersion (prelude.v "8.6"))
      then [prelude.types.Extensions.StarIsType False]
      else [] : List types.Extensions
    , build-depends
      = base-depends #
      [ packages.backprop
      , packages.hasktorch-zoo
      , packages.unordered-containers

      , packages.dlist
      , packages.ghc-typelits-natnormalise
      , packages.list-t
      , packages.mtl
      , packages.monad-loops
      , packages.mwc-random
      , packages.vector
      , packages.time
      , packages.transformers
      , packages.safe-exceptions
      ] # (
        if config.flag "cuda" == False
        then [] : List types.Dependency
        else
          [ packages.cuda
          , packages.hasktorch-ffi-thc
          , packages.hasktorch-types-thc
          ])

    }

in common.Package //
  { name = "hasktorch-examples"
  , synopsis = "Example usage"
  , description = "Example usage"
  , flags = [ common.flags.cuda, common.flags.debug ]
  , executables =
    [ exe-defaults False False "gradient-descent"    "GradientDescent.hs"
    , exe-defaults False False "multivariate-normal" "MultivariateNormal.hs"
    , exe-defaults True  False "lasso"               "Lasso.hs"
    , exe-defaults True  False "ad"                  "AD.hs"
    , exe-defaults True  True  "bayesian-regression" "BayesianRegression.hs"
    , exe-builder  True  False "download-mnist"      "DownloadMNIST.hs"
      [ packages.HTTP
      , packages.bytestring
      , packages.cryptonite
      , packages.directory
      , packages.filepath
      , packages.network-uri
      ]
    , exe-defaults True   False "ff-typed"            "FeedForwardTyped.hs"
    , exe-defaults True   False "ff-untyped"          "FeedForwardUntyped.hs"
    , exe-defaults False  True  "static-tensor-usage" "Main.hs"
    , { name = "lenet-cifar10"
      , executable =
        \(config : types.Config)
        -> lenet-experiment config //
          { main-is = "Manual.hs"
          , other-modules =
            [ "DataLoading"
            , "Criterion"
            , "LeNet"
            ]
          } }

    , { name = "lenet-cifar10-initialization"
      , executable =
        \(config : types.Config)
        -> lenet-experiment config //
          { main-is = "Initialization.hs"
          , other-modules = [ "Utils" ]
          } }

    , { name = "lenet-cifar10-trainonebatch"
      , executable =
        \(config : types.Config)
        -> lenet-experiment config //
          { main-is = "TrainOneBatch.hs"
          , other-modules = [ "Utils" ]
          } }

    , { name = "dense3-cifar10"
      , executable =
        \(config : types.Config)
        -> lenet-experiment config //
          { main-is = "Dense3CIFAR.hs"
          , hs-source-dirs = [ "lenet-cifar10" ]
          , other-modules = [ "Utils", "Dense3" ]
          } }

    , { name = "xor"
      , executable =
        \(config : types.Config)
        -> lenet-experiment config //
          { main-is = "Main.hs"
          , hs-source-dirs = [ "lenet-cifar10", "xor" ]
          , other-modules = [ "Utils", "Dense3" ]
          } }
    ]
  }

