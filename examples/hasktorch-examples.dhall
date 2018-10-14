   let prelude = ../dhall/dhall-to-cabal/dhall/prelude.dhall
in let types   = ../dhall/dhall-to-cabal/dhall/types.dhall
in let common  = ../dhall/common.dhall
in let fn      = ../dhall/common/functions.dhall
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
    , build-depends
      = base-depends #
      [ packages.backprop
      , packages.hasktorch-zoo
      , packages.unordered-containers

      , packages.dlist
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
          , packages.hasktorch-raw-thc
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
          , other-modules = [ "Utils", "LeNet.Forward" ]
          } }

    , { name = "lenet-cifar10-trainonebatch"
      , executable =
        \(config : types.Config)
        -> lenet-experiment config //
          { main-is = "TrainOneBatch.hs"
          , other-modules = [ "Utils", "LeNet.Forward" ]
          } }
    ]
  }

