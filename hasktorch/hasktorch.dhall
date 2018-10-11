   let prelude  = ../dhall/dhall-to-cabal/dhall/prelude.dhall
in let types    = ../dhall/dhall-to-cabal/dhall/types.dhall
in let List/map = ../dhall/Prelude/List/map
in let common = ../dhall/common.dhall
in let ReexportedModule = ../dhall/common/types/ReexportedModule.dhall
in let packages = common.packages
in let cabalvars = common.cabalvars
in let partials = ../dhall/backpack/partials.dhall
in let mixins = ../dhall/backpack/mixins.dhall
in let provides = ../dhall/backpack/provides.dhall
in let sublibraries = ../dhall/backpack/sublibraries.dhall
in let fn = ../dhall/common/functions.dhall
in let baseexports =
    λ(isth : Bool)  →
    λ(ttype : Text) →
      -- List/map Text ReexportedModule fn.renameNoop
        [ "Torch."++(if isth then "" else "Cuda.")++"${ttype}"
        , "Torch."++(if isth then "" else "Cuda.")++"${ttype}.Dynamic"
        , "Torch."++(if isth then "" else "Cuda.")++"${ttype}.Storage"
        ]
in let nnexports =
    λ(isth : Bool)  →
    λ(ttype : Text) →
    let namespace = if isth then "${ttype}" else "Cuda.${ttype}"
    in -- List/map Text ReexportedModule fn.renameNoop
      [ "Torch.${namespace}.NN"
      , "Torch.${namespace}.NN.Activation"
      , "Torch.${namespace}.NN.Backprop"
      , "Torch.${namespace}.NN.Conv1d"
      , "Torch.${namespace}.NN.Conv2d"
      , "Torch.${namespace}.NN.Criterion"
      , "Torch.${namespace}.NN.Layers"
      , "Torch.${namespace}.NN.Linear"
      , "Torch.${namespace}.NN.Math"
      , "Torch.${namespace}.NN.Padding"
      , "Torch.${namespace}.NN.Pooling"
      , "Torch.${namespace}.NN.Sampling"

      , "Torch.${namespace}.Dynamic.NN"
      , "Torch.${namespace}.Dynamic.NN.Activation"
      , "Torch.${namespace}.Dynamic.NN.Pooling"
      , "Torch.${namespace}.Dynamic.NN.Criterion"
      ]
in let cpu-lite-depends =
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
  ]
in let gpu-lite-depends =
  cpu-lite-depends #
  [ packages.hasktorch-raw-thc
  , packages.hasktorch-types-thc
  ]

in let lite-exposed =
  λ(isth : Bool)
  → baseexports isth "Long"
  # baseexports isth "Double"

in let lite-mixins =
  λ(isth : Bool)
  → [ { package = "hasktorch-indef-signed"
      , renaming =
        { provides = prelude.types.ModuleRenaming.renaming (provides.signed isth "Long")
        , requires = prelude.types.ModuleRenaming.renaming (mixins.signed isth "Long")
        } }
      , { package = "hasktorch-indef-floating"
        , renaming =
          { provides = prelude.types.ModuleRenaming.renaming (provides.floating isth "Double")
          , requires = prelude.types.ModuleRenaming.renaming (mixins.floating isth "Double")
        } } ]

in let full-mixins =
  λ(isth : Bool)
  → lite-mixins isth
  # [ { package = "hasktorch-indef-unsigned"
      , renaming =
        { provides = prelude.types.ModuleRenaming.renaming (provides.unsigned isth "Byte")
        , requires = prelude.types.ModuleRenaming.renaming (mixins.unsigned isth "Byte")
        } }
    , { package = "hasktorch-indef-unsigned"
      , renaming =
        { provides = prelude.types.ModuleRenaming.renaming (provides.unsigned isth "Char")
        , requires = prelude.types.ModuleRenaming.renaming (mixins.unsigned isth "Char")
        } }
    , { package = "hasktorch-indef-signed"
      , renaming =
        { provides = prelude.types.ModuleRenaming.renaming (provides.signed isth "Short")
        , requires = prelude.types.ModuleRenaming.renaming (mixins.signed isth "Short")
        } }
    , { package = "hasktorch-indef-signed"
      , renaming =
        { provides = prelude.types.ModuleRenaming.renaming (provides.signed isth "Int")
        , requires = prelude.types.ModuleRenaming.renaming (mixins.signed isth "Int")
        } }
    ]
  # (if isth then
    [ { package = "hasktorch-indef-floating"
      , renaming =
        { provides = prelude.types.ModuleRenaming.renaming (provides.floating isth "Float")
        , requires = prelude.types.ModuleRenaming.renaming (mixins.floating isth "Float")
        } }
    ] else [] : List types.Mixin)

in let backend-sublibrary =
  λ(isth : Bool)
  → { name = "hasktorch-${if isth then "cpu" else "gpu"}"
    , library =
      λ (config : types.Config)
      →  let lite-depends = if isth then cpu-lite-depends else gpu-lite-depends
      in let floatless-exports =
         ( baseexports isth "Long"
         # baseexports isth "Double"
         ) # (
          if config.flag "lite"
          then [] : List Text
          else
           ( baseexports isth "Byte"
           # baseexports isth "Char"
           # baseexports isth "Short"
           # baseexports isth "Int"
           ) )
      in common.Library //
        { hs-source-dirs = [ "utils", "src" ]
        , build-depends = if config.flag "lite" then lite-depends else lite-depends # [packages.hasktorch-indef-unsigned]
        , reexported-modules = List/map Text ReexportedModule fn.renameNoop (nnexports isth "Double")
        , exposed-modules = floatless-exports # (if isth then baseexports isth "Float" # nnexports isth "Float" else [] : List Text)
        , cpp-options = if isth then [] : List Text else [ "-DCUDA", "-DHASKTORCH_INTERNAL_CUDA" ]
        , mixins = if config.flag "lite" then lite-mixins isth else full-mixins isth
        }
     }
in let mkdefinite-exe
  = \(deftype : Text)
  -> { name = ("isdefinite" ++ deftype)
     , executable =
       \(config : types.Config)
       ->  prelude.defaults.Executable //
         { main-is = "Noop.hs"
         , build-depends =
           [ packages.base
           , fn.anyver ("hasktorch" ++ deftype)
           ]
           , default-language = cabalvars.default-language
           , hs-source-dirs = [ "exe" ]
           }
     }
in common.Package
   // { name = "hasktorch"
      , flags = [ common.flags.cuda, common.flags.lite ]
      , description = "core tensor abstractions wrapping raw TH bindings"
      , synopsis = "Torch for tensors and neural networks in Haskell"
      , executables =
          [ mkdefinite-exe "-cpu"
          , mkdefinite-exe "-gpu"
          , mkdefinite-exe ""
          , { name = "memcheck"
            , executable =
              λ(config : types.Config)
              → prelude.defaults.Executable
               // { main-is = "Memcheck.hs"
                  , build-depends =
                      [ packages.base
                      , packages.hasktorch
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
                      , packages.hasktorch
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
               ([ "Torch.Types.Numeric" ]
                # baseexports True "Long"
                # baseexports True "Double"
                # nnexports   True "Double"
               )
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
          in let cpu-lite-reexports = List/map Text ReexportedModule fn.renameNoop cpu-lite-exports
          in let cpu-full-reexports = List/map Text ReexportedModule fn.renameNoop cpu-full-exports
          in let gpu-lite-reexports = List/map Text ReexportedModule fn.renameNoop gpu-lite-exports
          in let gpu-full-reexports = List/map Text ReexportedModule fn.renameNoop gpu-full-exports
          in common.Library
          // { hs-source-dirs = [ "utils" ]
            , build-depends =
              [ packages.base
              , packages.containers
              , packages.deepseq
              , packages.dimensions
              , packages.managed
              , packages.microlens
              , packages.numeric-limits
              , packages.safe-exceptions
              , packages.singletons
              , packages.text
              , packages.typelits-witnesses

              , fn.anyver "hasktorch-cpu"
              , fn.anyver "hasktorch-gpu"
              , packages.hasktorch-raw-th
              , packages.hasktorch-types-th
              ]
            , exposed-modules =
              [ "Torch.Core.Exceptions"
              , "Torch.Core.Random"
              , "Torch.Core.LogAdd"
              ]
            , reexported-modules =
              if               config.flag "lite" && False == config.flag "cuda" then cpu-lite-reexports
              else if False == config.flag "lite" && False == config.flag "cuda" then cpu-full-reexports
              else if          config.flag "lite" &&          config.flag "cuda" then gpu-lite-reexports
              else
                gpu-full-reexports
            }
        ] : Optional (types.Config → types.Library)

      , sub-libraries =
        [ backend-sublibrary True
        , backend-sublibrary False
        , sublibraries.unsigned
        , sublibraries.signed
        , sublibraries.floating
        ]
    }

