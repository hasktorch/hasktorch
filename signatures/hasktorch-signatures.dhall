    let prelude = https://raw.githubusercontent.com/dhall-lang/dhall-to-cabal/master/dhall/prelude.dhall
in  let types = https://raw.githubusercontent.com/dhall-lang/dhall-to-cabal/master/dhall/types.dhall
in  let common = ../dhall/common.dhall
in  let packages = common.packages
in  let cabalvars = common.cabalvars
in  let fn = ../dhall/common/functions.dhall
in  let subLibrary = common.Library
        // { build-depends =
             [ packages.base
             , packages.hasktorch-partial
             , packages.hasktorch-signatures
             ]
           }
in  let exeScaffold =
    λ(sigset : Text) →
    λ(thtype : Text) →
      prelude.defaults.Executable
          // { main-is = "Main.hs"
             , default-language = cabalvars.default-language
             , hs-source-dirs = [ "exe" ]
             , build-depends =
               [ packages.base
               , fn.anyver ("hasktorch-raw-" ++ thtype)
               , fn.anyver ("hasktorch-types-" ++ thtype)
               , fn.anyver ("hasktorch-partial-" ++ sigset)
               ]
             }
in   let unsignedTHMixins =
      λ(ttype : Text)
       → prelude.types.ModuleRenaming.renaming
        [ { rename = "Torch.Sig.State"                 , to = "Torch.Types.TH" }
        , { rename = "Torch.Sig.Types.Global"          , to = "Torch.Types.TH" }
        , { rename = "Torch.Sig.Types"                 , to = "Torch.Types.TH." ++ ttype }
        , { rename = "Torch.Sig.Storage"               , to = "Torch.FFI.TH." ++ ttype ++ ".Storage" }
        , { rename = "Torch.Sig.Storage.Copy"          , to = "Torch.FFI.TH." ++ ttype ++ ".StorageCopy" }
        , { rename = "Torch.Sig.Storage.Memory"        , to = "Torch.FFI.TH." ++ ttype ++ ".FreeStorage" }
        , { rename = "Torch.Sig.Tensor"                , to = "Torch.FFI.TH." ++ ttype ++ ".Tensor" }
        , { rename = "Torch.Sig.Tensor.Copy"           , to = "Torch.FFI.TH." ++ ttype ++ ".TensorCopy" }
        , { rename = "Torch.Sig.Tensor.Memory"         , to = "Torch.FFI.TH." ++ ttype ++ ".FreeTensor" }
        , { rename = "Torch.Sig.Tensor.Index"          , to = "Torch.FFI.TH." ++ ttype ++ ".TensorMath" }
        , { rename = "Torch.Sig.Tensor.Masked"         , to = "Torch.FFI.TH." ++ ttype ++ ".TensorMath" }
        , { rename = "Torch.Sig.Tensor.Math"           , to = "Torch.FFI.TH." ++ ttype ++ ".TensorMath" }
        , { rename = "Torch.Sig.Tensor.Math.Compare"   , to = "Torch.FFI.TH." ++ ttype ++ ".TensorMath" }
        , { rename = "Torch.Sig.Tensor.Math.CompareT"  , to = "Torch.FFI.TH." ++ ttype ++ ".TensorMath" }
        , { rename = "Torch.Sig.Tensor.Math.Pairwise"  , to = "Torch.FFI.TH." ++ ttype ++ ".TensorMath" }
        , { rename = "Torch.Sig.Tensor.Math.Pointwise" , to = "Torch.FFI.TH." ++ ttype ++ ".TensorMath" }
        , { rename = "Torch.Sig.Tensor.Math.Reduce"    , to = "Torch.FFI.TH." ++ ttype ++ ".TensorMath" }
        , { rename = "Torch.Sig.Tensor.Math.Scan"      , to = "Torch.FFI.TH." ++ ttype ++ ".TensorMath" }
        , { rename = "Torch.Sig.Tensor.Mode"           , to = "Torch.FFI.TH." ++ ttype ++ ".TensorMath" }
        , { rename = "Torch.Sig.Tensor.ScatterGather"  , to = "Torch.FFI.TH." ++ ttype ++ ".TensorMath" }
        , { rename = "Torch.Sig.Tensor.Sort"           , to = "Torch.FFI.TH." ++ ttype ++ ".TensorMath" }
        , { rename = "Torch.Sig.Tensor.TopK"           , to = "Torch.FFI.TH." ++ ttype ++ ".TensorMath" }
        ]
in   let unsignedTHCMixins =
      λ(ttype : Text)
       → prelude.types.ModuleRenaming.renaming
        [ { rename = "Torch.Sig.State"                 , to = "Torch.FFI.THC.State" }
        , { rename = "Torch.Sig.Types.Global"          , to = "Torch.Types.THC" }
        , { rename = "Torch.Sig.Types"                 , to = "Torch.Types.THC." ++ ttype }
        , { rename = "Torch.Sig.Storage"               , to = "Torch.FFI.THC." ++ ttype ++ ".Storage" }
        , { rename = "Torch.Sig.Storage.Copy"          , to = "Torch.FFI.THC." ++ ttype ++ ".StorageCopy" }
        , { rename = "Torch.Sig.Storage.Memory"        , to = "Torch.FFI.THC." ++ ttype ++ ".Storage" }
        , { rename = "Torch.Sig.Tensor"                , to = "Torch.FFI.THC." ++ ttype ++ ".Tensor" }
        , { rename = "Torch.Sig.Tensor.Copy"           , to = "Torch.FFI.THC." ++ ttype ++ ".TensorCopy" }
        , { rename = "Torch.Sig.Tensor.Memory"         , to = "Torch.FFI.THC." ++ ttype ++ ".Tensor" }
        , { rename = "Torch.Sig.Tensor.Index"          , to = "Torch.FFI.THC." ++ ttype ++ ".TensorIndex" }
        , { rename = "Torch.Sig.Tensor.Masked"         , to = "Torch.FFI.THC." ++ ttype ++ ".TensorMasked" }
        , { rename = "Torch.Sig.Tensor.Math"           , to = "Torch.FFI.THC." ++ ttype ++ ".TensorMath" }
        , { rename = "Torch.Sig.Tensor.Math.Compare"   , to = "Torch.FFI.THC." ++ ttype ++ ".TensorMathCompare" }
        , { rename = "Torch.Sig.Tensor.Math.CompareT"  , to = "Torch.FFI.THC." ++ ttype ++ ".TensorMathCompareT" }
        , { rename = "Torch.Sig.Tensor.Math.Pairwise"  , to = "Torch.FFI.THC." ++ ttype ++ ".TensorMathPairwise" }
        , { rename = "Torch.Sig.Tensor.Math.Pointwise" , to = "Torch.FFI.THC." ++ ttype ++ ".TensorMathPointwise" }
        , { rename = "Torch.Sig.Tensor.Math.Reduce"    , to = "Torch.FFI.THC." ++ ttype ++ ".TensorMathReduce" }
        , { rename = "Torch.Sig.Tensor.Math.Scan"      , to = "Torch.FFI.THC." ++ ttype ++ ".TensorMathScan" }
        , { rename = "Torch.Sig.Tensor.Mode"           , to = "Torch.FFI.THC." ++ ttype ++ ".TensorMode" }
        , { rename = "Torch.Sig.Tensor.ScatterGather"  , to = "Torch.FFI.THC." ++ ttype ++ ".TensorScatterGather" }
        , { rename = "Torch.Sig.Tensor.Sort"           , to = "Torch.FFI.THC." ++ ttype ++ ".TensorSort" }
        , { rename = "Torch.Sig.Tensor.TopK"           , to = "Torch.FFI.THC." ++ ttype ++ ".TensorTopK" }
        ]

in   let signedTHMixins =
      λ(ttype : Text)
       → prelude.types.ModuleRenaming.renaming
         [ { rename = "Torch.Sig.State"                        , to = "Torch.Types.TH" }
         , { rename = "Torch.Sig.Types.Global"                 , to = "Torch.Types.TH" }
         , { rename = "Torch.Sig.Types"                        , to = "Torch.Types.TH." ++ ttype }
         , { rename = "Torch.Sig.Storage"                      , to = "Torch.FFI.TH."++ttype++".Storage" }
         , { rename = "Torch.Sig.Storage.Copy"                 , to = "Torch.FFI.TH."++ttype++".StorageCopy" }
         , { rename = "Torch.Sig.Storage.Memory"               , to = "Torch.FFI.TH."++ttype++".FreeStorage" }
         , { rename = "Torch.Sig.Tensor"                       , to = "Torch.FFI.TH."++ttype++".Tensor" }
         , { rename = "Torch.Sig.Tensor.Copy"                  , to = "Torch.FFI.TH."++ttype++".TensorCopy" }
         , { rename = "Torch.Sig.Tensor.Memory"                , to = "Torch.FFI.TH."++ttype++".FreeTensor" }
         , { rename = "Torch.Sig.Tensor.Index"                 , to = "Torch.FFI.TH."++ttype++".TensorMath" }
         , { rename = "Torch.Sig.Tensor.Masked"                , to = "Torch.FFI.TH."++ttype++".TensorMath" }
         , { rename = "Torch.Sig.Tensor.Math"                  , to = "Torch.FFI.TH."++ttype++".TensorMath" }
         , { rename = "Torch.Sig.Tensor.Math.Compare"          , to = "Torch.FFI.TH."++ttype++".TensorMath" }
         , { rename = "Torch.Sig.Tensor.Math.CompareT"         , to = "Torch.FFI.TH."++ttype++".TensorMath" }
         , { rename = "Torch.Sig.Tensor.Math.Pairwise"         , to = "Torch.FFI.TH."++ttype++".TensorMath" }
         , { rename = "Torch.Sig.Tensor.Math.Pointwise"        , to = "Torch.FFI.TH."++ttype++".TensorMath" }
         , { rename = "Torch.Sig.Tensor.Math.Reduce"           , to = "Torch.FFI.TH."++ttype++".TensorMath" }
         , { rename = "Torch.Sig.Tensor.Math.Scan"             , to = "Torch.FFI.TH."++ttype++".TensorMath" }
         , { rename = "Torch.Sig.Tensor.Mode"                  , to = "Torch.FFI.TH."++ttype++".TensorMath" }
         , { rename = "Torch.Sig.Tensor.ScatterGather"         , to = "Torch.FFI.TH."++ttype++".TensorMath" }
         , { rename = "Torch.Sig.Tensor.Sort"                  , to = "Torch.FFI.TH."++ttype++".TensorMath" }
         , { rename = "Torch.Sig.Tensor.TopK"                  , to = "Torch.FFI.TH."++ttype++".TensorMath" }
         , { rename = "Torch.Sig.Tensor.Math.Pointwise.Signed" , to = "Torch.FFI.TH."++ttype++".TensorMath" }
         ]

in   let signedTHCMixins =
      λ(ttype : Text)
       → prelude.types.ModuleRenaming.renaming
         [ { rename = "Torch.Sig.Types.Global"                 , to = "Torch.Types.THC" }
         , { rename = "Torch.Sig.Types"                        , to = "Torch.Types.THC." ++ ttype }
         , { rename = "Torch.Sig.State"                        , to = "Torch.FFI.THC.State" }
         , { rename = "Torch.Sig.Storage"                      , to = "Torch.FFI.THC."++ttype++".Storage" }
         , { rename = "Torch.Sig.Storage.Copy"                 , to = "Torch.FFI.THC."++ttype++".StorageCopy" }
         , { rename = "Torch.Sig.Storage.Memory"               , to = "Torch.FFI.THC."++ttype++".Storage" }
         , { rename = "Torch.Sig.Tensor"                       , to = "Torch.FFI.THC."++ttype++".Tensor" }
         , { rename = "Torch.Sig.Tensor.Copy"                  , to = "Torch.FFI.THC."++ttype++".TensorCopy" }
         , { rename = "Torch.Sig.Tensor.Memory"                , to = "Torch.FFI.THC."++ttype++".Tensor" }
         , { rename = "Torch.Sig.Tensor.Index"                 , to = "Torch.FFI.THC."++ttype++".TensorIndex" }
         , { rename = "Torch.Sig.Tensor.Masked"                , to = "Torch.FFI.THC."++ttype++".TensorMasked" }
         , { rename = "Torch.Sig.Tensor.Math"                  , to = "Torch.FFI.THC."++ttype++".TensorMath" }
         , { rename = "Torch.Sig.Tensor.Math.Compare"          , to = "Torch.FFI.THC."++ttype++".TensorMathCompare" }
         , { rename = "Torch.Sig.Tensor.Math.CompareT"         , to = "Torch.FFI.THC."++ttype++".TensorMathCompareT" }
         , { rename = "Torch.Sig.Tensor.Math.Pairwise"         , to = "Torch.FFI.THC."++ttype++".TensorMathPairwise" }
         , { rename = "Torch.Sig.Tensor.Math.Pointwise"        , to = "Torch.FFI.THC."++ttype++".TensorMathPointwise" }
         , { rename = "Torch.Sig.Tensor.Math.Reduce"           , to = "Torch.FFI.THC."++ttype++".TensorMathReduce" }
         , { rename = "Torch.Sig.Tensor.Math.Scan"             , to = "Torch.FFI.THC."++ttype++".TensorMathScan" }
         , { rename = "Torch.Sig.Tensor.Mode"                  , to = "Torch.FFI.THC."++ttype++".TensorMode" }
         , { rename = "Torch.Sig.Tensor.ScatterGather"         , to = "Torch.FFI.THC."++ttype++".TensorScatterGather" }
         , { rename = "Torch.Sig.Tensor.Sort"                  , to = "Torch.FFI.THC."++ttype++".TensorSort" }
         , { rename = "Torch.Sig.Tensor.TopK"                  , to = "Torch.FFI.THC."++ttype++".TensorTopK" }
         , { rename = "Torch.Sig.Tensor.Math.Pointwise.Signed" , to = "Torch.FFI.THC."++ttype++".TensorMathPointwise" }
         ]

in  let
  sub-lib-hasktorch-partial-floating =
    { name = "hasktorch-partial-floating"
    , library =
      λ(config : types.Config)
       → subLibrary
        // { reexported-modules =
             [ fn.renameNoop "Torch.Undefined.Tensor.Math.Random.TH"
             , fn.renameNoop "Torch.Undefined.Tensor.Random.TH"
             , fn.renameNoop "Torch.Undefined.Tensor.Random.THC"
             ]
           } }
in  let
  hasktorch-partial-signed-mixins =
    [ fn.renameSig "Undefined" "NN"
    , fn.renameSig "Undefined" "Types.NN"
    , fn.renameSig "Undefined" "Tensor.Math.Blas"
    , fn.renameSig "Undefined" "Tensor.Math.Floating"
    , fn.renameSig "Undefined" "Tensor.Math.Lapack"
    , fn.renameSig "Undefined" "Tensor.Math.Pointwise.Floating"
    , fn.renameSig "Undefined" "Tensor.Math.Reduce.Floating"
    , fn.renameSig "Undefined" "Tensor.Math.Random.TH"
    , fn.renameSig "Undefined" "Tensor.Random.TH"
    , fn.renameSig "Undefined" "Tensor.Random.THC"
    ]

in  let
  sub-lib-hasktorch-partial-signed =
    { name = "hasktorch-partial-signed"
    , library =
      λ(config : types.Config)
       → subLibrary
        // { mixins =
             [ { package = "hasktorch-signatures"
               , renaming =
                 { provides = prelude.types.ModuleRenaming.default {=}
                 , requires = fn.mixinRequirements
                   [ fn.renameSig "Undefined" "NN"
                   , fn.renameSig "Undefined" "Types.NN"
                   , fn.renameSig "Undefined" "Tensor.Math.Blas"
                   , fn.renameSig "Undefined" "Tensor.Math.Floating"
                   , fn.renameSig "Undefined" "Tensor.Math.Lapack"
                   , fn.renameSig "Undefined" "Tensor.Math.Pointwise.Floating"
                   , fn.renameSig "Undefined" "Tensor.Math.Reduce.Floating"
                   , fn.renameSig "Undefined" "Tensor.Math.Random.TH"
                   , fn.renameSig "Undefined" "Tensor.Random.TH"
                   , fn.renameSig "Undefined" "Tensor.Random.THC"
                   ]
                 }
               } ]
           } }

in  let
  sub-lib-hasktorch-partial-unsigned =
    { name = "hasktorch-partial-unsigned"
    , library =
      λ(config : types.Config)
       → subLibrary
        // { mixins =
             [ { package = "hasktorch-signatures"
               , renaming =
                 { provides = prelude.types.ModuleRenaming.default {=}
                 , requires = fn.mixinRequirements
                   [ fn.renameSig "Undefined" "NN"
                   , fn.renameSig "Undefined" "Types.NN"
                   , fn.renameSig "Undefined" "Tensor.Math.Blas"
                   , fn.renameSig "Undefined" "Tensor.Math.Floating"
                   , fn.renameSig "Undefined" "Tensor.Math.Lapack"
                   , fn.renameSig "Undefined" "Tensor.Math.Pointwise.Floating"
                   , fn.renameSig "Undefined" "Tensor.Math.Reduce.Floating"
                   , fn.renameSig "Undefined" "Tensor.Math.Random.TH"
                   , fn.renameSig "Undefined" "Tensor.Random.TH"
                   , fn.renameSig "Undefined" "Tensor.Random.THC"

                   -- List/append to sub-lib-hasktorch-partial-signed requires
                   , fn.renameSig "Undefined" "Tensor.Math.Pointwise.Signed"
                   ]
                 }
               } ]
           } }

in  common.Package
   // { name = "hasktorch-signatures"
      , description = "CFFI backpack signatures"
      , synopsis = "Backpack signature files to pair with hasktorch-raw and hasktorch-core"
      , library =
          [   λ(config : types.Config)
            →   common.Library
              // { hs-source-dirs = [ "src" ]
                 , build-depends =
                    [ packages.base
                    , packages.hasktorch-types-th
                    , packages.hasktorch-types-thc
                    , packages.hasktorch-signatures-types
                    ]
                , signatures =
                    [ "Torch.Sig.NN"
                    , "Torch.Sig.Storage"
                    , "Torch.Sig.Storage.Copy"
                    , "Torch.Sig.Tensor"
                    , "Torch.Sig.Tensor.Copy"
                    , "Torch.Sig.Tensor.Index"
                    , "Torch.Sig.Tensor.Masked"
                    , "Torch.Sig.Tensor.Math"
                    , "Torch.Sig.Tensor.Math.Blas"
                    , "Torch.Sig.Tensor.Math.Compare"
                    , "Torch.Sig.Tensor.Math.CompareT"
                    , "Torch.Sig.Tensor.Math.Floating"
                    , "Torch.Sig.Tensor.Math.Lapack"
                    , "Torch.Sig.Tensor.Math.Pairwise"
                    , "Torch.Sig.Tensor.Math.Pointwise"
                    , "Torch.Sig.Tensor.Math.Pointwise.Floating"
                    , "Torch.Sig.Tensor.Math.Pointwise.Signed"
                    , "Torch.Sig.Tensor.Math.Reduce"
                    , "Torch.Sig.Tensor.Math.Reduce.Floating"
                    , "Torch.Sig.Tensor.Math.Scan"
                    , "Torch.Sig.Tensor.Mode"
                    , "Torch.Sig.Tensor.ScatterGather"
                    , "Torch.Sig.Tensor.Sort"
                    , "Torch.Sig.Tensor.TopK"
                    , "Torch.Sig.Types.NN"
                    , "Torch.Sig.Tensor.Random.TH"
                    , "Torch.Sig.Tensor.Math.Random.TH"
                    , "Torch.Sig.Tensor.Random.THC"
                    ]
                }
          ] : Optional (types.Config → types.Library)

      , sub-libraries =
          [ sub-lib-hasktorch-partial-floating
          , sub-lib-hasktorch-partial-signed
          , sub-lib-hasktorch-partial-unsigned
          ]
      , executables =
          [ { name = "isdefinite-unsigned-th"
            , executable =
              λ(config : types.Config)
              → exeScaffold "unsigned" "th"
               // { mixins =
                   [ { package = "hasktorch-partial-unsigned"
                     , renaming =
                       { provides = prelude.types.ModuleRenaming.default {=}
                       , requires = unsignedTHMixins "Byte"
                       }
                     }
                   , { package = "hasktorch-partial-unsigned"
                     , renaming =
                       { provides = prelude.types.ModuleRenaming.default {=}
                       , requires = unsignedTHMixins "Char"
                       }
                     } ] } }
          , { name = "isdefinite-unsigned-thc"
            , executable =
              λ(config : types.Config) → exeScaffold "unsigned" "thc"
              // { mixins =
                   [ { package = "hasktorch-partial-unsigned"
                     , renaming =
                       { provides = prelude.types.ModuleRenaming.default {=}
                       , requires = unsignedTHCMixins "Byte"
                       }
                     }
                   , { package = "hasktorch-partial-unsigned"
                     , renaming =
                       { provides = prelude.types.ModuleRenaming.default {=}
                       , requires = unsignedTHCMixins "Char"
                       }
                     } ] } }
          , { name = "isdefinite-signed-th"
            , executable =
              λ(config : types.Config) → exeScaffold "signed" "th"
              // { mixins =
                   [ { package = "hasktorch-partial-signed"
                     , renaming =
                         { provides = prelude.types.ModuleRenaming.default {=}
                         , requires = signedTHMixins "Short"
                         }
                     }
                   , { package = "hasktorch-partial-signed"
                     , renaming =
                         { provides = prelude.types.ModuleRenaming.default {=}
                         , requires = signedTHMixins "Int"
                         }
                     }
                   , { package = "hasktorch-partial-signed"
                     , renaming =
                         { provides = prelude.types.ModuleRenaming.default {=}
                         , requires = signedTHMixins "Long"
                         }
                     }
                   ]
                 }
            }
          , { name = "isdefinite-signed-thc"
            , executable =
              λ(config : types.Config) → exeScaffold "signed" "thc"
              // { mixins =
                    [ { package = "hasktorch-partial-signed"
                      , renaming =
                         { provides = prelude.types.ModuleRenaming.default {=}
                         , requires = signedTHCMixins "Short"
                         }
                      }
                    , { package = "hasktorch-partial-signed"
                      , renaming =
                         { provides = prelude.types.ModuleRenaming.default {=}
                         , requires = signedTHCMixins "Int"
                         }
                      }
                    , { package = "hasktorch-partial-signed"
                      , renaming =
                         { provides = prelude.types.ModuleRenaming.default {=}
                         , requires = signedTHCMixins "Long"
                         }
                      }
                    ]
                  }
            }
          , { name = "isdefinite-floating-th"
            , executable =
              λ(config : types.Config) → exeScaffold "floating" "th"
              // { mixins =
                   [ { package = "hasktorch-partial-floating"
                     , renaming =
                         { provides = prelude.types.ModuleRenaming.renaming
                             [ { rename = "Torch.Undefined.Tensor.Random.THC"
                               , to = "Torch.Undefined.Float.Tensor.Random.THC"
                               } ]
                         , requires = prelude.types.ModuleRenaming.renaming
                             [ { rename = "Torch.Sig.State" , to = "Torch.Types.TH" }
                             , { rename = "Torch.Sig.Types.Global" , to = "Torch.Types.TH" }
                             , { rename = "Torch.Sig.Types" , to = "Torch.Types.TH.Float" }
                             , { rename = "Torch.Sig.Storage" , to = "Torch.FFI.TH.Float.Storage" }
                             , { rename = "Torch.Sig.Storage.Copy" , to = "Torch.FFI.TH.Float.StorageCopy" }
                             , { rename = "Torch.Sig.Storage.Memory" , to = "Torch.FFI.TH.Float.FreeStorage" }
                             , { rename = "Torch.Sig.Tensor" , to = "Torch.FFI.TH.Float.Tensor" }
                             , { rename = "Torch.Sig.Tensor.Copy" , to = "Torch.FFI.TH.Float.TensorCopy" }
                             , { rename = "Torch.Sig.Tensor.Memory" , to = "Torch.FFI.TH.Float.FreeTensor" }
                             , { rename = "Torch.Sig.Tensor.Index" , to = "Torch.FFI.TH.Float.TensorMath" }
                             , { rename = "Torch.Sig.Tensor.Masked" , to = "Torch.FFI.TH.Float.TensorMath" }
                             , { rename = "Torch.Sig.Tensor.Math" , to = "Torch.FFI.TH.Float.TensorMath" }
                             , { rename = "Torch.Sig.Tensor.Math.Compare" , to = "Torch.FFI.TH.Float.TensorMath" }
                             , { rename = "Torch.Sig.Tensor.Math.CompareT" , to = "Torch.FFI.TH.Float.TensorMath" }
                             , { rename = "Torch.Sig.Tensor.Math.Pairwise" , to = "Torch.FFI.TH.Float.TensorMath" }
                             , { rename = "Torch.Sig.Tensor.Math.Pointwise" , to = "Torch.FFI.TH.Float.TensorMath" }
                             , { rename = "Torch.Sig.Tensor.Math.Reduce" , to = "Torch.FFI.TH.Float.TensorMath" }
                             , { rename = "Torch.Sig.Tensor.Math.Scan" , to = "Torch.FFI.TH.Float.TensorMath" }
                             , { rename = "Torch.Sig.Tensor.Mode" , to = "Torch.FFI.TH.Float.TensorMath" }
                             , { rename = "Torch.Sig.Tensor.ScatterGather" , to = "Torch.FFI.TH.Float.TensorMath" }
                             , { rename = "Torch.Sig.Tensor.Sort" , to = "Torch.FFI.TH.Float.TensorMath" }
                             , { rename = "Torch.Sig.Tensor.TopK" , to = "Torch.FFI.TH.Float.TensorMath" }
                             , { rename = "Torch.Sig.Tensor.Math.Pointwise.Signed" , to = "Torch.FFI.TH.Float.TensorMath" }
                             , { rename = "Torch.Sig.Tensor.Math.Pointwise.Floating" , to = "Torch.FFI.TH.Float.TensorMath" }
                             , { rename = "Torch.Sig.Tensor.Math.Blas" , to = "Torch.FFI.TH.Float.TensorMath" }
                             , { rename = "Torch.Sig.Tensor.Math.Lapack" , to = "Torch.FFI.TH.Float.TensorLapack" }
                             , { rename = "Torch.Sig.Tensor.Math.Floating" , to = "Torch.FFI.TH.Float.TensorMath" }
                             , { rename = "Torch.Sig.Tensor.Math.Reduce.Floating" , to = "Torch.FFI.TH.Float.TensorMath" }
                             , { rename = "Torch.Sig.NN" , to = "Torch.FFI.TH.NN.Float" }
                             , { rename = "Torch.Sig.Types.NN" , to = "Torch.Types.TH" }
                             , { rename = "Torch.Sig.Tensor.Math.Random.TH" , to = "Torch.FFI.TH.Float.TensorMath" }
                             , { rename = "Torch.Sig.Tensor.Random.TH" , to = "Torch.FFI.TH.Float.TensorRandom" }
                             , { rename = "Torch.Sig.Tensor.Random.THC" , to = "Torch.Undefined.Float.Tensor.Random.THC" }
                             ]
                         }
                     }
                   , { package = "hasktorch-partial-floating"
                     , renaming =
                         { provides = prelude.types.ModuleRenaming.renaming
                             [ { rename = "Torch.Undefined.Tensor.Random.THC"
                               , to = "Torch.Undefined.Double.Tensor.Random.THC"
                               } ]
                         , requires = prelude.types.ModuleRenaming.renaming
                             [ { rename = "Torch.Sig.State" , to = "Torch.Types.TH" }
                             , { rename = "Torch.Sig.Types.Global" , to = "Torch.Types.TH" }
                             , { rename = "Torch.Sig.Types" , to = "Torch.Types.TH.Double" }
                             , { rename = "Torch.Sig.Storage" , to = "Torch.FFI.TH.Double.Storage" }
                             , { rename = "Torch.Sig.Storage.Copy" , to = "Torch.FFI.TH.Double.StorageCopy" }
                             , { rename = "Torch.Sig.Storage.Memory" , to = "Torch.FFI.TH.Double.FreeStorage" }
                             , { rename = "Torch.Sig.Tensor" , to = "Torch.FFI.TH.Double.Tensor" }
                             , { rename = "Torch.Sig.Tensor.Copy" , to = "Torch.FFI.TH.Double.TensorCopy" }
                             , { rename = "Torch.Sig.Tensor.Memory" , to = "Torch.FFI.TH.Double.FreeTensor" }
                             , { rename = "Torch.Sig.Tensor.Index" , to = "Torch.FFI.TH.Double.TensorMath" }
                             , { rename = "Torch.Sig.Tensor.Masked" , to = "Torch.FFI.TH.Double.TensorMath" }
                             , { rename = "Torch.Sig.Tensor.Math" , to = "Torch.FFI.TH.Double.TensorMath" }
                             , { rename = "Torch.Sig.Tensor.Math.Compare" , to = "Torch.FFI.TH.Double.TensorMath" }
                             , { rename = "Torch.Sig.Tensor.Math.CompareT" , to = "Torch.FFI.TH.Double.TensorMath" }
                             , { rename = "Torch.Sig.Tensor.Math.Pairwise" , to = "Torch.FFI.TH.Double.TensorMath" }
                             , { rename = "Torch.Sig.Tensor.Math.Pointwise" , to = "Torch.FFI.TH.Double.TensorMath" }
                             , { rename = "Torch.Sig.Tensor.Math.Reduce" , to = "Torch.FFI.TH.Double.TensorMath" }
                             , { rename = "Torch.Sig.Tensor.Math.Scan" , to = "Torch.FFI.TH.Double.TensorMath" }
                             , { rename = "Torch.Sig.Tensor.Mode" , to = "Torch.FFI.TH.Double.TensorMath" }
                             , { rename = "Torch.Sig.Tensor.ScatterGather" , to = "Torch.FFI.TH.Double.TensorMath" }
                             , { rename = "Torch.Sig.Tensor.Sort" , to = "Torch.FFI.TH.Double.TensorMath" }
                             , { rename = "Torch.Sig.Tensor.TopK" , to = "Torch.FFI.TH.Double.TensorMath" }
                             , { rename = "Torch.Sig.Tensor.Math.Pointwise.Signed" , to = "Torch.FFI.TH.Double.TensorMath" }
                             , { rename = "Torch.Sig.Tensor.Math.Pointwise.Floating" , to = "Torch.FFI.TH.Double.TensorMath" }
                             , { rename = "Torch.Sig.Tensor.Math.Blas" , to = "Torch.FFI.TH.Double.TensorMath" }
                             , { rename = "Torch.Sig.Tensor.Math.Lapack" , to = "Torch.FFI.TH.Double.TensorLapack" }
                             , { rename = "Torch.Sig.Tensor.Math.Floating" , to = "Torch.FFI.TH.Double.TensorMath" }
                             , { rename = "Torch.Sig.Tensor.Math.Reduce.Floating" , to = "Torch.FFI.TH.Double.TensorMath" }
                             , { rename = "Torch.Sig.NN" , to = "Torch.FFI.TH.NN.Double" }
                             , { rename = "Torch.Sig.Types.NN" , to = "Torch.Types.TH" }
                             , { rename = "Torch.Sig.Tensor.Math.Random.TH" , to = "Torch.FFI.TH.Double.TensorMath" }
                             , { rename = "Torch.Sig.Tensor.Random.TH" , to = "Torch.FFI.TH.Double.TensorRandom" }
                             , { rename = "Torch.Sig.Tensor.Random.THC" , to = "Torch.Undefined.Double.Tensor.Random.THC" }
                             ]
                         } }
                   ] } }
          , { name = "isdefinite-floating-thc"
            , executable =
              λ(config : types.Config) → exeScaffold "floating" "thc"
              // { mixins =
                   [ { package =
                         "hasktorch-partial-floating"
                     , renaming =
                         { provides = prelude.types.ModuleRenaming.default {=}
                         , requires = prelude.types.ModuleRenaming.renaming
                             [ { rename = "Torch.Sig.Types.Global" , to = "Torch.Types.THC" }
                             , { rename = "Torch.Sig.Types" , to = "Torch.Types.THC.Double" }
                             , { rename = "Torch.Sig.State" , to = "Torch.FFI.THC.State" }
                             , { rename = "Torch.Sig.Storage" , to = "Torch.FFI.THC.Double.Storage" }
                             , { rename = "Torch.Sig.Storage.Copy" , to = "Torch.FFI.THC.Double.StorageCopy" }
                             , { rename = "Torch.Sig.Storage.Memory" , to = "Torch.FFI.THC.Double.Storage" }
                             , { rename = "Torch.Sig.Tensor" , to = "Torch.FFI.THC.Double.Tensor" }
                             , { rename = "Torch.Sig.Tensor.Copy" , to = "Torch.FFI.THC.Double.TensorCopy" }
                             , { rename = "Torch.Sig.Tensor.Memory" , to = "Torch.FFI.THC.Double.Tensor" }
                             , { rename = "Torch.Sig.Tensor.Index" , to = "Torch.FFI.THC.Double.TensorIndex" }
                             , { rename = "Torch.Sig.Tensor.Masked" , to = "Torch.FFI.THC.Double.TensorMasked" }
                             , { rename = "Torch.Sig.Tensor.Math" , to = "Torch.FFI.THC.Double.TensorMath" }
                             , { rename = "Torch.Sig.Tensor.Math.Compare" , to = "Torch.FFI.THC.Double.TensorMathCompare" }
                             , { rename = "Torch.Sig.Tensor.Math.CompareT" , to = "Torch.FFI.THC.Double.TensorMathCompareT" }
                             , { rename = "Torch.Sig.Tensor.Math.Pairwise" , to = "Torch.FFI.THC.Double.TensorMathPairwise" }
                             , { rename = "Torch.Sig.Tensor.Math.Pointwise" , to = "Torch.FFI.THC.Double.TensorMathPointwise" }
                             , { rename = "Torch.Sig.Tensor.Math.Reduce" , to = "Torch.FFI.THC.Double.TensorMathReduce" }
                             , { rename = "Torch.Sig.Tensor.Math.Scan" , to = "Torch.FFI.THC.Double.TensorMathScan" }
                             , { rename = "Torch.Sig.Tensor.Mode" , to = "Torch.FFI.THC.Double.TensorMode" }
                             , { rename = "Torch.Sig.Tensor.ScatterGather" , to = "Torch.FFI.THC.Double.TensorScatterGather" }
                             , { rename = "Torch.Sig.Tensor.Sort" , to = "Torch.FFI.THC.Double.TensorSort" }
                             , { rename = "Torch.Sig.Tensor.TopK" , to = "Torch.FFI.THC.Double.TensorTopK" }
                             , { rename = "Torch.Sig.Tensor.Math.Pointwise.Signed" , to = "Torch.FFI.THC.Double.TensorMathPointwise" }
                             , { rename = "Torch.Sig.Tensor.Math.Blas" , to = "Torch.FFI.THC.Double.TensorMathBlas" }
                             , { rename = "Torch.Sig.Tensor.Math.Lapack" , to = "Torch.FFI.THC.Double.TensorMathMagma" }
                             , { rename = "Torch.Sig.Tensor.Math.Floating" , to = "Torch.FFI.THC.Double.TensorMath" }
                             , { rename = "Torch.Sig.Tensor.Math.Pointwise.Floating" , to = "Torch.FFI.THC.Double.TensorMathPointwise" }
                             , { rename = "Torch.Sig.Tensor.Math.Reduce.Floating" , to = "Torch.FFI.THC.Double.TensorMathReduce" }
                             , { rename = "Torch.Sig.NN" , to = "Torch.FFI.THC.NN.Double" }
                             , { rename = "Torch.Sig.Types.NN" , to = "Torch.Types.THC" }
                             , { rename = "Torch.Sig.Tensor.Math.Random.TH" , to = "Torch.Undefined.Tensor.Math.Random.TH" }
                             , { rename = "Torch.Sig.Tensor.Random.TH" , to = "Torch.Undefined.Tensor.Random.TH" }
                             , { rename = "Torch.Sig.Tensor.Random.THC" , to = "Torch.FFI.THC.Double.TensorRandom" }
                             ]
                         }
                     }
                   ]
                } }
          ]
      }
