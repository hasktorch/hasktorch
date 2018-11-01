    let prelude = ../dhall/dhall-to-cabal/dhall/prelude.dhall sha256:01509b3c6e9eaae4150a6e0ced124c2db191bf6046534a9d4973b7f05afd1d0a
in  let types   = ../dhall/dhall-to-cabal/dhall/types.dhall sha256:cfd7597246781e8d4c6dfa5f0eabba75f14dc3f3deb7527973909b37c93f42f5
in  let common = ../dhall/common.dhall
in  let packages = common.packages
in  let cabalvars = common.cabalvars
in  let partials = ../dhall/backpack/partials.dhall
in  let mixins = ../dhall/backpack/mixins.dhall
in  let fn = ../dhall/common/functions.dhall

in  let exeScaffold =
    \(sigset : Text) →
    \(isth : Bool) →
      prelude.defaults.Executable
          // { main-is = "Main.hs"
             , default-language = cabalvars.default-language
             , hs-source-dirs = [ "exe" ]
             , build-depends =
               [ packages.base
               , if isth then packages.hasktorch-ffi-th else packages.hasktorch-ffi-thc
               , if isth then packages.hasktorch-types-th else packages.hasktorch-types-thc
               , fn.anyver ("hasktorch-partial-" ++ sigset)
               ]
             }

in  common.Package
   // { name = "hasktorch-signatures"
      , synopsis  = "Backpack signatures for Tensor operations"
      , description = "Backpack signature files to glue FFI backends to Hasktorch"
      , library =
          [   λ(config : types.Config)
            →   common.Library config
              // { hs-source-dirs = [ "src" ]
                 , build-depends =
                    [ packages.base
                    , packages.hasktorch-types-th
                    , packages.hasktorch-types-thc
                    , packages.hasktorch-signatures-types
                    ]
                , default-extensions = [] : List types.Extensions
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
          [ partials.floating
          , partials.signed
          , partials.unsigned
          ]

      , executables =
          [ { name = "isdefinite-unsigned-th"
            , executable =
              λ(config : types.Config) → exeScaffold "unsigned" True
               // { mixins =
                   [ { package = partials.unsigned.name
                     , renaming =
                       { provides = prelude.types.ModuleRenaming.default {=}
                       , requires = prelude.types.ModuleRenaming.renaming (mixins.unsigned True "Byte")
                       }
                     }
                   , { package = partials.unsigned.name
                     , renaming =
                       { provides = prelude.types.ModuleRenaming.default {=}
                       , requires = prelude.types.ModuleRenaming.renaming (mixins.unsigned True "Char")
                       }
                     } ] } }

          , { name = "isdefinite-unsigned-thc"
            , executable =
              λ(config : types.Config) → exeScaffold "unsigned" False
              // { mixins =
                   [ { package = partials.unsigned.name
                     , renaming =
                       { provides = prelude.types.ModuleRenaming.default {=}
                       , requires = prelude.types.ModuleRenaming.renaming (mixins.unsigned False "Byte")
                       }
                     }
                   , { package = partials.unsigned.name
                     , renaming =
                       { provides = prelude.types.ModuleRenaming.default {=}
                       , requires = prelude.types.ModuleRenaming.renaming (mixins.unsigned False "Char")
                       }
                     } ] } }

          , { name = "isdefinite-signed-th"
            , executable =
              λ(config : types.Config) → exeScaffold "signed" True
              // { mixins =
                   [ { package = partials.signed.name
                     , renaming =
                         { provides = prelude.types.ModuleRenaming.default {=}
                         , requires = prelude.types.ModuleRenaming.renaming (mixins.signed True "Short")
                         }
                     }
                   , { package = partials.signed.name
                     , renaming =
                         { provides = prelude.types.ModuleRenaming.default {=}
                         , requires = prelude.types.ModuleRenaming.renaming (mixins.signed True "Int")
                         }
                     }
                   , { package = partials.signed.name
                     , renaming =
                         { provides = prelude.types.ModuleRenaming.default {=}
                         , requires = prelude.types.ModuleRenaming.renaming (mixins.signed True "Long")
                         }
                     }
                   ]
                 }
            }
          , { name = "isdefinite-signed-thc"
            , executable =
              λ(config : types.Config) → exeScaffold "signed" False
              // { mixins =
                    [ { package = partials.signed.name
                      , renaming =
                         { provides = prelude.types.ModuleRenaming.default {=}
                         , requires = prelude.types.ModuleRenaming.renaming (mixins.signed False "Short")
                         }
                      }
                    , { package = partials.signed.name
                      , renaming =
                         { provides = prelude.types.ModuleRenaming.default {=}
                         , requires = prelude.types.ModuleRenaming.renaming (mixins.signed False "Int")
                         }
                      }
                    , { package = partials.signed.name
                      , renaming =
                         { provides = prelude.types.ModuleRenaming.default {=}
                         , requires = prelude.types.ModuleRenaming.renaming (mixins.signed False "Long")
                         }
                      }
                    ]
                  }
            }
          , { name = "isdefinite-floating-th"
            , executable =
              λ(config : types.Config) → exeScaffold "floating" True
              // { mixins =
                   [ { package = partials.floating.name
                     , renaming =
                         { provides = prelude.types.ModuleRenaming.renaming
                             [ { rename = "Torch.Undefined.Tensor.Random.THC"
                               , to = "Torch.Undefined.Float.Tensor.Random.THC"
                               } ]
                         , requires = prelude.types.ModuleRenaming.renaming (mixins.floating True "Float")
                         }
                     }
                   , { package = partials.floating.name
                     , renaming =
                         { provides = prelude.types.ModuleRenaming.renaming
                             [ { rename = "Torch.Undefined.Tensor.Random.THC"
                               , to = "Torch.Undefined.Double.Tensor.Random.THC"
                               } ]
                         , requires = prelude.types.ModuleRenaming.renaming (mixins.floating True "Double")
                         } }
                   ] } }
          , { name = "isdefinite-floating-thc"
            , executable =
              λ(config : types.Config) → exeScaffold "floating" False
              // { mixins =
                   [ { package = partials.floating.name
                     , renaming =
                         { provides = prelude.types.ModuleRenaming.renaming
                             [ { rename = "Torch.Undefined.Tensor.Random.TH"
                               , to = "Torch.Undefined.Cuda.Double.Tensor.Random.TH"
                               }
                             , { rename = "Torch.Undefined.Tensor.Math.Random.TH"
                               , to = "Torch.Undefined.Cuda.Double.Tensor.Math.Random.TH"
                               } ]
                         , requires = prelude.types.ModuleRenaming.renaming (mixins.floating False "Double")
                         }
                     }
                   ]
                } }
          ]
      }
