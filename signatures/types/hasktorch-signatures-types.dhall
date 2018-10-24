    let prelude = ../../dhall/dhall-to-cabal/dhall/prelude.dhall
in  let types =   ../../dhall/dhall-to-cabal/dhall/types.dhall
in  let common = ../../dhall/common.dhall
in  let packages = common.packages
in  let cabalvars = common.cabalvars

in  common.Package
   // { name = "hasktorch-signatures-types"
      , synopsis = "Backpack signature types to pair with hasktorch-ffi and hasktorch"
      , description = "CFFI backpack signatures"
      , library =
          [   λ(config : types.Config)
            →   common.Library
              // { hs-source-dirs = [ "src" ]
                 , build-depends =
                    [ packages.base
                    , packages.deepseq
                    ]
                 , signatures =
                   [ "Torch.Sig.State"
                   , "Torch.Sig.Types"
                   , "Torch.Sig.Types.Global"
                   , "Torch.Sig.Tensor.Memory"
                   , "Torch.Sig.Storage.Memory"
                   ]
                }
          ] : Optional (types.Config → types.Library)
      }

