    let prelude = ../../dhall/dhall-to-cabal/dhall/prelude.dhall sha256:01509b3c6e9eaae4150a6e0ced124c2db191bf6046534a9d4973b7f05afd1d0a
in  let types   = ../../dhall/dhall-to-cabal/dhall/types.dhall sha256:cfd7597246781e8d4c6dfa5f0eabba75f14dc3f3deb7527973909b37c93f42f5
in  let common = ../../dhall/common.dhall
in  let packages = common.packages
in  let cabalvars = common.cabalvars

in  common.Package
   // { name = "hasktorch-signatures-types"
      , synopsis = "Core types for Hasktorch backpack signatures"
      , description = "This package includes core signature types to abstract over the hasktorch-types-* packages."
      , library =
          [   λ(config : types.Config)
            →   common.Library config
              // { hs-source-dirs = [ "src" ]
                 , default-extensions = [] : List types.Extensions
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

