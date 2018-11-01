    let prelude = ../../dhall/dhall-to-cabal/dhall/prelude.dhall sha256:01509b3c6e9eaae4150a6e0ced124c2db191bf6046534a9d4973b7f05afd1d0a
in  let types   = ../../dhall/dhall-to-cabal/dhall/types.dhall sha256:cfd7597246781e8d4c6dfa5f0eabba75f14dc3f3deb7527973909b37c93f42f5
in  let common = ../../dhall/common.dhall
in  let packages = common.packages
in  common.Package
   // { name = "hasktorch-signatures-support"
      , synopsis = "Signatures for support tensors in hasktorch"
      , description = "Backpack signatures which define redundant operators for mask tensors and index tensors in Hasktorch."
      , library =
          [   λ(config : types.Config)
            →   common.Library config
              // { hs-source-dirs = [ "src" ]
                 , default-extensions = [] : List types.Extensions
                 , build-depends =
                    [ packages.base
                    , packages.hasktorch-signatures-types
                    , packages.hasktorch-types-th
                    ]
                 , signatures =
                   [ "Torch.Sig.Index.Tensor"
                   , "Torch.Sig.Index.TensorFree"
                   , "Torch.Sig.Mask.Tensor"
                   , "Torch.Sig.Mask.TensorFree"
                   , "Torch.Sig.Mask.MathReduce"
                   ]
                }
          ] : Optional (types.Config → types.Library)
      }

