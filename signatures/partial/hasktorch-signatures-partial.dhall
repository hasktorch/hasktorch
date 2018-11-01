    let prelude = ../../dhall/dhall-to-cabal/dhall/prelude.dhall sha256:01509b3c6e9eaae4150a6e0ced124c2db191bf6046534a9d4973b7f05afd1d0a
in  let types   = ../../dhall/dhall-to-cabal/dhall/types.dhall sha256:cfd7597246781e8d4c6dfa5f0eabba75f14dc3f3deb7527973909b37c93f42f5
in  let common  = ../../dhall/common.dhall
in  let packages = common.packages
in  let cabalvars = common.cabalvars

in  common.Package
   // { name = "hasktorch-signatures-partial"
      , synopsis = "Functions to partially satisfy tensor signatures"
      , description = "Undefined functions to satisfy backpack signatures (never be exported in core)"
      , library =
          [   λ(config : types.Config)
            →   common.Library config
              // { hs-source-dirs = [ "src" ]
                 , default-extensions = [] : List types.Extensions
                 , build-depends =
                    [ packages.base
                    , packages.hasktorch-types-th
                    , packages.hasktorch-signatures-types
                    ]
                 , exposed-modules =
                   [ "Torch.Undefined.NN"
                   , "Torch.Undefined.Types.NN"
                   , "Torch.Undefined.Tensor.Math.Blas"
                   , "Torch.Undefined.Tensor.Math.Floating"
                   , "Torch.Undefined.Tensor.Math.Lapack"
                   , "Torch.Undefined.Tensor.Math.Pointwise.Signed"
                   , "Torch.Undefined.Tensor.Math.Pointwise.Floating"
                   , "Torch.Undefined.Tensor.Math.Reduce.Floating"

                   , "Torch.Undefined.Tensor.Random.TH"
                   , "Torch.Undefined.Tensor.Random.THC"
                   , "Torch.Undefined.Tensor.Math.Random.TH"
                   ]
                }
          ] : Optional (types.Config → types.Library)
      }


