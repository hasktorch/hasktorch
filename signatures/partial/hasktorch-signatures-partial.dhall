    let prelude = https://raw.githubusercontent.com/dhall-lang/dhall-to-cabal/master/dhall/prelude.dhall
in  let types = https://raw.githubusercontent.com/dhall-lang/dhall-to-cabal/master/dhall/types.dhall
in  let common = ../../dhall/common.dhall
in  let packages = common.packages
in  let cabalvars = common.cabalvars

in  common.Package
   // { name = "hasktorch-signatures-partial"
      , synopsis = "Torch for tensors and neural networks in Haskell"
      , description = "Undefined functions to satisfy backpack signatures (never be exported in core)"
      , library =
          [   λ(config : types.Config)
            →   common.Library
              // { hs-source-dirs = [ "src" ]
                 , build-depends =
                    [ packages.base
                    , packages.hasktorch-types-th
                    , packages.hasktorch-signatures-types
                    -- , packages.inline-c
                    , packages.text
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


