    let prelude = https://raw.githubusercontent.com/dhall-lang/dhall-to-cabal/master/dhall/prelude.dhall
in  let types = https://raw.githubusercontent.com/dhall-lang/dhall-to-cabal/master/dhall/types.dhall
in  let common = ../common.dhall
in  let packages = common.packages
in  let cabalvars = common.cabalvars
in  let fn = ../common/functions.dhall

in  let partialLibrary = common.Library
        // { build-depends =
             [ packages.base
             , packages.hasktorch-partial
             , packages.hasktorch-signatures
             ]
           }

in  let
  signed-definites =
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

in
{ unsigned =
    { name = "hasktorch-partial-unsigned"
    , library =
      λ(config : types.Config)
       → partialLibrary
        // { mixins =
             [ { package = "hasktorch-signatures"
               , renaming =
                 { provides = prelude.types.ModuleRenaming.default {=}
                 , requires = prelude.types.ModuleRenaming.renaming
                     (signed-definites # [ fn.renameSig "Undefined" "Tensor.Math.Pointwise.Signed" ])
                 }
               } ]
           } }


, signed =
    { name = "hasktorch-partial-signed"
    , library =
      λ(config : types.Config)
       → partialLibrary
        // { mixins =
             [ { package = "hasktorch-signatures"
               , renaming =
                 { provides = prelude.types.ModuleRenaming.default {=}
                 , requires = prelude.types.ModuleRenaming.renaming signed-definites
                 }
               } ]
           } }

, floating =
    { name = "hasktorch-partial-floating"
    , library =
      λ(config : types.Config)
       → partialLibrary
        // { reexported-modules =
             [ fn.renameNoop "Torch.Undefined.Tensor.Math.Random.TH"
             , fn.renameNoop "Torch.Undefined.Tensor.Random.TH"
             , fn.renameNoop "Torch.Undefined.Tensor.Random.THC"
             ]
           } }
}
