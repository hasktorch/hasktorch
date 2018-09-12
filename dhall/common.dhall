    let prelude = https://raw.githubusercontent.com/dhall-lang/dhall-to-cabal/master/dhall/prelude.dhall
in  let types = https://raw.githubusercontent.com/dhall-lang/dhall-to-cabal/master/dhall/types.dhall
in  let fn = ./common/functions.dhall
in  let packages =
  { base =
    { package = "base"
    , bounds =
        prelude.intersectVersionRanges
          ( prelude.unionVersionRanges
            (prelude.thisVersion (prelude.v "4.7"))
            (prelude.laterVersion (prelude.v "4.7")))
          (prelude.earlierVersion (prelude.v "5"))
    } : types.Dependency
  , text =
    { package = "text"
    , bounds =
        prelude.unionVersionRanges
          (prelude.thisVersion (prelude.v "1.2.2.2"))
          (prelude.laterVersion (prelude.v "1.2.2.2"))
    } : types.Dependency
  , hspec =
    { package = "hspec"
    , bounds =
        prelude.unionVersionRanges
          (prelude.thisVersion (prelude.v "2.4.4"))
          (prelude.laterVersion (prelude.v "2.4.4"))
    } : types.Dependency

  , hasktorch-raw-th    = fn.anyver "hasktorch-raw-th"
  , hasktorch-raw-thc   = fn.anyver "hasktorch-raw-thc"
  , hasktorch-types-th  = fn.anyver "hasktorch-types-th"
  , hasktorch-types-thc = fn.anyver "hasktorch-types-thc"

  , hasktorch-signatures-types = fn.anyver "hasktorch-signatures-types"
  , hasktorch-signatures       = fn.anyver "hasktorch-signatures"
  , hasktorch-partial          = fn.anyver "hasktorch-partial"

  , QuickCheck = fn.anyver "QuickCheck"
  , containers = fn.anyver "containers"
  , deepseq =
    { bounds =
      prelude.unionVersionRanges
        ( prelude.thisVersion (prelude.v "1.3.0.0") )
        ( prelude.laterVersion (prelude.v "1.3.0.0") )
    , package = "deepseq" }
  , dimensions =
    { bounds =
      prelude.unionVersionRanges
        ( prelude.thisVersion (prelude.v "1.0"))
        ( prelude.laterVersion (prelude.v "1.0"))
    , package = "dimensions" }
  , managed =
    { bounds =
      prelude.intersectVersionRanges
        ( prelude.unionVersionRanges
          ( prelude.thisVersion (prelude.v "1.0.0"))
          ( prelude.laterVersion (prelude.v "1.0.0")))
        ( prelude.earlierVersion (prelude.v "1.1"))
    , package = "managed" }
  , microlens =
    { bounds =
      prelude.unionVersionRanges
        ( prelude.thisVersion (prelude.v "0.4.8.1"))
        ( prelude.laterVersion (prelude.v "0.4.8.1"))
    , package = "microlens" }
  , numeric-limits = fn.anyver "numeric-limits"
  , safe-exceptions =
    { bounds =
      prelude.unionVersionRanges
        ( prelude.thisVersion (prelude.v "0.1.0.0"))
        ( prelude.laterVersion (prelude.v "0.1.0.0"))
    , package = "safe-exceptions" }
  , singletons =
    { bounds =
      prelude.unionVersionRanges
        ( prelude.thisVersion (prelude.v "2.2"))
        ( prelude.laterVersion (prelude.v "2.2"))
    , package = "singletons" }
  , text =
    { bounds =
      prelude.unionVersionRanges
        ( prelude.thisVersion (prelude.v "1.2.2.2"))
        ( prelude.laterVersion (prelude.v "1.2.2.2"))
    , package = "text" }
  , typelits-witnesses = { bounds =
      prelude.unionVersionRanges
        ( prelude.thisVersion (prelude.v "0.2.3.0"))
        ( prelude.laterVersion (prelude.v "0.2.3.0"))
    , package = "typelits-witnesses" }
  , backprop =
    { bounds =
      prelude.unionVersionRanges
        (prelude.thisVersion (prelude.v "0.2.5"))
        (prelude.laterVersion (prelude.v "0.2.5"))
    , package = "backprop" }
  , ghc-typelits-natnormalise = fn.anyver "ghc-typelits-natnormalise"
  , hasktorch-core = fn.anyver "hasktorch-core"
  , mtl = fn.anyver "mtl"
  , microlens-platform = fn.anyver "microlens-platform"
  , monad-loops = fn.anyver "monad-loops"
  , time = fn.anyver "time"
  , transformers = fn.anyver "transformers"

  , hasktorch-indef-floating = fn.anyver "hasktorch-indef-floating"
  , hasktorch-indef-signed   = fn.anyver "hasktorch-indef-signed"
  , hasktorch-indef-unsigned = fn.anyver "hasktorch-indef-unsigned"
  }

in  let cabalvars =
  { author = "Hasktorch dev team (Sam Stites, Austin Huang)" : Text
  , bug-reports = "https://github.com/hasktorch/hasktorch/issues" : Text
  , version = prelude.v "0.0.1.0" : types.Version
  , build-type = [ prelude.types.BuildTypes.Simple {=} ] : Optional types.BuildType
  , homepage = "https://github.com/hasktorch/hasktorch#readme" : Text

  , default-language =
    [ < Haskell2010 = {=} | UnknownLanguage : { _1 : Text } | Haskell98 : {} >
    ] : Optional types.Language
  , license =
      prelude.types.Licenses.SPDX
        ( prelude.SPDX.license
          (prelude.types.LicenseId.BSD_3_Clause {=})
          ([] : Optional types.LicenseExceptionId)
        ) : types.License
  , source-repos =
    [ prelude.defaults.SourceRepo
      // { type = [ prelude.types.RepoType.Git {=} ] : Optional types.RepoType
         , location = [ "https://github.com/hasktorch/hasktorch" ] : Optional Text
         }
    ] : List types.SourceRepo
  , default-extensions = [ prelude.types.Extensions.LambdaCase True
      , prelude.types.Extensions.DataKinds True
      , prelude.types.Extensions.TypeFamilies True
      , prelude.types.Extensions.TypeSynonymInstances True
      , prelude.types.Extensions.ScopedTypeVariables True
      , prelude.types.Extensions.FlexibleContexts True
      ]
  }

in
{ cabalvars = cabalvars
, packages = packages
, Package = prelude.defaults.Package
  // { version = cabalvars.version
     , author = cabalvars.author
     , bug-reports = cabalvars.bug-reports
     , build-type = cabalvars.build-type
     , homepage = cabalvars.homepage
     , license = cabalvars.license
     , source-repos = cabalvars.source-repos
     , category = "Tensors, Machine Learning"
     }
, Library = prelude.defaults.Library
  //
    { default-language = cabalvars.default-language
    , default-extensions = cabalvars.default-extensions
    }
, flags =
  { cuda =
      { name = "cuda"
      , description = "build with THC support"
      , manual = False
      , default = False
      }
  , lite =
      { name = "lite"
      , description = "only build with Double and Long support"
      , manual = False
      , default = False
      }
  }
}


