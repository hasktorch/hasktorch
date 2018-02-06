{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE OverloadedLists #-}
module ConditionalCases where

import GHC.Exts (IsString)
import Data.Text (Text)
import Data.HashMap.Strict (HashMap)
import Data.HashSet (HashSet)
import Data.Hashable (Hashable)
import qualified Data.Text as T
import qualified Data.HashMap.Strict as M (lookup)
import qualified Data.HashSet as S (member)

import CodeGenTypes

signatureAliases :: TemplateType -> HashSet HsTypeAlias
signatureAliases = \case
  GenByte ->
    [ CTensor  "THTypes.CTHByteTensor"
    , CReal    "Data.Word.CUChar"
    , CAccReal "Foreign.C.Types.CLong"
    , CStorage "THTypes.CTHByteStorage"
    ]
  GenChar ->
    [ CTensor  "THTypes.CTHCharTensor"
    , CReal    "Foreign.C.Types.CChar"
    , CAccReal "Foreign.C.Types.CLong"
    , CStorage "THTypes.CTHCharStorage"
    ]
  GenDouble ->
    [ CTensor  "THTypes.CTHDoubleTensor"
    , CReal    "Foreign.C.Types.CDouble"
    , CAccReal "Foreign.C.Types.CDouble"
    , CStorage "THTypes.CTHDoubleStorage"
    ]
  GenFloat ->
    [ CTensor  "THTypes.CTHFloatTensor"
    , CReal    "Foreign.C.Types.CFloat"
    , CAccReal "Foreign.C.Types.CDouble"
    , CStorage "THTypes.CTHFloatStorage"
    ]
  GenHalf ->
    [ CTensor  "THTypes.CTHHalfTensor"
    , CReal    "THTypes.CTHHalf"
    , CAccReal "Foreign.C.Types.CFloat"
    , CStorage "THTypes.CTHHalfStorage"
    ]
  GenInt ->
    [ CTensor  "THTypes.CTHIntTensor"
    , CReal    "Foreign.C.Types.CInt"
    , CAccReal "Foreign.C.Types.CLong"
    , CStorage "THTypes.CTHIntStorage"
    ]
  GenLong ->
    [ CTensor  "THTypes.CTHLongTensor"
    , CReal    "Foreign.C.Types.CLong"
    , CAccReal "Foreign.C.Types.CLong"
    , CStorage "THTypes.CTHLongStorage"
    ]
  GenShort ->
    [ CTensor  "THTypes.CTHShortTensor"
    , CReal    "Foreign.C.Types.CShort"
    , CAccReal "Foreign.C.Types.CLong"
    , CStorage "THTypes.CTHShortStorage"
    ]
  GenNothing -> []


tensorMathCases :: HashMap FunctionName (HashSet TemplateType)
tensorMathCases =
  [ ("abs",     [GenShort, GenInt, GenLong, GenFloat, GenDouble])
  , ("sigmoid", [GenFloat, GenDouble])
  , ("log",     [GenFloat, GenDouble])
  , ("lgamma",  [GenFloat, GenDouble])
  , ("log1p",   [GenFloat, GenDouble])
  , ("exp",     [GenFloat, GenDouble])
  , ("erf",     [GenFloat, GenDouble])
  , ("erfinv",  [GenFloat, GenDouble])
  , ("cos",     [GenFloat, GenDouble])
  , ("acos",    [GenFloat, GenDouble])
  , ("cosh",    [GenFloat, GenDouble])
  , ("sin",     [GenFloat, GenDouble])
  , ("asin",    [GenFloat, GenDouble])
  , ("sinh",    [GenFloat, GenDouble])
  , ("tan",     [GenFloat, GenDouble])
  , ("atan",    [GenFloat, GenDouble])
  , ("atan2",   [GenFloat, GenDouble])
  , ("tanh",    [GenFloat, GenDouble])
  , ("pow",     [GenFloat, GenDouble])
  , ("tpow",    [GenFloat, GenDouble])
  , ("sqrt",    [GenFloat, GenDouble])
  , ("rsqrt",   [GenFloat, GenDouble])
  , ("ceil",    [GenFloat, GenDouble])
  , ("floor",   [GenFloat, GenDouble])
  , ("round",   [GenFloat, GenDouble])
  , ("trunc",   [GenFloat, GenDouble])
  , ("frac",    [GenFloat, GenDouble])
  , ("lerp",    [GenFloat, GenDouble])
  , ("mean",    [GenFloat, GenDouble])
  , ("std",     [GenFloat, GenDouble])
  , ("var",     [GenFloat, GenDouble])
  , ("norm",    [GenFloat, GenDouble])
  , ("renorm",  [GenFloat, GenDouble])
  , ("dist",    [GenFloat, GenDouble])
  , ("histc",   [GenFloat, GenDouble])
  , ("bhistc",  [GenFloat, GenDouble])
  , ("meanall", [GenFloat, GenDouble])
  , ("varall",  [GenFloat, GenDouble])
  , ("stdall",  [GenFloat, GenDouble])
  , ("normall", [GenFloat, GenDouble])
  , ("linspace",[GenFloat, GenDouble])
  , ("logspace",[GenFloat, GenDouble])
  , ("rand",    [GenFloat, GenDouble])
  , ("randn",   [GenFloat, GenDouble])
  , ("logicalall", [GenByte])
  , ("logicalany", [GenByte])

  -- cinv doesn't seem to be excluded by the preprocessor, yet is not
  -- implemented for Int. TODO - file issue report?
  , ("cinv", [GenFloat, GenDouble])
  , ("neg",  [GenFloat, GenDouble, GenLong, GenShort, GenInt])
  ]

tensorRandomCases :: HashMap FunctionName (HashSet TemplateType)
tensorRandomCases =
  [ ("uniform",        [GenFloat, GenDouble])
  , ("normal",         [GenFloat, GenDouble])
  , ("normal_means",   [GenFloat, GenDouble])
  , ("normal_stddevs", [GenFloat, GenDouble])
  , ("normal_means_stddevs", [GenFloat, GenDouble])
  , ("exponential",    [GenFloat, GenDouble])
  , ("standard_gamma", [GenFloat, GenDouble])
  , ("digamma",        [GenFloat, GenDouble])
  , ("trigamma",       [GenFloat, GenDouble])
  , ("polygamma",      [GenFloat, GenDouble])
  , ("expm1",          [GenFloat, GenDouble])
  , ("dirichlet_grad", [GenFloat, GenDouble])
  , ("cauchy",         [GenFloat, GenDouble])
  , ("logNormal",      [GenFloat, GenDouble])
  , ("multinomial",    [GenFloat, GenDouble])
  , ("multinomialAliasSetup", [GenFloat, GenDouble])
  , ("multinomialAliasDraw",  [GenFloat, GenDouble])
  , ("getRNGState", [GenByte])
  , ("setRNGState", [GenByte])
  ]


-- TODO: check lapack bindings - not obvious from source, but there are
-- problems loading shared library with these functions for Byte
tensorLapackCases :: HashMap FunctionName (HashSet TemplateType)
tensorLapackCases =
  [ ("gesv",   [GenFloat, GenDouble])
  , ("trtrs",  [GenFloat, GenDouble])
  , ("gels",   [GenFloat, GenDouble])
  , ("syev",   [GenFloat, GenDouble])
  , ("geev",   [GenFloat, GenDouble])
  , ("gesvd",  [GenFloat, GenDouble])
  , ("gesvd2", [GenFloat, GenDouble])
  , ("getrf",  [GenFloat, GenDouble])
  , ("getrs",  [GenFloat, GenDouble])
  , ("getri",  [GenFloat, GenDouble])
  , ("potrf",  [GenFloat, GenDouble])
  , ("potrs",  [GenFloat, GenDouble])
  , ("potri",  [GenFloat, GenDouble])
  , ("qr",     [GenFloat, GenDouble])
  , ("geqrf",  [GenFloat, GenDouble])
  , ("orgqr",  [GenFloat, GenDouble])
  , ("ormqr",  [GenFloat, GenDouble])
  , ("pstrf",  [GenFloat, GenDouble])
  , ("btrifact",  [GenFloat, GenDouble])
  , ("btrisolve", [GenFloat, GenDouble])
  -- , ("geev", [GenFloat, GenDouble])
  -- , ("gels", [GenFloat, GenDouble])
  -- , ("gesv", [GenFloat, GenDouble])
  -- , ("gesvd", [GenFloat, GenDouble])
  ]

checkMath :: TemplateType -> FunctionName -> Bool
checkMath = checkMap tensorMathCases

checkRandom :: TemplateType -> FunctionName -> Bool
checkRandom = checkMap tensorRandomCases

checkLapack :: TemplateType -> FunctionName -> Bool
checkLapack = checkMap tensorLapackCases

checkMap
  :: HashMap FunctionName (HashSet TemplateType)
  -> TemplateType
  -> FunctionName
  -> Bool
checkMap map tt n = maybe False (tt `S.member`) (M.lookup n map)


-- | Warning a function that doesn't exist will return True by default
--
-- TODO: make this safer.
-- (stites): to make this safer I think we need to invert these maps so that we
--           are given function names instead of doing membership checks.
checkFunction :: TemplateType -> FunctionName -> Bool
checkFunction tt fn
  =  checkMath   tt fn
  && checkRandom tt fn
  && checkLapack tt fn

test :: IO ()
test = do
  print $ checkFunction GenByte  "logicalany"
  print $ checkFunction GenFloat "logicalany"
  print $ checkFunction GenByte  "multinomial"
  print $ checkFunction GenFloat "multinomial"
