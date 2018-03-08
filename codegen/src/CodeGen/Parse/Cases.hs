{-# LANGUAGE OverloadedLists #-}
module CodeGen.Parse.Cases
  ( type2real
  , type2hsreal
  , type2accreal
  , checkFunction
  ) where

import CodeGen.Prelude hiding (char)
import qualified Data.HashMap.Strict as M
import qualified Data.HashSet as S

import CodeGen.Types hiding (prefix)

uchar, long, char :: (HsRep -> CRep -> x) -> x
uchar  cons = cons "Foreign.C.Types.CUChar"  "unsigned char"
long   cons = cons "Foreign.C.Types.CLong"   "long"
char   cons = cons "Foreign.C.Types.CChar"   "char"
double cons = cons "Foreign.C.Types.CDouble" "double"
float  cons = cons "Foreign.C.Types.CFloat"  "float"
int    cons = cons "Foreign.C.Types.CInt"    "int"
short  cons = cons "Foreign.C.Types.CShort"  "short"
half   cons = cons (HsRep $ prefix TH "Half") "THHalf"

prefix :: LibType -> Text -> Text
prefix lt t = "Torch.Types." <> tshow lt <> ".C"<> tshow lt <> t

signatureAliases :: LibType -> TemplateType -> Maybe (CTensor, CReal, CAccReal, CStorage)
signatureAliases lt = \case
  GenByte   -> Just (mkTuple "Byte"   uchar  long)
  GenChar   -> Just (mkTuple "Char"   char   long)
  GenDouble -> Just (mkTuple "Double" double double)
  GenFloat  -> Just (mkTuple "Float"  float  double)
  GenHalf   -> Just (mkTuple "Half"   half   float)
  GenInt    -> Just (mkTuple "Int"    int    long)
  GenLong   -> Just (mkTuple "Long"   long   long)
  GenShort  -> Just (mkTuple "Short"  short  long)
  GenNothing -> Nothing
 where
  mkRep :: (HsRep -> CRep -> x) -> Text -> Text -> x
  mkRep cons suffix t = cons (HsRep . prefix lt $ t <> suffix) (CRep $ t <> suffix)

  mkCTensor :: Text -> CTensor
  mkCTensor = mkRep CTensor "Tensor"

  mkCStorage :: Text -> CStorage
  mkCStorage = mkRep CStorage "Storage"

  mkTuple :: Text -> ((HsRep -> CRep -> CReal) -> CReal) -> ((HsRep -> CRep -> CAccReal) -> CAccReal) -> (CTensor, CReal, CAccReal, CStorage)
  mkTuple t r ac = (mkCTensor t, r CReal, ac CAccReal, mkCStorage t)


type2real :: LibType -> TemplateType -> Text
type2real lt t = case signatureAliases lt t of
  Just (_, CReal hs _, _, _) -> stripModule hs
  Nothing -> "" -- impossible "TemplateType is concrete and should not have been called"

-- | spliced text to use for function names
type2hsreal :: TemplateType -> Text
type2hsreal = \case
  GenByte    -> "Byte"
  GenChar    -> "Char"
  GenDouble  -> "Double"
  GenFloat   -> "Float"
  GenHalf    -> "Half"
  GenInt     -> "Int"
  GenLong    -> "Long"
  GenShort   -> "Short"
  GenNothing -> ""


type2accreal :: LibType -> TemplateType -> Text
type2accreal lt t = case signatureAliases lt t of
  Just (_, _, CAccReal hs _, _) -> stripModule hs
  Nothing -> "" -- impossible "TemplateType is concrete and should not have been called"


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

  -- This keeps appearing but isn't in TH. TODO: find out what is happening
  , ("bernoulli_Tensor", [])
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

storageCases :: HashMap FunctionName (HashSet TemplateType)
storageCases =
  [ ("elementSize", [])
  ]

checkMath :: TemplateType -> FunctionName -> Bool
checkMath = checkMap tensorMathCases

checkRandom :: TemplateType -> FunctionName -> Bool
checkRandom = checkMap tensorRandomCases

checkLapack :: TemplateType -> FunctionName -> Bool
checkLapack = checkMap tensorLapackCases

checkStorage :: TemplateType -> FunctionName -> Bool
checkStorage = checkMap storageCases

checkMap
  :: HashMap FunctionName (HashSet TemplateType)
  -> TemplateType
  -> FunctionName
  -> Bool
checkMap map tt n = maybe True (tt `S.member`) (M.lookup n map)


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
  && checkStorage tt fn

test :: IO ()
test = do
  print $ checkFunction GenByte  "logicalany"
  print $ checkFunction GenFloat "logicalany"
  print $ checkFunction GenByte  "multinomial"
  print $ checkFunction GenFloat "multinomial"
