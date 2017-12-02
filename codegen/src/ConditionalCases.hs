{-# LANGUAGE OverloadedStrings #-}
module ConditionalCases where

import Data.List as L
import Data.Map as M
import Data.Set as S
import Data.Text

import CodeGenTypes

makeSet = S.fromList

tensorMathCases :: Map Text (Set TemplateType)
tensorMathCases = M.fromList [
  ("abs", makeSet [GenShort, GenInt, GenLong, GenFloat, GenDouble]),

  ("sigmoid", makeSet [GenFloat, GenDouble]),
  ("log", makeSet [GenFloat, GenDouble]),
  ("lgamma", makeSet [GenFloat, GenDouble]),
  ("log1p", makeSet [GenFloat, GenDouble]),
  ("exp", makeSet [GenFloat, GenDouble]),
  ("erf", makeSet [GenFloat, GenDouble]),
  ("erfinv", makeSet [GenFloat, GenDouble]),
  ("cos", makeSet [GenFloat, GenDouble]),
  ("acos", makeSet [GenFloat, GenDouble]),
  ("cosh", makeSet [GenFloat, GenDouble]),
  ("sin", makeSet [GenFloat, GenDouble]),
  ("asin", makeSet [GenFloat, GenDouble]),
  ("sinh", makeSet [GenFloat, GenDouble]),
  ("tan", makeSet [GenFloat, GenDouble]),
  ("atan", makeSet [GenFloat, GenDouble]),
  ("atan2", makeSet [GenFloat, GenDouble]),
  ("tanh", makeSet [GenFloat, GenDouble]),
  ("pow", makeSet [GenFloat, GenDouble]),
  ("tpow", makeSet [GenFloat, GenDouble]),
  ("sqrt", makeSet [GenFloat, GenDouble]),
  ("rsqrt", makeSet [GenFloat, GenDouble]),
  ("ceil", makeSet [GenFloat, GenDouble]),
  ("floor", makeSet [GenFloat, GenDouble]),
  ("round", makeSet [GenFloat, GenDouble]),
  -- ("abs", makeSet [GenFloat, GenDouble]), -- covered above
  ("trunc", makeSet [GenFloat, GenDouble]),
  ("frac", makeSet [GenFloat, GenDouble]),
  ("lerp", makeSet [GenFloat, GenDouble]),
  ("mean", makeSet [GenFloat, GenDouble]),
  ("std", makeSet [GenFloat, GenDouble]),
  ("var", makeSet [GenFloat, GenDouble]),
  ("norm", makeSet [GenFloat, GenDouble]),
  ("renorm", makeSet [GenFloat, GenDouble]),
  ("dist", makeSet [GenFloat, GenDouble]),
  ("histc", makeSet [GenFloat, GenDouble]),
  ("bhistc", makeSet [GenFloat, GenDouble]),
  ("meanall", makeSet [GenFloat, GenDouble]),
  ("varall", makeSet [GenFloat, GenDouble]),
  ("stdall", makeSet [GenFloat, GenDouble]),
  ("normall", makeSet [GenFloat, GenDouble]),
  ("linspace", makeSet [GenFloat, GenDouble]),
  ("logspace", makeSet [GenFloat, GenDouble]),
  ("rand", makeSet [GenFloat, GenDouble]),
  ("randn", makeSet [GenFloat, GenDouble]),

  ("logicalall", makeSet [GenByte]),
  ("logicalany", makeSet [GenByte]),

  -- cinv doesn't seem to be excluded by the preprocessor, yet is not
  -- implemented for Int. TODO - file issue report?
  ("cinv", makeSet [GenFloat, GenDouble]),
  ("neg", makeSet [GenFloat, GenDouble, GenLong, GenShort, GenInt])
  ]

tensorRandomCases :: Map Text (Set TemplateType)
tensorRandomCases = M.fromList [
  ("uniform", makeSet [GenFloat, GenDouble]),
  ("normal", makeSet [GenFloat, GenDouble]),
  ("normal_means", makeSet [GenFloat, GenDouble]),
  ("normal_stddevs", makeSet [GenFloat, GenDouble]),
  ("normal_means_stddevs", makeSet [GenFloat, GenDouble]),
  ("exponential", makeSet [GenFloat, GenDouble]),
  ("cauchy", makeSet [GenFloat, GenDouble]),
  ("logNormal", makeSet [GenFloat, GenDouble]),
  ("multinomial", makeSet [GenFloat, GenDouble]),
  ("multinomialAliasSetup", makeSet [GenFloat, GenDouble]),
  ("multinomialAliasDraw", makeSet [GenFloat, GenDouble]),
  ("getRNGState", makeSet [GenByte]),
  ("setRNGState", makeSet [GenByte])
  ]


-- TODO: check lapack bindings - not obvious from source, but there are
-- problems loading shared library with these functions for Byte
tensorLapackCases = M.fromList [
  ("gesv", makeSet [GenFloat, GenDouble]),
  ("trtrs", makeSet [GenFloat, GenDouble]),
  ("gels", makeSet [GenFloat, GenDouble]),
  ("syev", makeSet [GenFloat, GenDouble]),
  ("geev", makeSet [GenFloat, GenDouble]),
  ("gesvd", makeSet [GenFloat, GenDouble]),
  ("gesvd2", makeSet [GenFloat, GenDouble]),
  ("getrf", makeSet [GenFloat, GenDouble]),
  ("getrs", makeSet [GenFloat, GenDouble]),
  ("getri", makeSet [GenFloat, GenDouble]),
  ("potrf", makeSet [GenFloat, GenDouble]),
  ("potrs", makeSet [GenFloat, GenDouble]),
  ("potri", makeSet [GenFloat, GenDouble]),
  ("qr", makeSet [GenFloat, GenDouble]),
  ("geqrf", makeSet [GenFloat, GenDouble]),
  ("orgqr", makeSet [GenFloat, GenDouble]),
  ("ormqr", makeSet [GenFloat, GenDouble]),
  ("pstrf", makeSet [GenFloat, GenDouble]),
  ("btrifact", makeSet [GenFloat, GenDouble]),
  ("btrisolve", makeSet [GenFloat, GenDouble])]
  -- ("geev", makeSet [GenFloat, GenDouble]),
  -- ("gels", makeSet [GenFloat, GenDouble]),
  -- ("gesv", makeSet [GenFloat, GenDouble]),
  -- ("gesvd", makeSet [GenFloat, GenDouble])

checkMath :: TemplateType -> Text -> Bool
checkMath templateType funName = case M.lookup funName tensorMathCases of
  Just inclusion -> S.member templateType inclusion
  Nothing -> True

checkRandom :: TemplateType -> Text -> Bool
checkRandom templateType funName = case M.lookup funName tensorRandomCases of
  Just inclusion -> S.member templateType inclusion
  Nothing -> True

checkLapack :: TemplateType -> Text -> Bool
checkLapack templateType funName = case M.lookup funName tensorLapackCases of
  Just inclusion -> S.member templateType inclusion
  Nothing -> True

-- |Warning a function that doesn't exist will return True by default (TODO - make this safer)

checkFunction :: TemplateType -> Text -> Bool
checkFunction templateType funName =
  and [(checkMath templateType funName),
       (checkRandom templateType funName),
       (checkLapack templateType funName)
       ]

test = do
  print $ checkFunction GenByte "logicalany"
  print $ checkFunction GenFloat "logicalany"
  print $ checkFunction GenByte "multinomial"
  print $ checkFunction GenFloat "multinomial"
