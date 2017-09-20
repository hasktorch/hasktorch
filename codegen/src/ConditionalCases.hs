{-# LANGUAGE OverloadedStrings #-}
module ConditionalCases where

import Data.List as L
import Data.Map as M
import Data.Set as S
import Data.Text

import CodeGenTypes

tensorMathCases :: Map Text (Set TemplateType)
tensorMathCases = M.fromList [
  ("abs", S.fromList [GenShort, GenInt, GenLong, GenFloat, GenDouble]),

  ("sigmoid", S.fromList [GenFloat, GenDouble]),
  ("log", S.fromList [GenFloat, GenDouble]),
  ("lgamma", S.fromList [GenFloat, GenDouble]),
  ("log1p", S.fromList [GenFloat, GenDouble]),
  ("exp", S.fromList [GenFloat, GenDouble]),
  ("cos", S.fromList [GenFloat, GenDouble]),
  ("acos", S.fromList [GenFloat, GenDouble]),
  ("cosh", S.fromList [GenFloat, GenDouble]),
  ("sin", S.fromList [GenFloat, GenDouble]),
  ("asin", S.fromList [GenFloat, GenDouble]),
  ("sinh", S.fromList [GenFloat, GenDouble]),
  ("tan", S.fromList [GenFloat, GenDouble]),
  ("atan", S.fromList [GenFloat, GenDouble]),
  ("atan2", S.fromList [GenFloat, GenDouble]),
  ("tanh", S.fromList [GenFloat, GenDouble]),
  ("pow", S.fromList [GenFloat, GenDouble]),
  ("tpow", S.fromList [GenFloat, GenDouble]),
  ("sqrt", S.fromList [GenFloat, GenDouble]),
  ("rsqrt", S.fromList [GenFloat, GenDouble]),
  ("ceil", S.fromList [GenFloat, GenDouble]),
  ("floor", S.fromList [GenFloat, GenDouble]),
  ("round", S.fromList [GenFloat, GenDouble]),
  -- ("abs", S.fromList [GenFloat, GenDouble]), -- covered above
  ("trunc", S.fromList [GenFloat, GenDouble]),
  ("frac", S.fromList [GenFloat, GenDouble]),
  ("lerp", S.fromList [GenFloat, GenDouble]),
  ("mean", S.fromList [GenFloat, GenDouble]),
  ("std", S.fromList [GenFloat, GenDouble]),
  ("var", S.fromList [GenFloat, GenDouble]),
  ("norm", S.fromList [GenFloat, GenDouble]),
  ("renorm", S.fromList [GenFloat, GenDouble]),
  ("dist", S.fromList [GenFloat, GenDouble]),
  ("histc", S.fromList [GenFloat, GenDouble]),
  ("bhistc", S.fromList [GenFloat, GenDouble]),
  ("meanall", S.fromList [GenFloat, GenDouble]),
  ("varall", S.fromList [GenFloat, GenDouble]),
  ("stdall", S.fromList [GenFloat, GenDouble]),
  ("normall", S.fromList [GenFloat, GenDouble]),
  ("linspace", S.fromList [GenFloat, GenDouble]),
  ("logspace", S.fromList [GenFloat, GenDouble]),
  ("rand", S.fromList [GenFloat, GenDouble]),
  ("randn", S.fromList [GenFloat, GenDouble]),

  ("logicalall", S.fromList [GenByte]),
  ("logicalany", S.fromList [GenByte])
  ]

tensorRandomCases :: Map Text (Set TemplateType)
tensorRandomCases = M.fromList [
  ("uniform", S.fromList [GenFloat, GenDouble]),
  ("normal", S.fromList [GenFloat, GenDouble]),
  ("normal_means", S.fromList [GenFloat, GenDouble]),
  ("normal_stddevs", S.fromList [GenFloat, GenDouble]),
  ("normal_means_stddevs", S.fromList [GenFloat, GenDouble]),
  ("exponential", S.fromList [GenFloat, GenDouble]),
  ("cauchy", S.fromList [GenFloat, GenDouble]),
  ("logNormal", S.fromList [GenFloat, GenDouble]),
  ("multinomial", S.fromList [GenFloat, GenDouble]),
  ("multinomialAliasSetup", S.fromList [GenFloat, GenDouble]),
  ("multinomialAliasDraw", S.fromList [GenFloat, GenDouble]),

  ("getRNGState", S.fromList [GenByte]),
  ("setRNGState", S.fromList [GenByte])
  ]

checkMath :: TemplateType -> Text -> Bool
checkMath templateType funName = case M.lookup funName tensorMathCases of
  Just inclusion -> S.member templateType inclusion
  Nothing -> True

checkRandom :: TemplateType -> Text -> Bool
checkRandom templateType funName = case M.lookup funName tensorRandomCases of
  Just inclusion -> S.member templateType inclusion
  Nothing -> True

-- |Warning a function that doesn't exist will return True by default (TODO - make this safer)

checkFunction :: TemplateType -> Text -> Bool
checkFunction templateType funName =
  and [(checkMath templateType funName),
       (checkRandom templateType funName)]

test = do
  print $ checkFunction GenByte "logicalany"
  print $ checkFunction GenFloat "logicalany"
  print $ checkFunction GenByte "multinomial"
  print $ checkFunction GenFloat "multinomial"
