module Main where

import Prelude as P

import Data.Monoid ((<>))
import Data.Text
import Data.List (nub)
import Text.Megaparsec
import Text.Show.Pretty

import Text.RawString.QQ

import CodeGenParse
import CodeGen.Types
import ConditionalCases
import RenderShared

outDirManaged = "./output/tensor/src/" :: Text

files :: [(String, TemplateType -> [THFunction] -> HModule)]
files =
  [
    ("vendor/TH/generic/THTensor.h",
     (makeTHModule outDirManaged True "THTensor.h" "Tensor" "Tensor")),
    ("vendor/TH/generic/THTensorRandom.h",
     (makeTHModule outDirManaged True "THTensorRandom.h" "Tensor" "Tensor")),
    ("vendor/TH/generic/THTensorMath.h",
     (makeTHModule outDirManaged True "THTensorMath.h" "Tensor" "TensorMath"))
  ]



-- |variable names of int args corresponding to boolean values
fakeBoolVars = [
  -- Tensor
  "keepdim",
  -- TensorRandom
  "with_replacement"
  ] :: [Text]

-- |render Haskell to C conversions for a single function argument
renderConversion :: THArg -> Text
renderConversion arg =
  case (thArgType arg) of
    THChar -> integralCase
    THShort -> integralCase
    THInt -> integralCase
    THPtrDiff -> integralCase
    THLong -> integralCase
    THFloat -> realCase
    THDouble -> realCase
    _ -> ""
  where
    integralCase =
      ("    " <> (thArgName arg) <> "_C = fromIntegral " <> (thArgName arg) <> "\n")
    realCase =
      ("    " <> (thArgName arg) <> "_C = realToFrac" <> (thArgName arg) <> "\n")

-- |render Haskell to C conversions for all function arguments
renderConversions :: [THArg] -> Text
renderConversions args =
  if conversions /= "" then
    ("  where\n" <> conversions)
  else "  -- no argument conversions needed\n"
  where conversions = intercalate "" (renderConversion <$> args)

-- |entrypoint for processing managed-memory files
runPipelineManaged headerPath makeModuleConfig typeList = do
  -- let headerPath = "vendor/TH/generic/THTensorMath.h"
  parsedBindings <- parseFile headerPath
  let bindingsUniq = nub parsedBindings
  putStrLn $ "First signature:"
  putStrLn $ ppShow (P.take 1 bindingsUniq)
  -- let test = P.take 1 bindingsUniq
  -- let argtypes = thArgType <$> funArgs (test !! 0)
  let conversions = (renderConversions . funArgs) <$> bindingsUniq
  mapM_ (putStr . unpack) conversions
  pure ()


main = do
  mapM_ (\(file, spec) -> runPipelineManaged file spec files) files
  putStrLn "Done"
