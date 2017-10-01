{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE RecordWildCards #-}

module Main where

import Prelude as P

import Data.Text
import Data.List (nub)
import Text.Megaparsec
import Text.Show.Pretty

import Text.RawString.QQ

import CodeGenParse
import CodeGenTypes
import ConditionalCases
import RenderShared

outDirManaged = "./output/tensor/src/" :: Text

apply0String = [r|

-- |Generalize non-mutating collapse of a tensor to a constant or another tensor
apply0_ :: (Ptr CTHDoubleTensor -> a) -> TensorDouble_ -> a
apply0_ operation tensor = unsafePerformIO $ do
  withForeignPtr (tdTensor tensor) (\t -> pure $ operation t)

|]

files :: [(String, TemplateType -> [THFunction] -> HModule)]
files =
  [
    ("vendor/torch7/lib/TH/generic/THTensor.h",
     (makeModule outDirManaged True "THTensor.h" "Tensor" "Tensor")),
    ("vendor/torch7/lib/TH/generic/THTensorMath.h",
     (makeModule outDirManaged True "THTensorMath.h" "Tensor" "TensorMath"))
  ]

-- runPipelineManaged ::
--   [Char] -> (TemplateType -> [THFunction] -> HModule) -> [TemplateType]-> IO ()
runPipelineManaged headerPath makeModuleConfig typeList = do
  let headerPath = "vendor/torch7/lib/TH/generic/THTensorMath.h"
  parsedBindings <- parseFile headerPath
  let bindingsUniq = nub parsedBindings
  putStrLn $ "First signature:"
  putStrLn $ ppShow (P.take 1 bindingsUniq)
  pure ()
--   mapM_ (\x -> renderCHeaderFile x bindingsUniq makeModuleConfig) typeList
--   putStrLn $ "Number of functions generated: " ++
--     (show $ P.length typeList * P.length bindingsUniq)

main = do
  mapM_ (\(file, spec) -> runPipelineManaged file spec files) files
  putStrLn "Done"
