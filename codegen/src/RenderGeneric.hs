{-# LANGUAGE OverloadedLists #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FlexibleContexts #-}

module Main where

import Control.Monad (void)
import Data.List (nub)
import Data.Monoid ((<>))
import Data.Maybe
import Data.Void
import Data.Text
import Data.Text as T
import Text.Megaparsec
import Text.Megaparsec.Char
import Text.Megaparsec.Expr
import qualified Text.Megaparsec.Char.Lexer as L
import Prelude as P
import Text.Show.Pretty

import CodeGenParse
import CodeGenTypes
import ConditionalCases
import RenderShared

outDirGeneric = "./output/core/src/generic/" :: Text

genericFiles :: [(String, TemplateType -> [THFunction] -> HModule)]
genericFiles =
  [
    ("vendor/torch7/lib/TH/generic/THBlas.h",
     (makeModule outDirGeneric True "THBlas.h" "Blas" "Blas")),
    ("vendor/torch7/lib/TH/generic/THLapack.h",
     (makeModule outDirGeneric True "THLapack.h" "Lapack" "Lapack")),
    ("vendor/torch7/lib/TH/generic/THStorage.h",
     (makeModule outDirGeneric True "THStorage.h" "Storage" "Storage")),
    ("vendor/torch7/lib/TH/generic/THStorageCopy.h",
     (makeModule outDirGeneric True "THStorageCopy.h" "Storage" "StorageCopy")),
    ("vendor/torch7/lib/TH/generic/THTensor.h",
     (makeModule outDirGeneric True "THTensor.h" "Tensor" "Tensor")),
    ("vendor/torch7/lib/TH/generic/THTensorConv.h",
     (makeModule outDirGeneric True "THTensorConv.h" "Tensor" "TensorConv")),
    ("vendor/torch7/lib/TH/generic/THTensorCopy.h",
     (makeModule outDirGeneric True "THTensorCopy.h" "Tensor" "TensorCopy")),
    ("vendor/torch7/lib/TH/generic/THTensorLapack.h",
     (makeModule outDirGeneric True "THTensorLapack.h" "Tensor" "TensorLapack")),
    ("vendor/torch7/lib/TH/generic/THTensorMath.h",
     (makeModule outDirGeneric True "THTensorMath.h" "Tensor" "TensorMath")),
    ("vendor/torch7/lib/TH/generic/THTensorRandom.h",
     (makeModule outDirGeneric True "THTensorRandom.h" "Tensor" "TensorRandom")),
    ("vendor/torch7/lib/TH/generic/THVector.h",
     (makeModule outDirGeneric True "THVector.h" "Vector" "Vector"))
  ]

-- |TODO: make a unified module that re-exports all functions
makeReExports = do
  putStrLn "Re-exported Tensors"

testString inp parser = case (parse parser "" inp) of
  Left err -> putStrLn (parseErrorPretty err)
  Right val -> putStrLn $ (ppShow val)

check :: IO ()
check = do
  testString exampleText thParseGeneric
  where
    exampleText = "skip this garbage line line\n" <>
      "TH_API void THTensor_(setFlag)(THTensor *self,const char flag);" <>
      "another garbage line ( )@#R @# 324 32"

runPipeline ::
  [Char] -> (TemplateType -> [THFunction] -> HModule) -> [TemplateType]-> IO ()
runPipeline headerPath makeModuleConfig typeList = do
  parsedBindings <- parseFile headerPath
  let bindingsUniq = nub parsedBindings
  putStrLn $ "First signature:"
  putStrLn $ ppShow (P.take 1 bindingsUniq)
  mapM_ (\x -> renderCHeaderFile x bindingsUniq makeModuleConfig) typeList
  putStrLn $ "Number of functions generated: " ++
    (show $ P.length typeList * P.length bindingsUniq)

main :: IO ()
main = do
  mapM_ (\(file, spec) -> runPipeline file spec genericTypes) genericFiles
  putStrLn "Done"
