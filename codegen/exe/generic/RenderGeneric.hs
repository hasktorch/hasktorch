{-# LANGUAGE OverloadedLists #-}
{-# LANGUAGE FlexibleContexts #-}
module Main where

import Prelude

import Control.Monad (void)
import Data.List (nub, intercalate)
import Data.Monoid ((<>))
import Data.Void (Void)
import Data.Text (Text)
import Text.Megaparsec (parse, parseErrorPretty)
import Text.Show.Pretty (ppShow)

import CodeGenParse (THFunction, Parser, thParseGeneric)
import CodeGenTypes (HModule, TemplateType, genericTypes)
import RenderShared (makeModule, renderCHeaderFile, parseFile)

outDirGeneric :: Text
thDir, thnnDir :: String
outDirGeneric = "./output/raw/src/generic/"
thDir   = "vendor/TH/generic/"
thnnDir = "vendor/pytorch/aten/src/THNN/generic/"

genericFiles :: [(String, TemplateType -> [THFunction] -> HModule)]
genericFiles =
  [ (thnnDir <> "THNN.h"         , (makeGenericModule "THNN.h" "NN" "NN"))
  , (thDir <> "THBlas.h"         , (makeGenericModule "THBlas.h" "Blas" "Blas"))
  , (thDir <> "THLapack.h"       , (makeGenericModule "THLapack.h" "Lapack" "Lapack"))
  , (thDir <> "THStorage.h"      , (makeGenericModule "THStorage.h" "Storage" "Storage"))
  , (thDir <> "THStorageCopy.h"  , (makeGenericModule "THStorageCopy.h" "Storage" "StorageCopy"))
  , (thDir <> "THTensor.h"       , (makeGenericModule "THTensor.h" "Tensor" "Tensor"))
  , (thDir <> "THTensorConv.h"   , (makeGenericModule "THTensorConv.h" "Tensor" "TensorConv"))
  , (thDir <> "THTensorCopy.h"   , (makeGenericModule "THTensorCopy.h" "Tensor" "TensorCopy"))
  , (thDir <> "THTensorLapack.h" , (makeGenericModule "THTensorLapack.h" "Tensor" "TensorLapack"))
  , (thDir <> "THTensorMath.h"   , (makeGenericModule "THTensorMath.h" "Tensor" "TensorMath"))
  , (thDir <> "THTensorRandom.h" , (makeGenericModule "THTensorRandom.h" "Tensor" "TensorRandom"))
  , (thDir <> "THVector.h"       , (makeGenericModule "THVector.h" "Vector" "Vector"))
  ]
  where
    makeGenericModule :: FilePath -> Text -> Text -> (TemplateType -> [THFunction] -> HModule)
    makeGenericModule = makeModule outDirGeneric True

-- |TODO: make a unified module that re-exports all functions
makeReExports :: IO ()
makeReExports = putStrLn "Re-exported Tensors"

testString :: String -> Parser [Maybe THFunction] -> IO ()
testString inp parser =
  case parse parser "" inp of
    Left err -> putStrLn (parseErrorPretty err)
    Right val -> putStrLn (ppShow val)

check :: IO ()
check = testString exampleText thParseGeneric
  where
    exampleText :: String
    exampleText = intercalate "\n"
      [ "skip this garbage line line"
      , "TH_API void THTensor_(setFlag)(THTensor *self,const char flag);"
      , "another garbage line ( )@#R @# 324 32"
      ]

runPipeline :: [TemplateType] -> String -> (TemplateType -> [THFunction] -> HModule) -> IO ()
runPipeline typeList headerPath makeModuleConfig = do
  parsedBindings <- parseFile headerPath
  let bindingsUniq = nub parsedBindings
  putStrLn   "First signature:"
  putStrLn $ ppShow (take 1 bindingsUniq)
  mapM_ (\x -> renderCHeaderFile x bindingsUniq makeModuleConfig) typeList
  putStrLn $ "Number of functions generated: " ++ show (length typeList * length bindingsUniq)

main :: IO ()
main = do
  mapM_ (uncurry (runPipeline genericTypes)) genericFiles
  putStrLn "Done"
