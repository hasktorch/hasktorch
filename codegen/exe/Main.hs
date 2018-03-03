{-# LANGUAGE OverloadedLists #-}
{-# LANGUAGE FlexibleContexts #-}
module Main where

import Options.Applicative
import Control.Monad (void)
import Data.List (nub, intercalate)
import Data.Monoid ((<>))
import Data.Void (Void)
import Data.Text (Text)
import Text.Megaparsec (parse, parseErrorPretty)
import Text.Show.Pretty (ppShow)

import CodeGenParse (THFunction, Parser, thParseGeneric)
import CodeGen.Types (HModule, TemplateType, genericTypes)
import RenderShared (makeTHModule, renderCHeaderFile, parseFile)

data LibType
  = TH
  | THC
  | THNN
  | THCuNN
  | THS

describe :: LibType -> String
describe = \case
  TH -> "Torch7 (ATen re-export)"
  THC -> "Cuda-based Torch7 (ATen re-export)"
  THNN -> "THNN (ATen re-export)"
  THCuNN -> "Cuda-based THNN (ATen re-export)"
  THS -> "TH Sparse tensor support (ATen library)"

data Options
  = Hello [String]
  | Goodbye
  deriving (Eq, Show)

main :: IO ()
main = execParser opts >>= run

hello :: Options.Applicative.Parser Options
hello = Hello <$> many (argument str (metavar "TARGET..."))

sample :: Options.Applicative.Parser Options
sample = subparser
       ( command "generic"  (info hello (progDesc "Print greeting"))
      -- <> command "thc" (info (pure Goodbye) (progDesc "Say goodbye"))
       )

run :: Options -> IO ()
run (Hello targets) = putStrLn $ "Hello, " ++ intercalate ", " targets ++ "!"
run Goodbye = putStrLn "Goodbye."

opts :: ParserInfo Options
opts = info (sample <**> helper) idm

-------------------------------------------------------------------------------
-- RENDER GENNERIC

outDirGeneric :: Text
thDir, thnnDir :: String
outDirGeneric = "./output/raw/src/generic/"
thDir   = "vendor/TH/generic/"
thnnDir = "vendor/pytorch/aten/src/THNN/generic/"


thGenericFiles :: [(String, TemplateType -> [THFunction] -> HModule)]
thGenericFiles =
  [ (thDir <> "THBlas.h"         , (makeGenericModule "THBlas.h" "Blas" "Blas"))
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
    makeGenericModule = makeTHModule outDirGeneric True

thGenericRunPipeline
  :: [TemplateType] -> String -> (TemplateType -> [THFunction] -> HModule) -> IO ()
thGenericRunPipeline typeList headerPath makeModuleConfig = do
  parsedBindings <- parseFile headerPath
  let bindingsUniq = nub parsedBindings
  putStrLn $ "First signature of " ++ show (length bindingsUniq)
  putStrLn $ ppShow (take 1 bindingsUniq)

  mapM_ (\x ->  renderCHeaderFile x bindingsUniq makeModuleConfig) typeList
  putStrLn $ "Number of functions generated: " ++ show (length typeList * length bindingsUniq)

thGenericMain :: IO ()
thGenericMain = do
  mapM_ (uncurry (thGenericRunPipeline genericTypes)) thGenericFiles
  putStrLn "Done"

genericTestString :: String -> CodeGenParse.Parser [Maybe THFunction] -> IO ()
genericTestString inp parser =
  case parse parser "" inp of
    Left err -> putStrLn (parseErrorPretty err)
    Right val -> putStrLn (ppShow val)

genericCheck :: IO ()
genericCheck = genericTestString exampleText thParseGeneric
  where
    exampleText :: String
    exampleText = intercalate "\n"
      [ "skip this garbage line line"
      , "TH_API void THTensor_(setFlag)(THTensor *self,const char flag);"
      , "another garbage line ( )@#R @# 324 32"
      ]
