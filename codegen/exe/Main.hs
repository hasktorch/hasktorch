{-# LANGUAGE ScopedTypeVariables #-}
module Main where

import Control.Monad (when)
import Data.List (nub)
import Text.Show.Pretty (ppShow)
import Options.Applicative (ParserInfo, execParser, info, (<**>), helper, idm)

import CodeGen.FileMappings
import CodeGen.Types
import CodeGen.Render (writeHaskellModule, parseFile)

import CLIOptions

-- ========================================================================= --

main :: IO ()
main = execParser opts >>= run
 where
  opts :: ParserInfo Options
  opts = info (cliOptions <**> helper) idm


run :: Options -> IO ()
run os = do
  when (verbose os) $ putStrLn $ unwords
    [ "Running"
    , show gentype
    , "codegen for"
    , show lib
    , "files"
    ]

  case lib of
    TH -> mapM_ (runTHPipeline os) (files lib gentype)
    THC -> mapM_ (runTHPipeline os) (files lib gentype)
    lib -> putStrLn $ "Code generation not enabled for " ++ show lib

  when (verbose os) $ putStrLn "Done"

 where
  lib :: LibType
  lib = libraries os

  gentype :: CodeGenType
  gentype = codegenType os


runTHPipeline
  :: Options
  -> (String, TemplateType -> [THFunction] -> HModule)
  -> IO ()
runTHPipeline os (headerPath, makeModuleConfig) = do
  -- TODO: @nub@ is a hack until proper treatment of
  --       conditioned templates is implemented
  bindingsUniq <- nub <$> parseFile (codegenType os) headerPath
  
  when (verbose os) $ do
    putStrLn $ "First signature of " ++ show (length bindingsUniq)
    putStrLn $ ppShow (take 1 bindingsUniq)

  mapM_ (writeHaskellModule bindingsUniq makeModuleConfig) typeList

  when (verbose os) $
    putStrLn $ "Number of functions generated: "
      ++ show (length typeList * length bindingsUniq)
 where
  typeList :: [TemplateType]
  typeList = generatedTypes (codegenType os)


{-
-- generic tests
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
-}
