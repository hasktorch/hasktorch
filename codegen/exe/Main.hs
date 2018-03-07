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
  if supported lib
  then do
    mapM_ (runPipeline os)
      ( files lib gentype)
      -- (filter ((== "./vendor/pytorch/aten/src/TH/generic/THStorage.h") . fst) $ files lib gentype)
    when (verbose os) $ putStrLn "Done"
  else putStrLn ("Code generation not enabled for " ++ show lib)
 where
  lib :: LibType
  lib = libraries os

  gentype :: CodeGenType
  gentype = codegenType os


runPipeline
  :: Options
  -> (String, TemplateType -> [Function] -> HModule)
  -> IO ()
runPipeline os (headerPath, makeModuleConfig) = do
  -- TODO: @nub@ is a hack until proper treatment of
  --       conditioned templates is implemented
  bindingsUniq <- nub <$> parseFile (libraries os) (codegenType os) headerPath

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



