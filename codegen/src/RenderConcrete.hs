{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

module Main where

import Prelude as P
import Data.Text
import Data.List (nub)
import Text.Megaparsec

import CodeGenParse
import CodeGenTypes
import ConditionalCases
import RenderShared
import Text.Show.Pretty

outDirConcrete = "./output/raw/src/" :: Text

-- TODO re-factor to unify w/ parseFile
parseFileConcrete :: [Char] -> IO [THFunction]
parseFileConcrete file = do
  putStrLn $ "\nParsing " ++ file ++ " ... "
  res <- parseFromFile thParseConcrete file
  pure $ cleanList res
  where
    parseFromFile p file = runParser p file <$> readFile file

-- TODO re-factor to unify w/ runPipeline
runPipelineConcrete ::
  [Char] -> (TemplateType -> [THFunction] -> HModule) -> [TemplateType]-> IO ()
runPipelineConcrete headerPath makeModuleConfig typeList = do
  parsedBindings <- parseFileConcrete headerPath
  let bindingsUniq = nub parsedBindings
  -- TODO nub is a hack until proper treatment of conditioned templates is implemented
  putStrLn $ "First signature:"
  putStrLn $ ppShow (P.take 1 bindingsUniq)
  mapM_ (\x -> renderCHeaderFile x bindingsUniq makeModuleConfig) typeList
  putStrLn $ "Number of functions generated: " ++
    (show $ P.length typeList * P.length bindingsUniq)

concreteFiles :: [(String, TemplateType -> [THFunction] -> HModule)]
concreteFiles =
  [

    -- ("vendor/scratch.h",
    --  (makeModule outDirConcrete False "scratch.h" "Scratch" "Scratch"))

    ("vendor/TH/THFile.h",
     (makeModule outDirConcrete False "THFile.h" "File" "File")),
    ("vendor/TH/THDiskFile.h",
     (makeModule outDirConcrete False "THDiskFile.h" "DiskFile" "DiskFile")),
    ("vendor/TH/THAtomic.h",
     (makeModule outDirConcrete False "THDiskFile.h" "Atomic" "Atomic")),
    ("vendor/TH/THHalf.h",
     (makeModule outDirConcrete False "THHalf.h" "Half" "Half")),
    ("vendor/TH/THLogAdd.h",
     (makeModule outDirConcrete False "THLogAdd.h" "LogAdd" "LogAdd")),
    ("vendor/TH/THRandom.h",
     (makeModule outDirConcrete False "THRandom.h" "Random" "Random")),
    ("vendor/TH/THSize.h",
     (makeModule outDirConcrete False "THSize.h" "Size" "Size")),
    ("vendor/TH/THStorage.h",
     (makeModule outDirConcrete False "THStorage.h" "Storage" "Storage")),
    ("vendor/TH/THMemoryFile.h",
     (makeModule outDirConcrete False "THMemoryFile.h" "MemoryFile" "MemoryFile"))

  ]

main :: IO ()
main = do
  mapM_ (\(file, spec) -> runPipelineConcrete file spec concreteTypes) concreteFiles
  putStrLn "Done"
