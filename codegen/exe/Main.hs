{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}

module Main where

import qualified Options.Applicative as O
import qualified RenderClass as RC
import qualified RenderDeclarations as RD
import qualified RenderPure as RP
import qualified RenderTuples as RTL

{- CLI options -}

data Options = Options
  { specFileDL :: !String,
    outputDir :: !String
  }
  deriving (Show)

optsParser :: O.ParserInfo Options
optsParser =
  O.info
    (O.helper <*> versionOption <*> programOptions)
    ( O.fullDesc <> O.progDesc "ffi codegen"
        <> O.header
          "codegen for hasktorch 0.0.2"
    )

versionOption :: O.Parser (a -> a)
versionOption =
  O.infoOption "0.0.2" (O.long "version" <> O.help "Show version")

programOptions :: O.Parser Options
programOptions =
  Options
    <$> O.strOption
      ( O.long "declaration-spec"
          <> O.short 'd'
          <> O.metavar "FILENAME"
          <> O.value "spec/Declarations.yaml"
          <> O.help "Specification file of Declarations"
      )
    <*> O.strOption
      ( O.long "output-dir"
          <> O.short 'o'
          <> O.metavar "DIRNAME"
          <> O.value "output"
          <> O.help "Output-directory"
      )

main :: IO ()
main = do
  opts <- O.execParser optsParser
  --  RT.tensorBuilder
  RC.decodeAndCodeGenForTensor (outputDir opts) "spec/cppclass/tensor.yaml" (specFileDL opts) 4
  --  RC.decodeAndCodeGen (outputDir opts) "spec/cppclass/tensor.yaml" 4
  RC.decodeAndCodeGen (outputDir opts) "spec/cppclass/intarray.yaml" 1
  RC.decodeAndCodeGen (outputDir opts) "spec/cppclass/tensoroptions.yaml" 1
  RC.decodeAndCodeGen (outputDir opts) "spec/cppclass/generator.yaml" 1
  RC.decodeAndCodeGen (outputDir opts) "spec/cppclass/scalar.yaml" 1
  RC.decodeAndCodeGen (outputDir opts) "spec/cppclass/storage.yaml" 1
  RC.decodeAndCodeGen (outputDir opts) "spec/cppclass/tensorlist.yaml" 1
  RC.decodeAndCodeGen (outputDir opts) "spec/cppclass/context.yaml" 1
  RC.decodeAndCodeGen (outputDir opts) "spec/cppclass/constquantizerptr.yaml" 1
  RC.decodeAndCodeGen (outputDir opts) "spec/cppclass/dimname.yaml" 1
  RC.decodeAndCodeGen (outputDir opts) "spec/cppclass/dimnamelist.yaml" 1
  RC.decodeAndCodeGen (outputDir opts) "spec/cppclass/symbol.yaml" 1
  RC.decodeAndCodeGen (outputDir opts) "spec/cppclass/ivalue.yaml" 1
  RTL.decodeAndCodeGen (outputDir opts) (specFileDL opts)
  RD.decodeAndCodeGen (outputDir opts) (specFileDL opts)
  RP.decodeAndCodeGen (outputDir opts) (specFileDL opts) "spec/bindings.yaml"
  pure ()
