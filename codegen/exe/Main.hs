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

data Options
  = Options
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
  RC.decodeAndCodeGen (outputDir opts) "spec/cppclass/tensor.yaml"
  RC.decodeAndCodeGen (outputDir opts) "spec/cppclass/intarray.yaml"
  RC.decodeAndCodeGen (outputDir opts) "spec/cppclass/tensoroptions.yaml"
  RC.decodeAndCodeGen (outputDir opts) "spec/cppclass/generator.yaml"
  RC.decodeAndCodeGen (outputDir opts) "spec/cppclass/scalar.yaml"
  RC.decodeAndCodeGen (outputDir opts) "spec/cppclass/storage.yaml"
  RC.decodeAndCodeGen (outputDir opts) "spec/cppclass/tensorlist.yaml"
  RC.decodeAndCodeGen (outputDir opts) "spec/cppclass/context.yaml"
  RC.decodeAndCodeGen (outputDir opts) "spec/cppclass/constquantizerptr.yaml"
  RC.decodeAndCodeGen (outputDir opts) "spec/cppclass/dimname.yaml"
  RC.decodeAndCodeGen (outputDir opts) "spec/cppclass/dimnamelist.yaml"
  RC.decodeAndCodeGen (outputDir opts) "spec/cppclass/symbol.yaml"
  RC.decodeAndCodeGen (outputDir opts) "spec/cppclass/ivalue.yaml"
  RTL.decodeAndCodeGen (outputDir opts) (specFileDL opts)
  RD.decodeAndCodeGen (outputDir opts) (specFileDL opts)
  RP.decodeAndCodeGen (outputDir opts) (specFileDL opts) "spec/bindings.yaml"
  pure ()
