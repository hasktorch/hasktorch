{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}

module Main where

import qualified Options.Applicative as O
import qualified ParseFunctionSig as F
import qualified RenderDeclarations as RD
import qualified RenderTensor as RT

{- CLI options -}

data Options = Options
    { specFileDL :: !String
    , outputDir :: !String
    } deriving Show

optsParser :: O.ParserInfo Options
optsParser = O.info
  (O.helper <*> versionOption <*> programOptions)
  ( O.fullDesc <> O.progDesc "ffi codegen" <> O.header
    "codegen for hasktorch 0.0.2"
  )

versionOption :: O.Parser (a -> a)
versionOption =
  O.infoOption "0.0.2" (O.long "version" <> O.help "Show version")

programOptions :: O.Parser Options
programOptions =
  Options
    <$> O.strOption
          (  O.long "declaration-spec"
          <> O.short 'd'
          <> O.metavar "FILENAME"
          <> O.value "spec/Declarations.yaml"
          <> O.help "Specification file of Declarations"
          )
    <*> O.strOption
          (  O.long "output-dir"
          <> O.short 'o'
          <> O.metavar "DIRNAME"
          <> O.value "output"
          <> O.help "Output-directory"
          )

main = do
  opts <- O.execParser optsParser
  RD.decodeAndCodeGen (outputDir opts) (specFileDL opts)
  RT.tensorBuilder
  pure ()

