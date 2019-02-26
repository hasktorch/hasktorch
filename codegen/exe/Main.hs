{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}

module Main where

import qualified Options.Applicative as O
import qualified ParseNativeFunctions as NF
import qualified ParseDerivatives as D
import qualified ParseFunctionSig as F
import qualified RenderNativeFunctions as RNF
import qualified RenderNN as RNN

{- CLI options -}

data Options = Options
    { specFileNF :: !String
    , specFileNN :: !String
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
          (  O.long "nf-spec"
          <> O.short 'f'
          <> O.metavar "FILENAME"
          <> O.value "spec/native_functions_modified.yaml"
          <> O.help "Specification file of native-functions"
          )
    <*> O.strOption
          (  O.long "nn-spec"
          <> O.short 'n'
          <> O.metavar "FILENAME"
          <> O.value "spec/nn.yaml"
          <> O.help "Specification file of nn"
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
  RNF.decodeAndCodeGen (outputDir opts) (specFileNF opts)
  RNN.decodeAndCodeGen (outputDir opts) (specFileNN opts)
  pure ()

