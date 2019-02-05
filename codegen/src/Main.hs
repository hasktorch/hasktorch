{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}

module Main where

import qualified Options.Applicative as O
import qualified ParseNativeFunctions as NF
import qualified ParseDerivatives as D
import qualified ParseFunctionSig as F

{- CLI options -}

data Options = Options
    { specFile :: !String
      , mode :: !String
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
          (  O.long "spec-file"
          <> O.short 'f'
          <> O.metavar "FILENAME"
          <> O.value "spec/native_functions_modified.yaml"
          <> O.help "Specification file"
          )
    <*> O.strOption
          (  O.long "mode"
          <> O.short 'm'
          <> O.metavar "MODE"
          <> O.value "native-functions"
          <> O.help "native-functions or derivatives"
          )

main = do
  opts <- O.execParser optsParser
  NF.decodeAndCodeGen (specFile opts)
  pure ()

