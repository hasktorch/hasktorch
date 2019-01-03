{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}

module Main where

import GHC.Generics
import Data.Yaml

import qualified Options.Applicative as O
import qualified Data.Yaml as Y
import qualified Language.C.Inline.Cpp as C
import qualified Language.C.Inline.Cpp.Exceptions as C
import qualified Text.Megaparsec as M
import Text.Show.Prettyprint (prettyPrint)

{- native_functions.yaml -}

data Dispatch = Dispatch {
  cpu :: Maybe String -- FIXME: how to use generics with capital "CPU"?
  , gpu :: Maybe String -- FIXME: how to use generics with capital "GPU"?
  , sparseCPU :: Maybe String
  , sparseCUDA :: Maybe String
} deriving (Show, Generic)

data NativeFunction = NativeFunction {
  func :: String
  , variants :: Maybe String
  , python_module :: Maybe String
  , device_guard :: Maybe Bool
  , dispatch :: Maybe Dispatch
  , requires_tensor :: Maybe Bool
} deriving (Show, Generic)

instance FromJSON NativeFunction
instance FromJSON Dispatch

{- derivatives.yaml -}

data Derivative = Derivative {
  name :: String
  , grad_output :: Maybe String
  , output_differentiability :: [Bool]
  , self :: Maybe String
  , tensors :: Maybe String

} deriving (Show, Generic)

instance FromJSON Derivative

{- CLI options -}

data Options = Options
    { specFile :: !String
    } deriving Show

optsParser :: O.ParserInfo Options
optsParser =
  O.info
    (O.helper <*> versionOption <*> programOptions)
    (O.fullDesc <> O.progDesc "ffi codegen" <>
    O.header
    "codegen for hasktorch 0.0.2")

versionOption :: O.Parser (a -> a)
versionOption = O.infoOption "0.0.2" (O.long "version" <> O.help "Show version")

programOptions :: O.Parser Options
programOptions =
  Options <$> 
    O.strOption
    (O.long "spec-file" <> O.metavar "FILENAME" <> O.value "spec/small_test.yaml" <>
    O.help "Specification file")

{- Execution -}

main :: IO ()
main = do
  opts <- O.execParser optsParser
  file <- Y.decodeFileEither (specFile opts) :: IO (Either ParseException [NativeFunction])
  prettyPrint file

  putStrLn "Done"
