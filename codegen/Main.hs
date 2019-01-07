{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}

module Main where

import Data.Char (toUpper)
import GHC.Generics
import Data.Yaml

import qualified Options.Applicative as O
import qualified Data.Yaml as Y
import Data.Aeson.Types (defaultOptions, fieldLabelModifier, genericParseJSON)
import qualified Language.C.Inline.Cpp as C
import qualified Language.C.Inline.Cpp.Exceptions as C
import qualified Text.Megaparsec as M
import Text.Show.Prettyprint (prettyPrint)

{- native_functions.yaml -}

data Dispatch = Dispatch {
  cpu :: Maybe String
  , gpu :: Maybe String
  , cuda :: Maybe String
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

dispatchModifier fieldName
  | fieldName `elem` ["cpu", "gpu", "cuda"] = upper fieldName
  | otherwise = fieldName
  where upper = map toUpper

instance FromJSON NativeFunction
instance FromJSON Dispatch where
  parseJSON = genericParseJSON $
    defaultOptions {
      fieldLabelModifier = dispatchModifier
    }


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
optsParser = O.info
  (O.helper <*> versionOption <*> programOptions)
  ( O.fullDesc <> O.progDesc "ffi codegen" <> O.header
    "codegen for hasktorch 0.0.2"
  )

versionOption :: O.Parser (a -> a)
versionOption =
  O.infoOption "0.0.2" (O.long "version" <> O.help "Show version")

programOptions :: O.Parser Options
programOptions = Options <$> O.strOption
  (  O.long "spec-file"
  <> O.metavar "FILENAME"
  <> O.value "spec/small_test.yaml"
  <> O.help "Specification file"
  )

{- Execution -}


decodeAndPrint :: String -> IO ()
decodeAndPrint fileName = do
  file <-
    Y.decodeFileEither fileName :: IO (Either ParseException [NativeFunction])
  prettyPrint file

main :: IO ()
main = do
  opts <- O.execParser optsParser
  decodeAndPrint (specFile opts)
  putStrLn "Done"
