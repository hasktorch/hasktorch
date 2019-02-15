{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE QuasiQuotes #-}
module ParseNativeFunctions where

import Data.Char (toUpper)
import GHC.Generics
import Data.Yaml

import qualified Data.Yaml as Y
import Data.Aeson.Types (defaultOptions, fieldLabelModifier, genericParseJSON)
import Text.Show.Prettyprint (prettyPrint)
import qualified ParseFunctionSig as P
import Text.Megaparsec (parse, ParseErrorBundle, errorBundlePretty)
import Data.Void (Void)

{- native_functions_modified.yaml -}

data NativeFunction = NativeFunction {
  func :: String
  , variants :: Maybe String
  , python_module :: Maybe String
  , device_guard :: Maybe Bool
  , dispatch :: Maybe Dispatch
  , requires_tensor :: Maybe Bool
} deriving (Show, Generic)


data NativeFunction' = NativeFunction' {
  func' :: P.Function
  , variants' :: Maybe String
  , python_module' :: Maybe String
  , device_guard' :: Maybe Bool
  , dispatch' :: Maybe Dispatch
  , requires_tensor' :: Maybe Bool
} deriving (Show, Generic)

instance FromJSON NativeFunction' where
  parseJSON v = do
    nf :: NativeFunction <- parseJSON v
    case parse P.func "" (func nf) of
      Left err -> fail (errorBundlePretty err)
      Right f ->
        pure $ NativeFunction' f
          (variants nf)
          (python_module nf)
          (device_guard nf)
          (dispatch nf)
          (requires_tensor nf)


data Dispatch = Dispatch {
  cpu :: Maybe String
  , gpu :: Maybe String
  , cuda :: Maybe String
  , sparseCPU :: Maybe String
  , sparseCUDA :: Maybe String
} deriving (Show, Generic)

dispatchModifier :: [Char] -> [Char]
dispatchModifier fieldName
  | fieldName `elem` ["cpu", "gpu", "cuda"] = upper fieldName
  | fieldName == "sparseCPU"  = "SparseCPU"
  | fieldName == "sparseGPU"  = "SparseGPU"
  | fieldName == "sparseCUDA" = "SparseCUDA"
  | otherwise                 = fieldName
  where upper = map toUpper

instance FromJSON NativeFunction

instance FromJSON Dispatch where
  parseJSON = genericParseJSON $
    defaultOptions {
      fieldLabelModifier = dispatchModifier
    }

instance ToJSON Dispatch

{- Execution -}

decodeAndPrint :: String -> IO ()
decodeAndPrint fileName = do
  file <-
    Y.decodeFileEither fileName :: IO (Either ParseException [NativeFunction])
  prettyPrint file


parseNativeFunction :: NativeFunction -> Either (ParseErrorBundle String Void) NativeFunction'
parseNativeFunction NativeFunction{..} =
  case parse P.func "" func of
    Right v -> Right $ NativeFunction' v variants python_module device_guard dispatch requires_tensor
    Left err -> Left err
