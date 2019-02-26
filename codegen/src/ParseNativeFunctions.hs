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
import Data.Aeson ((.:!))
import Data.String.Conversions (cs)


import qualified Data.Yaml as Y
import Data.Aeson.Types (defaultOptions, fieldLabelModifier, genericParseJSON)
import Text.Show.Prettyprint (prettyPrint)
import qualified ParseFunctionSig as P
import Text.Megaparsec (parse, errorBundlePretty)

{- native_functions_modified.yaml -}

data NativeFunction = NativeFunction {
  func :: String
  , variants :: Maybe String
  , python_module :: Maybe String
  , device_guard :: Maybe Bool
  , dispatch :: Maybe Dispatch
  , requires_tensor :: Maybe Bool
  , matches_jit_signature :: Maybe Bool
} deriving (Show, Generic)


data NativeFunction' = NativeFunction' {
  func' :: P.Function
  , variants' :: Maybe String
  , python_module' :: Maybe String
  , device_guard' :: Maybe Bool
  , dispatch' :: Maybe Dispatch
  , requires_tensor' :: Maybe Bool
  , matches_jit_signature' :: Maybe Bool
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
          (matches_jit_signature nf)


data Dispatch = Dispatch {
  cpu :: Maybe String
  , cuda :: Maybe String
  , sparseCPU :: Maybe String
  , sparseCUDA :: Maybe String
} deriving (Show, Generic)

instance FromJSON NativeFunction

instance FromJSON Dispatch where
  parseJSON (Object v) =  Dispatch <$> v .:! "CPU" <*> v .:! "CUDA" <*> v .:! "SparseCPU" <*> v .:! "SparseCUDA"
  parseJSON (String v) =  pure $ Dispatch (Just (cs v)) Nothing Nothing Nothing

instance ToJSON Dispatch

{- Execution -}

decodeAndPrint :: String -> IO ()
decodeAndPrint fileName = do
  file <-
    Y.decodeFileEither fileName :: IO (Either ParseException [NativeFunction])
  prettyPrint file

