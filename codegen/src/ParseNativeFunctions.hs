{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

module ParseNativeFunctions where

import Data.Char (toUpper)
import GHC.Generics
import Data.Yaml

import qualified Data.Yaml as Y
import Data.Aeson.Types (defaultOptions, fieldLabelModifier, genericParseJSON)
import qualified Language.C.Inline.Cpp as C
import qualified Language.C.Inline.Cpp.Exceptions as C
import Text.Show.Prettyprint (prettyPrint)
import qualified ParseFunctionSig as P
import Text.Megaparsec (parse, ParseErrorBundle)
import Data.Void (Void)
import Control.Monad (forM_)

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


data Dispatch = Dispatch {
  cpu :: Maybe String
  , gpu :: Maybe String
  , cuda :: Maybe String
  , sparseCPU :: Maybe String
  , sparseCUDA :: Maybe String
} deriving (Show, Generic)

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
parseNativeFunction nfunc@NativeFunction{..} =
  case parse P.func "" func of
    Right v -> Right $ NativeFunction' v variants python_module device_guard dispatch requires_tensor
    Left err -> Left err

decodeAndCodeGen :: String -> IO ()
decodeAndCodeGen fileName = do
  funcs <-
    Y.decodeFileEither fileName :: IO (Either ParseException [NativeFunction])
  case funcs of
    Left err -> print err
    Right funcs' ->
      forM_ (map parseNativeFunction funcs') $ \v -> do
        case v of
          Left err' -> print err'
          Right funcs'' -> print funcs''

