{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}

module ParseNativeFunctions where

import Data.Char (toUpper)
import GHC.Generics
import Data.Yaml

import qualified Data.Yaml as Y
import Data.Aeson.Types (defaultOptions, fieldLabelModifier, genericParseJSON)
import qualified Language.C.Inline.Cpp as C
import qualified Language.C.Inline.Cpp.Exceptions as C
import qualified Text.Megaparsec as M
import Text.Show.Prettyprint (prettyPrint)

{- native_functions_modified.yaml -}

data NativeFunction = NativeFunction {
  func :: String
  , variants :: Maybe String
  , python_module :: Maybe String
  , device_guard :: Maybe Bool
  , dispatch :: Maybe Dispatch
  , requires_tensor :: Maybe Bool
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


{- derivatives.yaml -}

data Derivative = Derivative {
  name :: String
  , grad_output :: Maybe String
  , output_differentiability :: [Bool]
  , self :: Maybe String
  , tensors :: Maybe String

} deriving (Show, Generic)

instance FromJSON Derivative

{- Execution -}


decodeAndPrint :: String -> IO ()
decodeAndPrint fileName = do
  file <-
    Y.decodeFileEither fileName :: IO (Either ParseException [NativeFunction])
  prettyPrint file