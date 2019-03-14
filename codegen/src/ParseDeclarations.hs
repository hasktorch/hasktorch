
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}

module ParseDeclarations where

import Data.Char (toUpper)
import GHC.Generics
import Data.Yaml

import qualified Data.Yaml as Y
import Data.Aeson.Types (defaultOptions, fieldLabelModifier, genericParseJSON)
import qualified Language.C.Inline.Cpp as C
import qualified Language.C.Inline.Cpp.Exceptions as C
import Text.Show.Prettyprint (prettyPrint)

{- Declarations.yaml -}
{- --A example--
- name: _th_set_
  matches_jit_signature: false
  schema_string: ''
  method_prefix_derived: ''
  arguments:
  - dynamic_type: Tensor
    name: self
    type: Tensor &
  - dynamic_type: Storage
    name: source
    type: Storage
  method_of:
  - Type
  - namespace
  mode: TH
  python_module: ''
  buffers: []
  returns:
  - dynamic_type: Tensor
    name: self
    type: Tensor &
  inplace: true
  is_factory_method: false
  abstract: true
  requires_tensor: false
  device_guard: false
  with_gil: false
  deprecated: false
-}

data Type = Type
  { name' :: String
  , dynamic_type' :: String
  , type' :: String
} deriving (Show, Generic)

data Mode
  = TH
  | THC
  deriving (Show, Generic)

data Declaration = Derivative
  { name :: String
  , matches_jit_signature :: Bool
  , schema_string :: [String]
  , method_prefix_derived :: [String]
  , arguments :: [Type]
  , method_of :: [String]
  , mode :: Mode
  , python_module :: String
  , buffers :: [String]
  , returns :: Type
  , inplace :: Bool
  , is_factory_method :: Bool
  , abstract :: Bool
  , requires_tensor :: Bool
  , device_guard :: Bool
  , with_gil :: Bool
  , deprecated :: Bool
} deriving (Show, Generic)

instance FromJSON Type where
    parseJSON = genericParseJSON defaultOptions{ fieldLabelModifier = reverse.(drop 1).reverse }

instance FromJSON Mode

instance FromJSON Declaration


decodeAndPrint :: String -> IO ()
decodeAndPrint fileName = do
  file <- Y.decodeFileEither fileName :: IO (Either ParseException [Declaration])
  prettyPrint file
