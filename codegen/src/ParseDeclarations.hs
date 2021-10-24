{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}

module ParseDeclarations where

import Data.Aeson.Types (defaultOptions, fieldLabelModifier, genericParseJSON)
import Data.Yaml
import GHC.Generics
-- import Text.Show.Prettyprint (prettyPrint)
import qualified ParseFunctionSig as S

{- Declarations.yaml -}
{- --A example--
- name: _th_set_
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
  device_guard: false
  with_gil: false
  deprecated: false
-}

data Type = Type
  { name' :: String,
    dynamic_type' :: S.Parsable,
    type' :: String,
    size' :: Maybe Int,
    default' :: Maybe String
  }
  deriving (Show, Eq, Generic)

type2type :: Type -> S.Parsable
type2type typ =
  case dynamic_type' typ of
    S.TenType S.Scalar -> if type' typ == "Tensor" then S.TenType S.Tensor else S.TenType S.Scalar
    S.TenType (S.IntList s) ->
      case size' typ of
        Nothing -> S.TenType (S.IntList {S.dim = s})
        Just s' -> S.TenType (S.IntList {S.dim = Just [s']})
    a -> a

data Mode
  = TH
  | THC
  | NN
  | Native
  deriving (Show, Eq, Generic)

data Declaration = Declaration
  { name :: String,
    schema_string :: String,
    --  , method_prefix_derived :: String
    arguments :: [Type],
    method_of :: [String],
    mode :: Mode,
    python_module :: String,
    --  , buffers :: [String]
    returns :: [Type],
    inplace :: Bool,
    is_factory_method :: Maybe Bool,
    abstract :: Bool,
    device_guard :: Maybe Bool,
    with_gil :: Maybe Bool,
    deprecated :: Maybe Bool
  }
  deriving (Show, Eq, Generic)

instance FromJSON Type where
  parseJSON = genericParseJSON defaultOptions {fieldLabelModifier = reverse . (drop 1) . reverse}

instance FromJSON Mode where
  parseJSON (String "TH") = pure TH
  parseJSON (String "THC") = pure THC
  parseJSON (String "NN") = pure NN
  parseJSON (String "native") = pure Native
  parseJSON v = fail $ show v <> " is not a string of Mode."

instance FromJSON Declaration

-- decodeAndPrint :: String -> IO ()
-- decodeAndPrint fileName = do
--   file <- Y.decodeFileEither fileName :: IO (Either ParseException [Declaration])
--   prettyPrint file
