{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}

module ParseNN where

import GHC.Generics
import Data.Yaml

import Data.Aeson.Types ()
import qualified ParseFunctionSig as P
import qualified Data.List as L
import Text.Megaparsec (parse, errorBundlePretty, ParseErrorBundle)
import Data.Void (Void)

{- nn.yaml -}

data ScalarCheck = ScalarCheck {
  output :: Maybe String
  , grad_input :: Maybe String
  , is_target :: Maybe String
  , total_weight :: Maybe String
  , buffer :: Maybe String
  , self :: Maybe String
} deriving (Show, Generic)

data WrapDim = WrapDim {
  dim :: String
} deriving (Show, Generic)

data DefaultInit = DefaultInit {
  stride :: String
} deriving (Show, Generic)

data NN = NN {
  name :: String
  , cname :: String
  , scalar_check :: Maybe ScalarCheck
  , buffers :: Maybe [String]
  , has_inplace :: Maybe Bool
  , wrap_dim :: Maybe WrapDim
  , default_init :: Maybe DefaultInit
} deriving (Show, Generic)

data NN' = NN' {
  func' :: P.Function
  , cname' :: String
  , scalar_check' :: Maybe ScalarCheck
  , buffers' :: Maybe [String]
  , has_inplace' :: Maybe Bool
  , wrap_dim' :: Maybe WrapDim
  , default_init' :: Maybe DefaultInit
} deriving (Show, Generic)

instance FromJSON ScalarCheck
instance FromJSON WrapDim
instance FromJSON DefaultInit
instance FromJSON NN

nnWithReturnType :: NN -> String
nnWithReturnType nn' = ((name nn') ++ ret_types nn')
  where
    ret_types nn =
      case (buffers nn) of
        Just [] -> " -> Tensor"
        Just v' -> " -> (" <> L.intercalate ", " (take (1 + length v') (repeat "Tensor")) <> ")"
        Nothing -> " -> Tensor"

parseNN :: NN -> Either (ParseErrorBundle String Void) P.Function
parseNN nn = parse P.func "" (nnWithReturnType nn)

instance FromJSON NN' where
  parseJSON v = do
    nn :: NN <- parseJSON v
    case parseNN nn of
      Left err -> fail (errorBundlePretty err)
      Right f ->
        pure $ NN' f
          (cname nn)
          (scalar_check nn)
          (buffers nn)
          (has_inplace nn)
          (wrap_dim nn)
          (default_init nn)
