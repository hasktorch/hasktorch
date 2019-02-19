
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}

module ParseNN where

import GHC.Generics
import Data.Yaml

import qualified Data.Yaml as Y
import Data.Aeson.Types ()
import Text.Show.Prettyprint (prettyPrint)

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

instance FromJSON ScalarCheck
instance FromJSON WrapDim
instance FromJSON DefaultInit
instance FromJSON NN

decodeAndPrint :: String -> IO ()
decodeAndPrint fileName = do
  file <- Y.decodeFileEither fileName :: IO (Either ParseException [NN])
  prettyPrint file
