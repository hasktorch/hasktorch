
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}

module ParseDerivatives where

import GHC.Generics
import Data.Yaml

import qualified Data.Yaml as Y
import Text.Show.Prettyprint (prettyPrint)

{- derivatives.yaml -}

data Derivative = Derivative {
  name :: String
  , self :: Maybe String
  , other :: Maybe String
  , tensor1 :: Maybe String
  , tensor2 :: Maybe String
  , tensors :: Maybe String
  , mat1 :: Maybe String
  , mat2 :: Maybe String
  , vec :: Maybe String
  , batch1 :: Maybe String
  , batch2 :: Maybe String
  , output_differentiability :: Maybe [Bool]
  , value :: Maybe String
  , exponent :: Maybe String
  , src :: Maybe String
  , grad_output :: Maybe String
  , weight :: Maybe String
  , bias :: Maybe String
  , input :: Maybe String
  , input2 :: Maybe String
  , input3 :: Maybe String
  , input_gates :: Maybe String
  , input_bias :: Maybe String
  , hidden_gates :: Maybe String
  , hidden_bias :: Maybe String
  , cx :: Maybe String
  , hx :: Maybe String
  , save_mean :: Maybe String
  , save_var :: Maybe String
  , grid :: Maybe String
  , i1 :: Maybe String
  , i2 :: Maybe String
  , i3 :: Maybe String
} deriving (Show, Generic)

instance FromJSON Derivative

decodeAndPrint :: String -> IO ()
decodeAndPrint fileName = do
  file <- Y.decodeFileEither fileName :: IO (Either ParseException [Derivative])
  prettyPrint file
