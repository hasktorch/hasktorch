{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}

module Main where

import GHC.Generics
import Data.Yaml

import qualified Data.Yaml as Y
import qualified Language.C.Inline.Cpp as C
import qualified Language.C.Inline.Cpp.Exceptions as C
import qualified Text.Megaparsec as M
import Text.Show.Prettyprint

{- native_functions.yaml -}

data Dispatch = Dispatch {
  cpu :: Maybe String -- FIXME: how to use generics with capital "CPU"?
  , gpu :: Maybe String -- FIXME: how to use generics with capital "GPU"?
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

instance FromJSON NativeFunction
instance FromJSON Dispatch

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

main :: IO ()
main = do
  file <- Y.decodeFileEither "spec/small_test.yaml" :: IO (Either ParseException [NativeFunction])
  -- putStrLn (maybe "Error" show file)
  -- putStrLn (show file)
  prettyPrint file

  putStrLn "Done"
