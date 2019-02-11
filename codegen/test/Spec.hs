{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE ScopedTypeVariables #-}
module Main where

import Test.Hspec
import Control.Exception.Safe (throwString, throw)
import System.Directory (doesFileExist)
import qualified Data.Yaml as Y

import ParseNativeFunctions

main :: IO ()
main = hspec $ do
  describe "parsing native_functions.yaml" nativeFunctionsSpec

nativeFunctionsSpec :: Spec
nativeFunctionsSpec = do
  it "parses the same number of functions as a vanilla" $ do
    xs <- vanillaParse nativeFunctionsPath
    Y.decodeFileEither nativeFunctionsPath >>= \case
      Left exception -> throw exception
      Right (fs::[NativeFunction]) -> do
        (length fs) `shouldBe` (length xs)
 where
  nativeFunctionsPath :: FilePath
  nativeFunctionsPath = "spec/native_functions_modified.yaml"

vanillaParse :: FilePath -> IO [Y.Value]
vanillaParse fp = do
  doesFileExist fp >>= \case
    False -> throwString "Spec doesn't exist! Review README to get spec yaml"
    True -> Y.decodeFileThrow fp
