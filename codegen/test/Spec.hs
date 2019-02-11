{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
module Main where

import Control.Exception.Safe (throwString, throw)
import Data.Proxy
import ParseNativeFunctions (NativeFunction, NativeFunction')
import System.Directory (doesFileExist)
import Test.Hspec
import qualified Data.Yaml as Y

main :: IO ()
main = hspec $
  describe "parsing native_functions.yaml" $ do
    describe "NativeFunction Spec" nativeFunctionsSpec

nativeFunctionsPath :: FilePath
nativeFunctionsPath = "../spec/native_functions_modified.yaml"

nativeFunctionsSpec :: Spec
nativeFunctionsSpec = do
  xs <- runIO $ vanillaParse nativeFunctionsPath

  it "parses the same number of stringy functions as a vanilla parsing" $
     testit xs (Proxy @ NativeFunction)

  it "parses the same number of typed functions as a vanilla parsing" $
     testit xs (Proxy @ NativeFunction')

 where
  testit :: forall x funtype . Y.FromJSON funtype => [x] -> Proxy funtype -> IO ()
  testit xs _ = do
    print (length xs)
    Y.decodeFileEither nativeFunctionsPath >>= \case
      Left exception -> throw exception
      Right (fs::[funtype]) -> do
        (length fs) `shouldBe` (length xs)

vanillaParse :: FilePath -> IO [Y.Value]
vanillaParse fp = do
  doesFileExist fp >>= \case
    False -> throwString "Spec doesn't exist! Review README to get spec yaml"
    True -> Y.decodeFileThrow fp
