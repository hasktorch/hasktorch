{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
module Main where

import Control.Exception.Safe (throwString, throw)
import Data.Proxy
import Text.Megaparsec (parse, errorBundlePretty)
import ParseDerivatives (Derivative)
import ParseDeclarations (Declaration)
import System.Directory (doesFileExist)
import Test.Hspec
import qualified Data.Yaml as Y
import ParseFunctionSig

main :: IO ()
main = hspec $ do
  describe "parsing derivatives.yaml" $ do
    describe "Derivatives Spec" derivativesSpec
  describe "parsing Declarations.yaml" $ do
    describe "Declarations Spec" declarationsSpec


derivativesPath :: FilePath
derivativesPath = "../deps/pytorch/tools/autograd/derivatives.yaml"

derivativesSpec :: Spec
derivativesSpec = do
  xs <- runIO $ vanillaParse derivativesPath

  it "parses the same number of stringy functions as a vanilla parsing" $ do
    fs <- parseWith (Proxy @ Derivative)
    (length fs) `shouldBe` (length xs)

 where
  parseWith :: forall funtype . Y.FromJSON funtype => Proxy funtype -> IO [funtype]
  parseWith _ = do
    Y.decodeFileEither derivativesPath >>= \case
      Left exception -> throw exception
      Right (fs::[funtype]) -> pure fs

vanillaParse :: FilePath -> IO [Y.Value]
vanillaParse fp = do
  doesFileExist fp >>= \case
    False -> throwString $ "Spec " ++ fp ++ " doesn't exist! Review README to get spec yaml"
    True -> Y.decodeFileThrow fp


declarationsPath :: FilePath
declarationsPath = "../spec/Declarations.yaml"

declarationsSpec :: Spec
declarationsSpec = do
  xs <- runIO $ vanillaParse declarationsPath

  it "parses the same number of stringy functions as a vanilla parsing" $ do
    fs <- parseWith (Proxy @ Declaration)
    (length fs) `shouldBe` (length xs)

 where
  parseWith :: forall funtype . Y.FromJSON funtype => Proxy funtype -> IO [funtype]
  parseWith _ = do
    Y.decodeFileEither declarationsPath >>= \case
      Left exception -> throw exception
      Right (fs::[funtype]) -> pure fs

