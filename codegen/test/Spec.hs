{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}

module Main where

import Control.Exception.Safe (throw, throwString)
import Data.Proxy
import qualified Data.Yaml as Y
import ParseDeclarations (Declaration)
import System.Directory (doesFileExist)
import Test.Hspec

main :: IO ()
main = hspec $ do
  describe "parsing Declarations.yaml" $ do
    describe "Declarations Spec" declarationsSpec

vanillaParse :: FilePath -> IO [Y.Value]
vanillaParse fp = do
  doesFileExist fp >>= \case
    False -> throwString $ "Spec " ++ fp ++ " doesn't exist! Review README to get spec yaml"
    True -> Y.decodeFileThrow fp

declarationsPath :: FilePath
declarationsPath = "../spec/Declarations.yaml"

declarationsSpec :: Spec
declarationsSpec = do
  it "parses the same number of stringy functions as a vanilla parsing" $ do
    yamlFile <-
      doesFileExist declarationsPath >>= \case
        False -> do
          print $ "*** Spec " ++ declarationsPath ++ " doesn't exist! Review README to get spec yaml ***"
          return "test/Declarations.yaml"
        True -> return declarationsPath
    xs <- vanillaParse yamlFile
    fs <- parseWith yamlFile (Proxy @Declaration)
    (length fs) `shouldBe` (length xs)
  where
    parseWith :: forall funtype. Y.FromJSON funtype => FilePath -> Proxy funtype -> IO [funtype]
    parseWith yamlFile _ = do
      Y.decodeFileEither yamlFile >>= \case
        Left exception -> throw exception
        Right (fs :: [funtype]) -> pure fs
