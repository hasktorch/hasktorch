
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}

module ParseClass where

import GHC.Generics
import Data.Yaml

import qualified Data.Yaml as Y
import Text.Show.Prettyprint (prettyPrint)
import qualified ParseFunctionSig as S

data CppClassSpec = CppClassSpec
  { signature :: String
  , cppname :: String
  , hsname :: String
  , constructors :: [S.Function]
  , methods :: [S.Function]
  , functions :: [S.Function]
} deriving (Show, Eq, Generic)

instance FromJSON CppClassSpec


decodeAndPrint :: String -> IO ()
decodeAndPrint fileName = do
  file <- Y.decodeFileEither fileName :: IO (Either ParseException CppClassSpec)
  prettyPrint file
