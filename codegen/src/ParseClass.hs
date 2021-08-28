{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}

module ParseClass where

import Data.Yaml
import GHC.Generics
-- import Text.Show.Prettyprint (prettyPrint)
import qualified ParseFunctionSig as S

data CppClassSpec = CppClassSpec
  { signature :: String,
    cppname :: String,
    headers :: [String],
    hsname :: String,
    constructors :: [S.Function],
    methods :: [S.Function],
    functions :: [S.Function]
  }
  deriving (Show, Eq, Generic)

instance FromJSON CppClassSpec

-- decodeAndPrint :: String -> IO ()
-- decodeAndPrint fileName = do
--   file <- Y.decodeFileEither fileName :: IO (Either ParseException CppClassSpec)
--   prettyPrint file

trimSpace :: String -> String
trimSpace [] = []
trimSpace (' ' : xs) = trimSpace xs
trimSpace (x : xs) = x : trimSpace xs

hasSpace :: String -> Bool
hasSpace [] = False
hasSpace (' ' : _) = True
hasSpace (_ : xs) = hasSpace xs

hsnameWithoutSpace :: CppClassSpec -> String
hsnameWithoutSpace typ_ = trimSpace $ hsname typ_

hsnameWithParens :: CppClassSpec -> String
hsnameWithParens typ_ = if hasSpace name then "(" <> name <> ")" else name
  where
    name = hsname typ_
