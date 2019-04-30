{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}

module ParseTuples where

import GHC.Generics
import Data.Yaml

import qualified ParseFunctionSig as S

{- spec/tuples.yaml -}

data Tuple = Tuple {
  types :: [S.Parsable]
} deriving (Show, Eq, Generic)

instance FromJSON Tuple

