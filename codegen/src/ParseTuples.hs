{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}

module ParseTuples where

import Data.Yaml
import GHC.Generics
import qualified ParseFunctionSig as S

{- spec/tuples.yaml -}

data Tuple = Tuple
  { types :: [S.Parsable]
  }
  deriving (Show, Eq, Generic)

instance FromJSON Tuple
