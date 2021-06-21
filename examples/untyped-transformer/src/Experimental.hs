{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE RecordWildCards #-}

module Experimental where

import Torch as T
import GHC.Generics

data EmbeddingSpec =  EmbeddingSpec {
  numEmbeddings :: Int,
  embeddingDim :: Int
} deriving (Show, Eq)

data Embedding = Embedding {
  embScaleGradByFreq :: Bool,
  embIsSparse :: Bool,
  embPaddingIdx :: Int,
  embWeight :: Parameter 
} deriving (Show, Generic, Parameterized)

embeddingForward Embedding{..} = 
  embedding embScaleGradByFreq embIsSparse embWeight' embPaddingIdx
  where
    embWeight' = toDependent embWeight
