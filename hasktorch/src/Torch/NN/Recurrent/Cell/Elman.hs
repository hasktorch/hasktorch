{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Torch.NN.Recurrent.Cell.Elman where

import Control.Monad.State.Strict
import Data.List (foldl', scanl', intersperse)
import GHC.Generics

import Torch
import Torch.NN.Recurrent.Cell.RecurrentLayer

data ElmanSpec = ElmanSpec { 
    iSize :: Int,
    hSize :: Int 
} deriving (Eq, Show)

data ElmanCell = ElmanCell {
    ihWeights :: Parameter,
    hhWeights :: Parameter,
    ihBias :: Parameter,
    hhBias :: Parameter
} deriving (Generic, Show)

elman ElmanCell{..} input hidden =
    rnnReluCell ihWeights' hhWeights' ihBias' hhBias' hidden input
    where
        ihWeights' = toDependent ihWeights
        hhWeights' = toDependent hhWeights
        ihBias' = toDependent ihBias
        hhBias' = toDependent ihBias

instance Randomizable ElmanSpec ElmanCell where
    sample ElmanSpec{..} = do
      ihWeights <- makeIndependent =<< randnIO' [hSize, iSize]
      hhWeights <- makeIndependent =<< randnIO' [hSize, hSize]
      ihBias <- makeIndependent =<< randnIO' [hSize]
      hhBias <- makeIndependent =<< randnIO' [hSize]
      return $ ElmanCell ihWeights hhWeights ihBias hhBias
