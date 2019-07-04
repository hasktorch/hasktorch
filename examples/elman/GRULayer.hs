{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FunctionalDependencies #-}

module GRULayer where

import Torch.Tensor
import Torch.DType
import Torch.TensorFactories
import Torch.Functions
import Torch.TensorOptions
import Torch.Autograd

import System.IO.Unsafe
import Control.Monad.State.Strict
import Data.List (foldl', scanl', intersperse)

import Parameters
import Utils

data GRUCellSpec

data GRUCell

instance Randomizable GRUCell

instance Parametrized GRUCell

instance Show GRUCell

instance Recurrent GRUCell where

  run 

  runOverTimesteps