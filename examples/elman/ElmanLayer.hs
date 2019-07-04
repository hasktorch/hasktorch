{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FunctionalDependencies #-}

module ElmanLayer where

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

data ElmanCellSpec

data ElmanCell

instance Randomizable ElmanCell

instance Parametrized ElmanCell

instance Show ElmanCell

instance Recurrent ElmanCell where

  run 

  runOverTimesteps