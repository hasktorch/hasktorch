{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FunctionalDependencies #-}

module LSTMLayer where

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


data LSTMCellSpec

data LSTMCell

instance Randomizable LSTMCell

instance Parametrized LSTMCell

instance Show LSTMCell

instance Recurrent LSTMCell where

  run 

  runOverTimesteps