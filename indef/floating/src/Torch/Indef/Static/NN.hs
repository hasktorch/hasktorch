module Torch.Indef.Static.NN where

import Torch.Dimensions

import qualified Torch.Class.NN as Dynamic
import qualified Torch.Class.NN.Static as Class

import Torch.Indef.Types

instance Class.NN Tensor where

