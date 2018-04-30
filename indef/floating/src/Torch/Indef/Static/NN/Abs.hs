module Torch.Indef.Static.NN.Abs where

import qualified Torch.Class.NN as Dynamic
import qualified Torch.Class.NN.Static.Abs as Class

import Torch.Indef.Types
import Torch.Indef.Dynamic.NN ()
import Torch.Indef.Static.Tensor ()

instance Class.Abs Tensor where
  _abs_updateOutput i o = Dynamic.abs_updateOutput (asDynamic i) (asDynamic o)
  _abs_updateGradInput i go gi = Dynamic.abs_updateGradInput (asDynamic i) (asDynamic go) (asDynamic gi)


