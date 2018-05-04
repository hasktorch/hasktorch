module Torch.Indef.Static.NN.Conv where

import qualified Torch.Class.NN as Dynamic
import qualified Torch.Class.NN.Static.Conv as Class

import Torch.Indef.Types
import Torch.Indef.Dynamic.NN ()
import Torch.Indef.Static.Tensor ()

instance Class.TemporalConvolutions Tensor where
  _temporalConvolution_updateOutput i o w b = Dynamic.temporalConvolution_updateOutput (asDynamic i) (asDynamic o) (asDynamic w) (asDynamic b)
  _temporalConvolution_updateGradInput a0 a1 a2 a3 = Dynamic.temporalConvolution_updateGradInput (asDynamic a0) (asDynamic a1) (asDynamic a2) (asDynamic a3)



