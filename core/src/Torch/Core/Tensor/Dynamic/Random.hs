-- reexport instances, typeclass, and generalized functions from classes-c
module Torch.Core.Tensor.Dynamic.Random
  ( module Random
  ) where

import Torch.Class.Tensor.Random as Random

import Torch.Core.ByteTensor.Dynamic.Random   ()
-- import Torch.Core.CharTensor.Dynamic.Random   ()
import Torch.Core.ShortTensor.Dynamic.Random  ()
import Torch.Core.IntTensor.Dynamic.Random    ()
import Torch.Core.LongTensor.Dynamic.Random   ()
-- import Torch.Core.HalfTensor.Dynamic          ()
import Torch.Core.FloatTensor.Dynamic.Random  ()
import Torch.Core.DoubleTensor.Dynamic.Random ()

import Torch.Core.FloatTensor.Dynamic.Random.Floating  ()
import Torch.Core.DoubleTensor.Dynamic.Random.Floating ()
