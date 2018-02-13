{-# LANGUAGE TypeSynonymInstances #-}
module Torch.Core.Tensor.Dynamic.Math
  ( module CCall
  ) where

import Torch.Class.C.Internal
import GHC.Int
import Torch.Class.C.Tensor.Math as CCall

import THTypes
import Foreign
import qualified Torch.Core.FloatTensor.Dynamic  as F
import qualified Torch.Core.ByteTensor.Dynamic   as B
-- import qualified Torch.Core.CharTensor.Dynamic   as C
import qualified Torch.Core.ShortTensor.Dynamic  as S
import qualified Torch.Core.IntTensor.Dynamic    as I
import qualified Torch.Core.DoubleTensor.Dynamic as D
-- import qualified Torch.Core.HalfTensor.Dynamic   as H


import qualified Torch.Core.FloatTensor.Dynamic.Math  as F
import qualified Torch.Core.ByteTensor.Dynamic.Math   as B
-- import qualified Torch.Core.CharTensor.Dynamic.Math   as C
import qualified Torch.Core.ShortTensor.Dynamic.Math  as S
import qualified Torch.Core.IntTensor.Dynamic.Math    as I
import qualified Torch.Core.DoubleTensor.Dynamic.Math as D
-- import qualified Torch.Core.HalfTensor.Dynamic.Math   as H



