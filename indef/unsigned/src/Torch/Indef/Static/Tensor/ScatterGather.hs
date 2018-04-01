module Torch.Indef.Static.Tensor.ScatterGather where

import Torch.Dimensions
import qualified Torch.Class.Tensor.ScatterGather.Static as Class
import qualified Torch.Class.Tensor.ScatterGather as Dynamic

import Torch.Indef.Types
import Torch.Indef.Dynamic.Tensor.ScatterGather ()

instance Class.TensorScatterGather Tensor where
  gather_ :: Tensor d -> Tensor d -> DimVal -> IndexTensor '[n] -> IO ()
  gather_ r src d ix = Dynamic.gather_ (asDynamic r) (asDynamic src) d (longAsDynamic ix)

  scatter_ :: Tensor d -> DimVal -> IndexTensor '[n] -> Tensor d -> IO ()
  scatter_ r d ix src = Dynamic.scatter_ (asDynamic r) d (longAsDynamic ix) (asDynamic src)

  scatterAdd_   :: Tensor d -> DimVal -> IndexTensor '[n] -> Tensor d -> IO ()
  scatterAdd_ r d ix src = Dynamic.scatterAdd_ (asDynamic r) d (longAsDynamic ix) (asDynamic src)

  scatterFill_  :: Tensor d -> DimVal -> IndexTensor '[n] -> HsReal -> IO ()
  scatterFill_ r d ix = Dynamic.scatterFill_ (asDynamic r) d (longAsDynamic ix)
