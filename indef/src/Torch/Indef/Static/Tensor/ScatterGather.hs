module Torch.Indef.Static.Tensor.ScatterGather where


import Torch.Indef.Types
import qualified Torch.Indef.Dynamic.Tensor.ScatterGather as Dynamic

_gather :: Tensor d -> Tensor d -> DimVal -> IndexTensor '[n] -> IO ()
_gather r src d ix = Dynamic._gather (asDynamic r) (asDynamic src) d (longAsDynamic ix)

_scatter :: Tensor d -> DimVal -> IndexTensor '[n] -> Tensor d -> IO ()
_scatter r d ix src = Dynamic._scatter (asDynamic r) d (longAsDynamic ix) (asDynamic src)

_scatterAdd   :: Tensor d -> DimVal -> IndexTensor '[n] -> Tensor d -> IO ()
_scatterAdd r d ix src = Dynamic._scatterAdd (asDynamic r) d (longAsDynamic ix) (asDynamic src)

_scatterFill  :: Tensor d -> DimVal -> IndexTensor '[n] -> HsReal -> IO ()
_scatterFill r d ix = Dynamic._scatterFill (asDynamic r) d (longAsDynamic ix)
