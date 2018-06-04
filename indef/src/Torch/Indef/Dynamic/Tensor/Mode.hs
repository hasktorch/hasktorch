module Torch.Indef.Dynamic.Tensor.Mode where

import qualified Torch.Sig.Tensor.Mode as Sig
import qualified Torch.Indef.Index as Ix

import Torch.Indef.Types

_mode :: (Dynamic, IndexDynamic) -> Dynamic -> DimVal -> Maybe KeepDim -> IO ()
_mode (t0, ix) t1 i0 i1 = with2DynamicState t0 t1 $ \s' t0' t1' ->
  Ix.withDynamicState ix $ \_ ix' ->
    Sig.c_mode s' t0' ix' t1' (fromIntegral i0) (fromKeepDim i1)


