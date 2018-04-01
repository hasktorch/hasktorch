module Torch.Indef.Dynamic.Tensor.Mode where

import Torch.Dimensions
import Torch.Class.Tensor.Mode
import qualified Torch.Sig.Tensor.Mode as Sig

import Torch.Indef.Types

instance TensorMode Dynamic where
  mode_ :: (Dynamic, IndexDynamic) -> Dynamic -> DimVal -> Maybe KeepDim -> IO ()
  mode_ (t0, ix) t1 i0 i1 = with2DynamicState t0 t1 $ \s' t0' t1' -> withIx ix $ \ix' ->
    Sig.c_mode s' t0' ix' t1' (fromIntegral i0) (fromKeepDim i1)



