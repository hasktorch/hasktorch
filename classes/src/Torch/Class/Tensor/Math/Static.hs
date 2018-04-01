module Torch.Class.Tensor.Math.Static where

import Torch.Dimensions
import Torch.Class.Types
import qualified Torch.Types.TH as TH (IndexStorage)

class TensorMath t where
  fill_        :: t d -> HsReal (t d) -> IO ()
  zero_        :: t d -> IO ()
  zeros_       :: t d -> IndexStorage (t d) -> IO ()
  zerosLike_   :: t d -> t d -> IO ()
  ones_        :: t d -> TH.IndexStorage -> IO ()
  onesLike_    :: t d -> t d -> IO ()
  numel        :: t d -> IO Integer
  reshape_     :: t d -> t d -> TH.IndexStorage -> IO ()
  cat_         :: t d -> t d -> t d -> DimVal -> IO ()
  catArray_    :: t d -> [AsDynamic (t d)] -> Int -> DimVal -> IO ()
  nonzero_     :: IndexTensor (t d) d -> t d -> IO ()
  tril_        :: t d -> t d -> Integer -> IO ()
  triu_        :: t d -> t d -> Integer -> IO ()
  diag_        :: t d -> t d -> Int -> IO ()
  eye_         :: t d -> Integer -> Integer -> IO ()
  trace        :: t d -> IO (HsAccReal (t d))
  arange_      :: t d -> HsAccReal (t d) -> HsAccReal (t d) -> HsAccReal (t d) -> IO ()
  range_       :: t d -> HsAccReal (t d) -> HsAccReal (t d) -> HsAccReal (t d) -> IO ()


