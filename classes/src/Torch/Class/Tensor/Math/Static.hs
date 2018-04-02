module Torch.Class.Tensor.Math.Static where

import GHC.Int
import Torch.Dimensions
import Torch.Class.Types
import Torch.Class.Tensor.Static
import qualified Torch.Types.TH as TH (IndexStorage)

class Tensor t => TensorMath t where
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

class TensorMathFloating t where
  linspace_ :: Dimensions d => t d -> HsReal (t d) -> HsReal (t d) -> Int64 -> IO ()
  logspace_ :: Dimensions d => t d -> HsReal (t d) -> HsReal (t d) -> Int64 -> IO ()

constant :: (TensorMath t, Tensor t, Dimensions d) => HsReal (t d) -> IO (t d)
constant v = do
  t <- new
  fill_ t v
  pure t

_tenLike
  :: (Dimensions d, TensorMath t)
  => (t d -> t d -> IO ())
  -> IO (t d)
_tenLike fn_ = do
  src <- new
  shape <- new
  fn_ src shape
  pure src

onesLike, zerosLike :: (Dimensions d, Tensor t, TensorMath t) => IO (t d)
onesLike = _tenLike onesLike_
zerosLike = _tenLike zerosLike_



