{-# LANGUAGE DataKinds #-}
module Torch.Class.Tensor.Math where

import Foreign hiding (new)
import Foreign.C.Types
import GHC.TypeLits (Nat)
import Torch.Dimensions
import Torch.Class.Types
import GHC.Int
import Torch.Class.Tensor (IsTensor(empty), withInplace, new)
import qualified Torch.Types.TH as TH

class IsTensor t => TensorMath t where
  fill_        :: t -> HsReal t -> IO ()
  zero_        :: t -> IO ()
  zeros_       :: t -> IndexStorage t -> IO ()
  zerosLike_   :: t -> t -> IO ()
  ones_        :: t -> TH.IndexStorage -> IO ()
  onesLike_    :: t -> t -> IO ()
  numel        :: t -> IO Integer
  reshape_     :: t -> t -> TH.IndexStorage -> IO ()
  cat_         :: t -> t -> t -> DimVal -> IO ()
  catArray_    :: t -> [t] -> Int -> DimVal -> IO ()
  nonzero_     :: IndexDynamic t -> t -> IO ()
  tril_        :: t -> t -> Integer -> IO ()
  triu_        :: t -> t -> Integer -> IO ()
  diag_        :: t -> t -> Int -> IO ()
  eye_         :: t -> Integer -> Integer -> IO ()
  trace        :: t -> IO (HsAccReal t)
  arange_      :: t -> HsAccReal t -> HsAccReal t -> HsAccReal t -> IO ()
  range_       :: t -> HsAccReal t -> HsAccReal t -> HsAccReal t -> IO ()

class CPUTensorMath t where
  match    :: t -> t -> t -> IO (HsReal t)
  kthvalue :: t -> IndexDynamic t -> t -> Integer -> Int -> IO Int
  randperm :: t -> Generator t -> Integer -> IO ()

class TensorMathFloating t where
  linspace_     :: t -> HsReal t -> HsReal t -> Int64 -> IO ()
  logspace_     :: t -> HsReal t -> HsReal t -> Int64 -> IO ()

constant :: (TensorMath t) => Dim (d :: [Nat]) -> HsReal t -> IO t
constant d v = new d >>= \r -> fill_ r v >> pure r

_tenLike
  :: (TensorMath t)
  => (t -> t -> IO ())
  -> Dim (d::[Nat]) -> IO t
_tenLike fn_ d = do
  src <- new d
  shape <- new d
  fn_ src shape
  pure src

onesLike, zerosLike
  :: (TensorMath t)
  => Dim (d::[Nat]) -> IO t
onesLike = _tenLike onesLike_
zerosLike = _tenLike zerosLike_

range
  :: (TensorMath t)
  => Dim (d::[Nat])
  -> HsAccReal t
  -> HsAccReal t
  -> HsAccReal t
  -> IO t
range d a b c = withInplace (\r -> range_ r a b c) d



