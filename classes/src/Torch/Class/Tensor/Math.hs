{-# LANGUAGE DataKinds #-}
module Torch.Class.Tensor.Math where

import Foreign hiding (new)
import Foreign.C.Types
import GHC.TypeLits (Nat)
import Torch.Dimensions
import Torch.Class.Types
import GHC.Int
import Torch.Class.Tensor (IsTensor(empty), withInplace, withEmpty, new)
import qualified Torch.Types.TH as TH

class IsTensor t => TensorMath t where
  _fill        :: t -> HsReal t -> IO ()
  _zero        :: t -> IO ()
  _zeros       :: t -> IndexStorage t -> IO ()
  _zerosLike   :: t -> t -> IO ()
  _ones        :: t -> TH.IndexStorage -> IO ()
  _onesLike    :: t -> t -> IO ()
  numel        :: t -> IO Integer
  _reshape     :: t -> t -> TH.IndexStorage -> IO ()
  _cat         :: t -> t -> t -> DimVal -> IO ()
  _catArray    :: t -> [t] -> Int -> DimVal -> IO ()
  _nonzero     :: IndexDynamic t -> t -> IO ()
  _tril        :: t -> t -> Integer -> IO ()
  _triu        :: t -> t -> Integer -> IO ()
  _diag        :: t -> t -> Int -> IO ()
  _eye         :: t -> Integer -> Integer -> IO ()
  trace        :: t -> IO (HsAccReal t)
  _arange      :: t -> HsAccReal t -> HsAccReal t -> HsAccReal t -> IO ()
  _range       :: t -> HsAccReal t -> HsAccReal t -> HsAccReal t -> IO ()

class CPUTensorMath t where
  match    :: t -> t -> t -> IO (HsReal t)
  kthvalue :: t -> IndexDynamic t -> t -> Integer -> Int -> IO Int
  randperm :: t -> Generator t -> Integer -> IO ()

class TensorMathFloating t where
  _linspace     :: t -> HsReal t -> HsReal t -> Int64 -> IO ()
  _logspace     :: t -> HsReal t -> HsReal t -> Int64 -> IO ()

constant :: (TensorMath t) => Dim (d :: [Nat]) -> HsReal t -> IO t
constant d v = new d >>= \r -> _fill r v >> pure r

diag_, diag :: TensorMath t => t -> Int -> IO t
diag_ t d = _diag t t d >> pure t
diag  t d = withEmpty $ \r -> _diag r t d

diag1d :: TensorMath t => t -> IO t
diag1d t = diag t 1

_tenLike
  :: (TensorMath t)
  => (t -> t -> IO ())
  -> Dim (d::[Nat]) -> IO t
_tenLike _fn d = do
  src <- new d
  shape <- new d
  _fn src shape
  pure src

onesLike, zerosLike
  :: (TensorMath t)
  => Dim (d::[Nat]) -> IO t
onesLike = _tenLike _onesLike
zerosLike = _tenLike _zerosLike

range
  :: (TensorMath t)
  => Dim (d::[Nat])
  -> HsAccReal t
  -> HsAccReal t
  -> HsAccReal t
  -> IO t
range d a b c = withInplace (\r -> _range r a b c) d



