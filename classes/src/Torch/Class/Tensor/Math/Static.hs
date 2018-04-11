{-# LANGUAGE TypeFamilies #-}
module Torch.Class.Tensor.Math.Static where

import GHC.Int
import Torch.Dimensions
import Torch.Class.Types
import Torch.Class.Tensor.Static
import qualified Torch.Types.TH as TH (IndexStorage)

class IsTensor t => TensorMath t where
  _fill        :: Dimensions  d    => t d  -> HsReal (t d) -> IO ()
  _zero        :: Dimensions  d    => t d  -> IO ()
  _zeros       :: Dimensions  d    => t d  -> IndexStorage (t d) -> IO ()
  _zerosLike   :: Dimensions2 d d' => t d' -> t d -> IO ()
  _ones        :: Dimensions  d    => t d  -> TH.IndexStorage -> IO ()
  _onesLike    :: Dimensions2 d d' => t d' -> t d -> IO ()
  numel        :: Dimensions  d    => t d  -> IO Integer
  _reshape     :: Dimensions2 d d' => t d' -> t d -> TH.IndexStorage -> IO ()
  _cat         :: Dimensions3 d d' d'' => t d'' -> t d -> t d' -> DimVal -> IO ()
  _catArray    :: Dimensions  d    => t d  -> [AsDynamic t] -> Int -> DimVal -> IO ()
  _nonzero     :: Dimensions  d    => IndexTensor t d -> t d -> IO ()
  _tril        :: Dimensions2 d d' => t d' -> t d -> Integer -> IO ()
  _triu        :: Dimensions2 d d' => t d' -> t d -> Integer -> IO ()
  _diag        :: Dimensions2 d d' => t d' -> t d -> Int -> IO ()
  _eye         :: Dimensions  d    => t d  -> Integer -> Integer -> IO ()
  trace        :: Dimensions  d    => t d  -> IO (HsAccReal (t d))
  _arange      :: Dimensions  d    => t d  -> HsAccReal (t d) -> HsAccReal (t d) -> HsAccReal (t d) -> IO ()
  _range       :: Dimensions  d    => t d  -> HsAccReal (t d) -> HsAccReal (t d) -> HsAccReal (t d) -> IO ()

class TensorMathFloating t where
  _linspace :: Dimensions d => t d -> HsReal (t d) -> HsReal (t d) -> Int64 -> IO ()
  _logspace :: Dimensions d => t d -> HsReal (t d) -> HsReal (t d) -> Int64 -> IO ()

constant :: (TensorMath t, Dimensions d) => HsReal (t d) -> IO (t d)
constant v = do
  t <- new
  _fill t v
  pure t

diag_ :: (TensorMath t, CoerceDims t d d') => t d -> Int -> IO (t d')
diag_ t d = sudoInplace t $ \r t' -> _diag r t d

diag :: (TensorMath t, Dimensions2 d d') => t d -> Int -> IO (t d')
diag t d = withEmpty $ \r -> _diag r t d

-- | Create a diagonal matrix from a 1D vector
diag1d :: (KnownNatDim n, TensorMath t) => t '[n] -> IO (t '[n, n])
diag1d t = diag t 0

cat_
  :: (IsStatic t)
  => (CoerceDims t d d', Dimensions3 d d' d'')
  => (TensorMath t)
  => t d -> t d' -> DimVal -> IO (t d'')
cat_ a b d = _cat a a b d >> pure (asStatic (asDynamic a))

cat :: (TensorMath t, Dimensions3 d d' d'') => t d -> t d' -> DimVal -> IO (t d'')
cat a b d = withEmpty $ \r -> _cat r a b d

cat1d :: (TensorMath t, SingDim3 n1 n2 n, n ~ Sum [n1, n2]) => t '[n1] -> t '[n2] -> IO (t '[n])
cat1d a b = cat a b 0

_tenLike
  :: (Dimensions d, TensorMath t)
  => (t d -> t d -> IO ())
  -> IO (t d)
_tenLike _fn = do
  src <- new
  shape <- new
  _fn src shape
  pure src

onesLike, zerosLike :: (Dimensions d, TensorMath t) => IO (t d)
onesLike = _tenLike _onesLike
zerosLike = _tenLike _zerosLike



