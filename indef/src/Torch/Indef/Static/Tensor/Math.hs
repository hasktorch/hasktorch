{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE ScopedTypeVariables #-}
module Torch.Indef.Static.Tensor.Math where

import Torch.Dimensions
import Torch.Indef.Types
import Torch.Indef.Static.Tensor
import qualified Torch.Types.TH as TH
import qualified Torch.Indef.Dynamic.Tensor.Math as Dynamic

_fill r = Dynamic._fill (asDynamic r)
_zero r = Dynamic._zero (asDynamic r)
_zeros r = Dynamic._zeros (asDynamic r)
_zerosLike r t = Dynamic._zerosLike (asDynamic r) (asDynamic t)
_ones r = Dynamic._ones (asDynamic r)
_onesLike r t = Dynamic._onesLike (asDynamic r) (asDynamic t)
numel t = Dynamic.numel (asDynamic t)
_reshape r t = Dynamic._reshape (asDynamic r) (asDynamic t)
_cat r a b = Dynamic._cat (asDynamic r) (asDynamic a) (asDynamic b)
_catArray r = Dynamic._catArray (asDynamic r)
_nonzero r t = Dynamic._nonzero (longAsDynamic r) (asDynamic t)
_tril r t = Dynamic._tril (asDynamic r) (asDynamic t)
_triu r t = Dynamic._triu (asDynamic r) (asDynamic t)
_diag r t = Dynamic._diag (asDynamic r) (asDynamic t)
_eye r = Dynamic._eye (asDynamic r)
trace r = Dynamic.trace (asDynamic r)
_arange r = Dynamic._arange (asDynamic r)
_range r = Dynamic._range (asDynamic r)

constant :: (Dimensions d) => HsReal -> IO (Tensor d)
constant v = do
  t <- new
  _fill t v
  pure t

diag_ :: (Dimensions2 d d') => Tensor d -> Int -> IO (Tensor d')
diag_ t d = sudoInplace t $ \r t' -> _diag r t d

diag :: (Dimensions2 d d') => Tensor d -> Int -> IO (Tensor d')
diag t d = withEmpty $ \r -> _diag r t d

-- | Create a diagonal matrix from a 1D vector
diag1d :: (KnownNatDim n) => Tensor '[n] -> IO (Tensor '[n, n])
diag1d t = diag t 0

cat_
  :: (Dimensions3 d d' d'')
  => Tensor d -> Tensor d' -> DimVal -> IO (Tensor d'')
cat_ a b d = _cat a a b d >> pure (asStatic (asDynamic a))

cat :: (Dimensions3 d d' d'') => Tensor d -> Tensor d' -> DimVal -> IO (Tensor d'')
cat a b d = withEmpty $ \r -> _cat r a b d

cat1d :: (SingDim3 n1 n2 n, n ~ Sum [n1, n2]) => Tensor '[n1] -> Tensor '[n2] -> IO (Tensor '[n])
cat1d a b = cat a b 0

cat2d1 :: (SingDim4 n m m0 m1, m ~ Sum [m0, m1]) => Tensor '[n, m0] -> Tensor '[n, m1] -> IO (Tensor '[n, m])
cat2d1 a b = cat a b 1

cat2d0 :: (SingDim4 n m n0 n1, n ~ Sum [n0, n1]) => Tensor '[n0, m] -> Tensor '[n1, m] -> IO (Tensor '[n, m])
cat2d0 a b = cat a b 0

catArray :: (Dimensions d) => [Dynamic] -> DimVal -> IO (Tensor d)
catArray ts dv = empty >>= \r -> _catArray r ts (length ts) dv >> pure r


_tenLike
  :: (Dimensions d)
  => (Tensor d -> Tensor d -> IO ())
  -> IO (Tensor d)
_tenLike _fn = do
  src <- new
  shape <- new
  _fn src shape
  pure src

onesLike, zerosLike :: (Dimensions d) => IO (Tensor d)
onesLike = _tenLike _onesLike
zerosLike = _tenLike _zerosLike



