{-# LANGUAGE ScopedTypeVariables #-}
module Torch.Indef.Static.Tensor.Math.Reduce where

import Torch.Dimensions
import qualified Torch.Indef.Dynamic.Tensor.Math.Reduce as Dynamic

import Torch.Indef.Types

minall :: Tensor d -> IO HsReal
minall t = Dynamic.minall (asDynamic t)

maxall :: Tensor d -> IO HsReal
maxall t = Dynamic.maxall (asDynamic t)

medianall :: Tensor d -> IO HsReal
medianall t = Dynamic.medianall (asDynamic t)

sumall :: Tensor d -> IO HsAccReal
sumall t = Dynamic.sumall (asDynamic t)

prodall :: Tensor d -> IO HsAccReal
prodall t = Dynamic.prodall (asDynamic t)

_max :: (Tensor d, IndexTensor d) -> Tensor d -> DimVal -> Maybe KeepDim -> IO ()
_max (r, ix) t1 = Dynamic._max (asDynamic r, longAsDynamic ix) (asDynamic t1)

_min :: (Tensor d, IndexTensor d) -> Tensor d -> DimVal -> Maybe KeepDim -> IO ()
_min (r, ix) t1 = Dynamic._min (asDynamic r, longAsDynamic ix) (asDynamic t1)

_median :: (Tensor d, IndexTensor d) -> Tensor d -> DimVal -> Maybe KeepDim -> IO ()
_median (r, ix) t1 = Dynamic._median (asDynamic r, longAsDynamic ix) (asDynamic t1)

_sum r t = Dynamic._sum (asDynamic r) (asDynamic t)

_prod :: Tensor d -> Tensor d -> DimVal -> Maybe KeepDim -> IO ()
_prod r t = Dynamic._prod (asDynamic r) (asDynamic t)


-- withKeepDim
--   :: forall d n . (Dimensions d, KnownNat n)
--   => ((Tensor d, IndexTensor '[n]) -> Tensor d -> DimVal -> Maybe KeepDim -> IO ())
--   -> Tensor d -> DimVal -> Maybe KeepDim -> IO (Tensor d, Maybe (IndexTensor '[n]))
-- withKeepDim _fn t d k = do
--   ret :: Tensor d <- new
--   ix  :: IndexTensor '[n] <- newIx
--   _fn (ret, longAsStatic ix) t d k
--   pure (ret, maybe (Just $ asStatic ix) (pure Nothing) k)
-- 
-- max, min, median
--   :: forall d n . (Dimensions d, KnownNat n)
--   => Tensor d -> DimVal -> Maybe KeepDim -> IO (Tensor d, Maybe (IndexTensor '[n]))
-- max = withKeepDim _max
-- min = withKeepDim _min
-- median = withKeepDim _median
-- 
-- sum
--   :: forall t d d' . (TensorMathReduce t, CoerceDims t d d')
--   => t d -> DimVal -> Maybe KeepDim -> IO (t d')
-- sum t d k = sudoInplace t $ \r t' -> _sum r t' d k
-- 
-- rowsum
--   :: (KnownNatDim2 r c, TensorMathReduce t, CoerceDims t '[1, c] '[r, c])
--   => t '[r, c] -> IO (t '[1, c])
-- rowsum t = Torch.Class.Tensor.Math.Reduce.Static.sum t 0 (Just keep)
-- 
-- colsum
--   :: (KnownNatDim2 r c, TensorMathReduce t, CoerceDims t '[r, 1] '[r, c])
--   => t '[r, c] -> IO (t '[r, 1])
-- colsum t = Torch.Class.Tensor.Math.Reduce.Static.sum t 0 (Just keep)


