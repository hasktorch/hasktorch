{-# LANGUAGE ScopedTypeVariables #-}
module Torch.Indef.Static.Tensor.Math.Reduce
  ( minall
  , maxall
  , medianall
  , sumall
  , prodall
  , Torch.Indef.Static.Tensor.Math.Reduce.min
  , Torch.Indef.Static.Tensor.Math.Reduce.max
  , median
  , Torch.Indef.Static.Tensor.Math.Reduce.sum, rowsum, colsum
  , _prod

  ) where

import Data.Coerce
import System.IO.Unsafe

import Torch.Dimensions
import Torch.Indef.Index
import Torch.Indef.Static.Tensor
import Torch.Indef.Types
import qualified Torch.Indef.Dynamic.Tensor.Math.Reduce as Dynamic


minall :: Tensor d -> HsReal
minall t = Dynamic.minall (asDynamic t)

maxall :: Tensor d -> HsReal
maxall t = Dynamic.maxall (asDynamic t)

medianall :: Tensor d -> HsReal
medianall t = Dynamic.medianall (asDynamic t)

sumall :: Tensor d -> HsAccReal
sumall t = Dynamic.sumall (asDynamic t)

prodall :: Tensor d -> HsAccReal
prodall t = Dynamic.prodall (asDynamic t)

max, min, median
  :: (Dimensions d, KnownDim n) => Tensor d -> Idx dimval -> KeepDim -> (Tensor d, Maybe (IndexTensor '[n]))
max    = withKeepDim Dynamic._max
min    = withKeepDim Dynamic._min
median = withKeepDim Dynamic._median

_prod :: Tensor d -> Tensor d -> DimVal -> Maybe KeepDim -> IO ()
_prod r t = Dynamic._prod (asDynamic r) (asDynamic t)

-------------------------------------------------------------------------------

withKeepDim
  :: forall d n dimval . (Dimensions d, KnownDim n)
  => ((Dynamic, IndexDynamic) -> Dynamic -> DimVal -> Maybe KeepDim -> IO ())
  -> Tensor d -> Idx dimval -> KeepDim -> (Tensor d, Maybe (IndexTensor '[n]))
withKeepDim _fn t d k = unsafePerformIO $ do
  ret :: Tensor d <- new
  let ix :: IndexTensor '[n] = newIx
  _fn (asDynamic ret, longAsDynamic ix) (asDynamic t) (fromIntegral $ idxToWord d) (Just k)
  pure (ret, if coerce k then Just ix else Nothing)
{-# NOINLINE withKeepDim #-}


sum :: Dimensions d' => Tensor d -> DimVal -> KeepDim -> Tensor d'
sum t d k = unsafePerformIO $ do
  r <- new
  Dynamic._sum (asDynamic r) (asDynamic t) d (Just k)
  pure r
{-# NOINLINE sum #-}

rowsum :: (KnownDim2 r c) => Tensor '[r, c] -> (Tensor '[1, c])
rowsum t = Torch.Indef.Static.Tensor.Math.Reduce.sum t 0 keep

colsum :: (KnownDim2 r c) => Tensor '[r, c] -> (Tensor '[r, 1])
colsum t = Torch.Indef.Static.Tensor.Math.Reduce.sum t 0 keep


