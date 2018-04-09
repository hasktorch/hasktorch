{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE ConstraintKinds #-}
module Torch.Class.Tensor.Math.Reduce.Static where

import Torch.Class.Types
import Torch.Class.Tensor.Static
import Torch.Dimensions
import qualified Torch.Class.Tensor as Dynamic

class IsTensor t => TensorMathReduce t where
  minall       :: t d -> IO (HsReal (t d))
  maxall       :: t d -> IO (HsReal (t d))
  medianall    :: t d -> IO (HsReal (t d))
  sumall       :: t d -> IO (HsAccReal (t d))
  prodall      :: t d -> IO (HsAccReal (t d))
  max_         :: (t d, IndexTensor (t d) d) -> t d -> DimVal -> Maybe KeepDim -> IO ()
  min_         :: (t d, IndexTensor (t d) d) -> t d -> DimVal -> Maybe KeepDim -> IO ()
  median_      :: (t d, IndexTensor (t d) d) -> t d -> DimVal -> Maybe KeepDim -> IO ()
  -- TH_API void THTensor_(sum)(THTensor *r_, THTensor *t, int dimension, int keepdim);
  sum_         :: t d -> t d' -> DimVal -> Maybe KeepDim -> IO ()
  prod_        :: Dimensions d => t d -> t d -> DimVal -> Maybe KeepDim -> IO ()

type WithCoercableIndex t d =
  ( Dynamic.IsTensor (AsDynamic (IndexTensor (t d) d))
  , IsStatic (IndexTensor (t d) d)
  )

withKeepDim
  :: forall d t . (TensorMathReduce t, Dimensions d, WithCoercableIndex t d)
  => ((t d, IndexTensor (t d) d) -> t d -> DimVal -> Maybe KeepDim -> IO ())
  -> t d -> DimVal -> Maybe KeepDim -> IO (t d, Maybe (IndexTensor (t d) d))
withKeepDim fn_ t d k = do
  ret :: t d <- new
  ix  :: AsDynamic (IndexTensor (t d) d) <- Dynamic.new (dim :: Dim d)
  fn_ (ret, asStatic ix) t d k
  pure (ret, maybe (Just $ asStatic ix) (pure Nothing) k)

max, min, median
  :: (TensorMathReduce t, Dimensions d, WithCoercableIndex t d)
  => t d -> DimVal -> Maybe KeepDim -> IO (t d, Maybe (IndexTensor (t d) d))
max = withKeepDim max_
min = withKeepDim min_
median = withKeepDim median_

sum
  :: forall t d d' . (TensorMathReduce t, CoerceDims t d d')
  => t d -> DimVal -> Maybe KeepDim -> IO (t d')
sum t d k = sudoInplace t $ \r t' -> sum_ r t' d k

rowsum
  :: (KnownNatDim2 r c, TensorMathReduce t, CoerceDims t '[1, c] '[r, c])
  => t '[r, c] -> IO (t '[1, c])
rowsum t = Torch.Class.Tensor.Math.Reduce.Static.sum t 0 (Just keep)

colsum
  :: (KnownNatDim2 r c, TensorMathReduce t, CoerceDims t '[r, 1] '[r, c])
  => t '[r, c] -> IO (t '[r, 1])
colsum t = Torch.Class.Tensor.Math.Reduce.Static.sum t 0 (Just keep)

class TensorMathReduceFloating t where
  var_    :: (Dimensions d, Dimensions d') => t d -> t d' -> Int -> Int -> Int -> IO ()
  std_    :: (Dimensions d, Dimensions d') => t d -> t d' -> Int -> Int -> Int -> IO ()
  renorm_ :: (Dimensions d, Dimensions d') => t d -> t d' -> HsReal (t d') -> Int -> HsReal (t d') -> IO ()
  norm_   :: (Dimensions d, Dimensions d') => t d -> t d' -> HsReal (t d') -> Int -> Int -> IO ()
  mean_   :: (Dimensions d, Dimensions d') => t d -> t d' -> Int -> Int -> IO ()

  dist    :: (Dimensions d, Dimensions d') => t d -> t d' -> HsReal (t d') -> IO (HsAccReal (t d'))
  varall  :: t d -> Int -> IO (HsAccReal (t d))
  stdall  :: t d -> Int -> IO (HsAccReal (t d'))
  normall :: t d -> HsReal (t d) -> IO (HsAccReal (t d))
  meanall :: t d -> IO (HsAccReal (t d'))

-- * not in THC.BYte
-- c_renorm :: Ptr CState -> t -> t -> HsReal t -> CInt -> HsReal t -> IO ()
-- c_std :: Ptr CState -> t -> Ptr CTensor -> CInt -> CInt -> CInt -> IO ()
-- c_stdall :: Ptr CState -> Ptr CTensor -> CInt -> IO HsReal t
-- c_var :: Ptr CState -> Ptr CTensor -> Ptr CTensor -> CInt -> CInt -> CInt -> IO ()
-- c_varall :: Ptr CState -> Ptr CTensor -> CInt -> IO HsReal t
-- c_dist :: Ptr CState -> Ptr CTensor -> Ptr CTensor -> HsReal t -> IO HsReal t

-- * not in TH.Byte
-- c_norm :: Ptr CState -> Ptr CTensor -> Ptr CTensor -> HsReal t -> CInt -> CInt -> IO ()
-- c_normall :: Ptr CState -> Ptr CTensor -> HsReal t -> IO HsReal t
-- c_mean :: Ptr CState -> Ptr CTensor -> Ptr CTensor -> CInt -> CInt -> IO ()
-- c_meanall :: Ptr CState -> Ptr CTensor -> IO HsReal t
