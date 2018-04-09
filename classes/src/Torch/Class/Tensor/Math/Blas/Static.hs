{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# OPTIONS_GHC -fno-cse #-}
module Torch.Class.Tensor.Math.Blas.Static where

import Torch.Class.Types
import Torch.Class.Tensor.Static
import Torch.Class.Tensor.Math.Static
import Torch.Dimensions
import System.IO.Unsafe

class (Tensor t, TensorMath t) => TensorMathBlas t where
  addmv_   :: Dimensions4 d d' d'' d''' => t d -> HsReal (t d) -> t d' -> HsReal (t d) -> t d'' -> t d''' -> IO ()
  addmm_   :: Dimensions4 d d' d'' d''' => t d -> HsReal (t d) -> t d' -> HsReal (t d) -> t d'' -> t d''' -> IO ()

  -- outer product between a 1D tensor and a 1D tensor:
  -- https://github.com/torch/torch7/blob/aed31711c6b8846b8337a263a7f9f998697994e7/doc/maths.md#res-torchaddrres-v1-mat-v2-vec1-vec2
  --
  -- res_ij = (v1 * mat_ij) + (v2 * vec1_i * vec2_j)
  addr_    :: t '[r, c] -> HsReal (t '[r,c]) -> t '[r,c] -> HsReal (t '[r,c]) -> t '[r] -> t '[c] -> IO ()
  addbmm_  :: Dimensions4 d d' d'' d''' => t d -> HsReal (t d) -> t d' -> HsReal (t d) -> t d'' -> t d''' -> IO ()
  baddbmm_ :: Dimensions4 d d' d'' d''' => t d -> HsReal (t d) -> t d' -> HsReal (t d) -> t d'' -> t d''' -> IO ()
  dot      :: Dimensions2 d d' => t d -> t d' -> IO (HsAccReal (t d))


addr
  :: (TensorMathBlas t, KnownNatDim2 r c)
  => HsReal (t '[r,c]) -> t '[r,c] -> HsReal (t '[r,c]) -> t '[r] -> t '[c] -> IO (t '[r, c])
addr a t b x y = withEmpty $ \r -> addr_ r a t b x y

outer
  :: forall t r c . (Num (HsReal (t '[r, c])), TensorMathBlas t, KnownNatDim2 r c)
  => t '[r] -> t '[c] -> IO (t '[r, c])
outer v1 v2 = do
  t :: t '[r, c] <- zerosLike
  addr 0 t 1 v1 v2

(<.>)
  :: (TensorMathBlas t, Dimensions2 d d')
  => t d
  -> t d'
  -> HsAccReal (t d)
(<.>) a b = unsafePerformIO $ dot a b
{-# NOINLINE (<.>) #-}

