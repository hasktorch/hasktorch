{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# OPTIONS_GHC -fno-cse #-}
module Torch.Class.Tensor.Math.Blas.Static where

import Torch.Class.Types
import Torch.Class.Tensor.Static
import Torch.Class.Tensor.Math.Static
import Torch.Dimensions
import System.IO.Unsafe

class (TensorMath t) => TensorMathBlas t where
  -- | beta * t + alpha * (src1 #> src2)
  _addmv :: KnownNatDim2 r c => t '[r] -> HsReal (t '[r]) -> t '[r] -> HsReal (t '[r]) -> t '[r, c] -> t '[c] -> IO ()

  -- only matrix-matrix multiplication:
  -- https://github.com/torch/torch7/blob/aed31711c6b8846b8337a263a7f9f998697994e7/doc/maths.md#res-torchaddmmres-v1-m-v2-mat1-mat2
  _addmm :: KnownNatDim3 a b c => t '[a, c] -> HsReal (t '[a, c]) -> t '[a, c] -> HsReal (t '[a, c]) -> t '[a, b] -> t '[b, c] -> IO ()

  -- outer product between a 1D tensor and a 1D tensor:
  -- https://github.com/torch/torch7/blob/aed31711c6b8846b8337a263a7f9f998697994e7/doc/maths.md#res-torchaddrres-v1-mat-v2-vec1-vec2
  --
  -- res_ij = (v1 * mat_ij) + (v2 * vec1_i * vec2_j)
  _addr :: t '[r, c] -> HsReal (t '[r,c]) -> t '[r,c] -> HsReal (t '[r,c]) -> t '[r] -> t '[c] -> IO ()

  _addbmm  :: Dimensions4 d d' d'' d''' => t d -> HsReal (t d) -> t d' -> HsReal (t d) -> t d'' -> t d''' -> IO ()
  _baddbmm :: Dimensions4 d d' d'' d''' => t d -> HsReal (t d) -> t d' -> HsReal (t d) -> t d'' -> t d''' -> IO ()
  dot      :: Dimensions2 d d' => t d -> t d' -> IO (HsAccReal (t d))

-- | inplace addmv
addmv
  :: (KnownNatDim2 r c, TensorMathBlas t)
  => HsReal (t '[r]) -> t '[r] -> HsReal (t '[r]) -> t '[r, c] -> t '[c] -> IO (t '[r])
addmv a b c d e = withEmpty $ \r -> _addmv r a b c d e

-- | added simplified use of addmv: src1 #> src2
mv
  :: (KnownNatDim2 r c, Num (HsReal (t '[r])), TensorMathBlas t)
  => t '[r, c] -> t '[c] -> IO (t '[r])
mv m v = constant 0 >>= \n -> addmv 0 n 1 m v

(!*) :: (KnownNatDim2 r c, Num (HsReal (t '[r])), TensorMathBlas t) => t '[r, c] -> t '[c] -> t '[r]
(!*) a b = unsafePerformIO $ mv a b
{-# NOINLINE (!*) #-}

addmm
  :: KnownNatDim3 a b c
  => TensorMathBlas t
  => Num (HsReal (t '[a, c]))
  => HsReal (t '[a, c]) -> t '[a, c] -> HsReal (t '[a, c]) -> t '[a, b] -> t '[b, c] -> IO (t '[a, c])
addmm a m b x y = withEmpty $ \r -> _addmm r a m b x y

mmult
  :: KnownNatDim3 a b c
  => TensorMathBlas t
  => Num (HsReal (t '[a, c]))
  => t '[a, b] -> t '[b, c] -> IO (t '[a, c])
mmult x y = constant 0 >>= \n -> addmm 1 n 1 x y

(!*!) :: (KnownNatDim3 a b c, TensorMathBlas t, Num (HsReal (t '[a, c]))) => t '[a, b] -> t '[b, c] -> t '[a, c]
(!*!) a b = unsafePerformIO $ mmult a b
{-# NOINLINE (!*!) #-}


addr
  :: (TensorMathBlas t, KnownNatDim2 r c)
  => HsReal (t '[r,c]) -> t '[r,c] -> HsReal (t '[r,c]) -> t '[r] -> t '[c] -> IO (t '[r, c])
addr a t b x y = withEmpty $ \r -> _addr r a t b x y

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

