{-# LANGUAGE TypeApplications #-}

module Torch.Functions where

import System.IO.Unsafe
import Foreign.ForeignPtr

import qualified ATen.Managed.Native as ATen
import qualified ATen.Managed.Type.Tensor as ATen
import qualified ATen.Managed.Type.Scalar as ATen
import qualified ATen.Const as ATen
import qualified ATen.Type as ATen
import qualified ATen.Managed.Cast
import ATen.Cast

import Torch.Tensor
import Torch.DType

kOne :: ForeignPtr ATen.Scalar
kOne = unsafePerformIO $ ATen.newScalar_i 1

instance Num Tensor where
  (+) = add
  (-) = sub
  (*) = mul
  negate t = unsafePerformIO $ (cast1 ATen.neg_t) t
  abs t = unsafePerformIO $ (cast1 ATen.abs_t) t
  signum t = unsafePerformIO $ (cast1 ATen.sign_t) t
  fromInteger = asTensor

instance Fractional Tensor where
  a / b = unsafePerformIO $ (cast2 ATen.div_tt) a b
  recip t = unsafePerformIO $ (cast1 ATen.reciprocal_t) t
  fromRational = asTensor . (fromRational @Double)

sumAll :: Tensor -> Tensor
sumAll t = unsafePerformIO $ (cast1 ATen.sum_t) t

abs :: Tensor -> Tensor
abs t = unsafePerformIO $ (cast1 ATen.abs_t) t

add :: Tensor -> Tensor -> Tensor
add a b = unsafePerformIO $ (cast3 ATen.add_tts) a b kOne

ceil :: Tensor -> Tensor
ceil t = unsafePerformIO $ (cast1 ATen.ceil_t) t

floor :: Tensor -> Tensor
floor t = unsafePerformIO $ (cast1 ATen.floor_t) t

min :: Tensor -> Tensor
min t = unsafePerformIO $ (cast1 ATen.min_t) t

max :: Tensor -> Tensor
max t = unsafePerformIO $ (cast1 ATen.max_t) t

median :: Tensor -> Tensor
median t = unsafePerformIO $ (cast1 ATen.median_t) t

sub :: Tensor -> Tensor -> Tensor
sub a b = unsafePerformIO $ (cast3 ATen.sub_tts) a b kOne

mul :: Tensor -> Tensor -> Tensor
mul a b = unsafePerformIO $ (cast2 ATen.mul_tt) a b

matmul :: Tensor -> Tensor -> Tensor
matmul a b =
    unsafePerformIO $ case (dim a, dim b) of
      (2, 2) -> mm a b
      _ -> error "Unsupported case in matmul!"
  where
    mm = cast2 ATen.mm_tt

erf :: Tensor -> Tensor
erf t = unsafePerformIO $ (cast1 ATen.erf_t) t

exp :: Tensor -> Tensor
exp t = unsafePerformIO $ (cast1 ATen.exp_t) t

log1p :: Tensor -> Tensor
log1p t = unsafePerformIO $ (cast1 ATen.log1p_t) t

log2 :: Tensor -> Tensor
log2 t = unsafePerformIO $ (cast1 ATen.log2_t) t

log10 :: Tensor -> Tensor
log10 t = unsafePerformIO $ (cast1 ATen.log10_t) t

relu :: Tensor -> Tensor
relu t = unsafePerformIO $ (cast1 ATen.relu_t) t

selu :: Tensor -> Tensor
selu t = unsafePerformIO $ (cast1 ATen.selu_t) t

sigmoid :: Tensor -> Tensor
sigmoid t = unsafePerformIO $ (cast1 ATen.sigmoid_t) t

sin :: Tensor -> Tensor
sin t = unsafePerformIO $ (cast1 ATen.sin_t) t

sinh :: Tensor -> Tensor
sinh t = unsafePerformIO $ (cast1 ATen.sinh_t) t

cos :: Tensor -> Tensor
cos t = unsafePerformIO $ (cast1 ATen.cos_t) t

sqrt :: Tensor -> Tensor
sqrt t = unsafePerformIO $ (cast1 ATen.sqrt_t) t

tanh :: Tensor -> Tensor
tanh t = unsafePerformIO $ (cast1 ATen.tanh_t) t

gt :: Tensor -> Tensor -> Tensor
gt a b = unsafePerformIO $ (cast2 ATen.gt_tt) a b

toDType :: DType -> Tensor -> Tensor
toDType dtype t = unsafePerformIO $ (cast4 ATen.tensor_to_sbb) t dtype False False

squeezeAll :: Tensor -> Tensor
squeezeAll t = unsafePerformIO $ (cast1 ATen.squeeze_t) t

mse_loss :: Tensor -> Tensor -> Tensor
mse_loss a b = unsafePerformIO $ (cast3 ATen.mse_loss_ttl) a b ATen.kMean
