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

add :: Tensor -> Tensor -> Tensor
add a b = unsafePerformIO $ (cast3 ATen.add_tts) a b kOne

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

relu :: Tensor -> Tensor
relu t = unsafePerformIO $ (cast1 ATen.relu_t) t

sigmoid :: Tensor -> Tensor
sigmoid t = unsafePerformIO $ (cast1 ATen.sigmoid_t) t

tanh :: Tensor -> Tensor
tanh t = unsafePerformIO $ (cast1 ATen.tanh_t) t

gt :: Tensor -> Tensor -> Tensor
gt a b = unsafePerformIO $ (cast2 ATen.gt_tt) a b

(>.) = gt

lt :: Tensor -> Tensor -> Tensor
lt a b = unsafePerformIO $ (cast2 ATen.lt_tt) a b

(<.) = lt

ge :: Tensor -> Tensor -> Tensor
ge a b = unsafePerformIO $ (cast2 ATen.ge_tt) a b

(>=.) = ge

le :: Tensor -> Tensor -> Tensor
le a b = unsafePerformIO $ (cast2 ATen.le_tt) a b

(<=.) = le

eq :: Tensor -> Tensor -> Tensor
eq a b = unsafePerformIO $ (cast2 ATen.eq_tt) a b

(==.) = eq

ne :: Tensor -> Tensor -> Tensor
ne a b = unsafePerformIO $ (cast2 ATen.ne_tt) a b

(/=.) = ne

toDType :: DType -> Tensor -> Tensor
toDType dtype t = unsafePerformIO $ (cast4 ATen.tensor_to_sbb) t dtype False False

squeezeAll :: Tensor -> Tensor
squeezeAll t = unsafePerformIO $ (cast1 ATen.squeeze_t) t

mse_loss :: Tensor -> Tensor -> Tensor
mse_loss a b = unsafePerformIO $ (cast3 ATen.mse_loss_ttl) a b ATen.kMean

conv2d :: Tensor -> Tensor -> Tensor -> (Int, Int) -> (Int, Int) -> Tensor
conv2d input weight bias (dh, dw) (ph, pw) = unsafePerformIO $
    (cast7 ATen.conv2d_tttllll) input weight bias
                                [dh, dw] [ph, pw] ([1, 1] :: [Int]) (0 :: Int)

maxPool2d :: Tensor -> (Int, Int) -> (Int, Int) -> (Int, Int) -> Tensor
maxPool2d input (kh, kw) (dh, dw) (ph, pw) = unsafePerformIO $
    (cast6 ATen.max_pool2d_tllllb) input [kh, kw] [dh, dw] [ph, pw] ([1, 1] :: [Int]) False

logSoftmax :: Tensor -> Int -> Tensor
logSoftmax input dim = unsafePerformIO $ (cast3 ATen.log_softmax_tls) input dim (dtype input)
