{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE UndecidableSuperClasses #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE NoStarIsType #-}

module Torch.Static.Native where

import           Prelude                 hiding ( sin
                                                , abs
                                                , max
                                                , min
                                                , floor
                                                )
import           Data.Proxy
import           Data.Reflection
import           Data.Finite                    ( )
import           Data.Kind                      ( Constraint )
import           GHC.TypeLits
import qualified Data.Int                      as I

import           System.IO.Unsafe
import           Foreign.ForeignPtr

import qualified ATen.Managed.Native           as ATen
import qualified ATen.Managed.Type.Tensor      as ATen
import qualified ATen.Managed.Type.Scalar      as ATen
import qualified ATen.Managed.Type.Tuple       as ATen
import qualified ATen.Const                    as ATen
import qualified ATen.Type                     as ATen
import qualified ATen.Cast                     as ATen
import qualified ATen.Managed.Cast

import qualified Torch.Tensor                  as D
import qualified Torch.TensorFactories         as D
import qualified Torch.Functions               as D
import qualified Torch.DType                   as D
import           Torch.Static
import           Torch.Static.Factories
import qualified Torch.Scalar                  as D
import qualified Torch.TensorOptions           as D

-- |
-- >>> dim (ones :: Tensor Float '[2,2])
-- 2
dim :: Tensor dtype shape -> Int
dim t = D.dim $ toDynamic t

-- |
-- >>> shape (ones :: Tensor Float '[2,2])
-- [2,2]
shape :: Tensor dtype shape -> [Int]
shape t = D.shape $ toDynamic t

-- |
-- >>> dtype (ones :: Tensor Float '[2,2])
-- Float
dtype :: Tensor dtype shape -> D.DType
dtype t = D.dtype $ toDynamic t

-- |
-- >>> toInt (ones :: Tensor Float '[])
-- 1
toInt :: Tensor dtype shape -> Int
toInt t = D.toInt $ toDynamic t

-- |
-- >>> t = sumAll (ones :: Tensor Float '[2,2])
-- >>> (dtype t,shape t,toInt t)
-- (Float,[],4)
sumAll :: Tensor dtype shape -> Tensor dtype shape
sumAll t = unsafePerformIO $ (ATen.cast1 ATen.sum_t) t

-- |
-- >>> t = abs (ones :: Tensor Float '[2,2])
-- >>> (dtype t,shape t)
-- (Float,[2,2])
abs :: Tensor dtype shape -> Tensor dtype shape
abs t = unsafePerformIO $ (ATen.cast1 ATen.abs_t) t

-- |
-- >>> t = ceil (ones :: Tensor Float '[2,2])
-- >>> (dtype t,shape t)
-- (Float,[2,2])
ceil :: Tensor dtype shape -> Tensor dtype shape
ceil t = unsafePerformIO $ (ATen.cast1 ATen.ceil_t) t

-- |
-- >>> t = floor (ones :: Tensor Float '[2,2])
-- >>> (dtype t,shape t)
-- (Float,[2,2])
floor :: Tensor dtype shape -> Tensor dtype shape
floor t = unsafePerformIO $ (ATen.cast1 ATen.floor_t) t

-- |
-- >>> t = min (ones :: Tensor Float '[2,2])
-- >>> (dtype t,shape t)
-- (Float,[])
min :: Tensor dtype shape -> Tensor dtype '[]
min t = unsafePerformIO $ (ATen.cast1 ATen.min_t) t

-- |
-- >>> t = max (ones :: Tensor Float '[2,2])
-- >>> (dtype t,shape t)
-- (Float,[])
max :: Tensor dtype shape -> Tensor dtype '[]
max t = unsafePerformIO $ (ATen.cast1 ATen.max_t) t

-- |
-- >>> t = median (ones :: Tensor Float '[2,2])
-- >>> (dtype t,shape t)
-- (Float,[])
median :: Tensor dtype shape -> Tensor dtype '[]
median t = unsafePerformIO $ (ATen.cast1 ATen.median_t) t

-- |
-- >>> t = cmul (ones :: Tensor Float '[2,2]) 1
-- >>> (dtype t,shape t)
-- (Float,[2,2])
cmul :: D.Scalar a => Tensor dtype shape -> a -> Tensor dtype shape
cmul t a = unsafePerformIO $ (ATen.cast2 ATen.mul_ts) t a

erf :: Tensor dtype shape -> Tensor dtype shape
erf t = unsafePerformIO $ (ATen.cast1 ATen.erf_t) t

exp :: Tensor dtype shape -> Tensor dtype shape
exp t = unsafePerformIO $ (ATen.cast1 ATen.exp_t) t

log1p :: Tensor dtype shape -> Tensor dtype shape
log1p t = unsafePerformIO $ (ATen.cast1 ATen.log1p_t) t

log2 :: Tensor dtype shape -> Tensor dtype shape
log2 t = unsafePerformIO $ (ATen.cast1 ATen.log2_t) t

log10 :: Tensor dtype shape -> Tensor dtype shape
log10 t = unsafePerformIO $ (ATen.cast1 ATen.log10_t) t

pow :: D.Scalar a => Tensor dtype shape -> a -> Tensor dtype shape
pow t s = unsafePerformIO $ (ATen.cast2 ATen.pow_ts) t s

relu :: Tensor dtype shape -> Tensor dtype shape
relu t = unsafePerformIO $ (ATen.cast1 ATen.relu_t) t

selu :: Tensor dtype shape -> Tensor dtype shape
selu t = unsafePerformIO $ (ATen.cast1 ATen.selu_t) t

sigmoid :: Tensor dtype shape -> Tensor dtype shape
sigmoid t = unsafePerformIO $ (ATen.cast1 ATen.sigmoid_t) t

sin :: Tensor dtype shape -> Tensor dtype shape
sin t = unsafePerformIO $ (ATen.cast1 ATen.sin_t) t

sinh :: Tensor dtype shape -> Tensor dtype shape
sinh t = unsafePerformIO $ (ATen.cast1 ATen.sinh_t) t

cos :: Tensor dtype shape -> Tensor dtype shape
cos t = unsafePerformIO $ (ATen.cast1 ATen.cos_t) t

sqrt :: Tensor dtype shape -> Tensor dtype shape
sqrt t = unsafePerformIO $ (ATen.cast1 ATen.sqrt_t) t

tanh :: Tensor dtype shape -> Tensor dtype shape
tanh t = unsafePerformIO $ (ATen.cast1 ATen.tanh_t) t

-- |
-- >>> t = gt (ones :: Tensor Float '[2,2]) (zeros :: Tensor Float '[2,2])
-- >>> (dtype t,shape t)
-- (Bool,[2,2])
gt :: Tensor dtype shape -> Tensor dtype shape -> Tensor Bool shape
gt a b = unsafePerformIO $ (ATen.cast2 ATen.gt_tt) a b

(>.) = gt

lt :: Tensor dtype shape -> Tensor dtype shape -> Tensor Bool shape
lt a b = unsafePerformIO $ (ATen.cast2 ATen.lt_tt) a b

(<.) = lt

ge :: Tensor dtype shape -> Tensor dtype shape -> Tensor Bool shape
ge a b = unsafePerformIO $ (ATen.cast2 ATen.ge_tt) a b

(>=.) = ge

le :: Tensor dtype shape -> Tensor dtype shape -> Tensor Bool shape
le a b = unsafePerformIO $ (ATen.cast2 ATen.le_tt) a b

(<=.) = le

-- |
-- >>> t = eq (ones :: Tensor Float '[2,2]) (zeros :: Tensor Float '[2,2])
-- >>> (dtype t,shape t)
-- (Bool,[2,2])
eq :: Tensor dtype shape -> Tensor dtype shape -> Tensor Bool shape
eq a b = unsafePerformIO $ (ATen.cast2 ATen.eq_tt) a b

(==.) = eq

ne :: Tensor dtype shape -> Tensor dtype shape -> Tensor Bool shape
ne a b = unsafePerformIO $ (ATen.cast2 ATen.ne_tt) a b

(/=.) = ne

-- |
-- >>> t = (toDType (ones :: Tensor Float '[2,2]) :: Tensor Double '[2,2])
-- >>> (dtype t,shape t)
-- (Double,[2,2])
toDType
  :: forall dtype dtype' shape
   . (Reifies dtype' D.DType)
  => Tensor dtype shape
  -> Tensor dtype' shape
toDType t = unsafePerformIO $ (ATen.cast4 ATen.tensor_to_sbb)
  t
  (reflect (Proxy @dtype') :: D.DType)
  False
  False

-- |
-- >>> t = squeezeAll (ones :: Tensor Float '[2,1,2,1,2]) :: Tensor Float '[2,2,2]
-- >>> (dtype t,shape t)
-- (Float,[2,2,2])
type family SqueezeAll (shape :: [Nat]) :: [Nat] where
    SqueezeAll '[] = '[]
    SqueezeAll (1: xs) = SqueezeAll xs
    SqueezeAll (x: xs) = x ': SqueezeAll xs
squeezeAll :: Tensor dtype shape -> Tensor dtype (SqueezeAll shape)
squeezeAll t = unsafePerformIO $ (ATen.cast1 ATen.squeeze_t) t

-- |
-- >>> :kind! ConditionalReduction '[3,2] D.ReduceNone
-- ConditionalReduction '[3,2] D.ReduceNone :: [Nat]
-- = '[3, 2]
-- >>> :kind! ConditionalReduction '[3,2] D.ReduceMean
-- ConditionalReduction '[3,2] D.ReduceMean :: [Nat]
-- = '[]
type family ConditionalReduction (shape :: [Nat]) (reduction :: D.Reduction) :: [Nat] where
    ConditionalReduction shape D.ReduceNone = shape
    ConditionalReduction shape _ = '[]

class KnownReduction reduction where
    reductionVal :: Int

instance KnownReduction D.ReduceNone where
  reductionVal = 0
instance KnownReduction D.ReduceMean where
  reductionVal = 1
instance KnownReduction D.ReduceSum where
  reductionVal = 2

-- |
-- >>> tt = ones :: Tensor Float '[2,2]
-- >>> t = binaryCrossEntropy @D.ReduceNone tt tt tt :: Tensor Float '[2,2]
-- >>> (dtype t,shape t)
-- (Float,[2,2])
-- >>> t = binaryCrossEntropy @D.ReduceMean tt tt tt :: Tensor Float '[]
-- >>> (dtype t,shape t)
-- (Float,[])
-- >>> t = binaryCrossEntropy @D.ReduceSum tt tt tt :: Tensor Float '[]
-- >>> (dtype t,shape t)
-- (Float,[])
binaryCrossEntropy
  :: forall reduction dtype shape
   . (KnownReduction reduction)
  => Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype (ConditionalReduction shape reduction)
binaryCrossEntropy t target weight =
  unsafePerformIO $ (ATen.cast4 ATen.binary_cross_entropy_tttl)
    t
    target
    weight
    (reductionVal @reduction)

-- |
-- >>> t = mseLoss (ones :: Tensor Float '[2,2]) (ones :: Tensor Float '[2,2])
-- >>> (dtype t,shape t)
-- (Float,[])
mseLoss :: Tensor dtype shape -> Tensor dtype shape -> Tensor dtype '[]
mseLoss a b = unsafePerformIO $ (ATen.cast3 ATen.mse_loss_ttl) a b ATen.kMean

conv2d
  :: Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> (Int, Int)
  -> (Int, Int)
  -> Tensor dtype shape
conv2d input weight bias (dh, dw) (ph, pw) =
  unsafePerformIO $ (ATen.cast7 ATen.conv2d_tttllll) input
                                                     weight
                                                     bias
                                                     [dh, dw]
                                                     [ph, pw]
                                                     ([1, 1] :: [Int])
                                                     (0 :: Int)

-- maxPool2d
--   :: Tensor dtype shape
--   -> (Int, Int)
--   -> (Int, Int)
--   -> (Int, Int)
--   -> Tensor dtype shape
-- maxPool2d input (kh, kw) (dh, dw) (ph, pw) =
--   unsafePerformIO $ (ATen.cast6 ATen.max_pool2d_tllllb) input
--                                                         [kh, kw]
--                                                         [dh, dw]
--                                                         [ph, pw]
--                                                         ([1, 1] :: [Int])
--                                                         False

logSoftmax :: Tensor dtype shape -> Int -> Tensor dtype shape
logSoftmax input dim =
  unsafePerformIO $ (ATen.cast3 ATen.log_softmax_tls) input dim (dtype input)

inverse :: Tensor dtype shape -> Tensor dtype shape
inverse t = unsafePerformIO $ (ATen.cast1 ATen.inverse_t) t

symeig
  :: Tensor dtype shape
  -> Bool
  -> D.Tri
  -> (Tensor dtype shape, Tensor dtype shape)
symeig t eigenvectors upper = unsafePerformIO
  $ (ATen.cast3 ATen.symeig_tbb) t eigenvectors boolUpper
  where boolUpper = D.isUpper upper


eig :: Tensor dtype shape -> Bool -> (Tensor dtype shape, Tensor dtype shape)
eig t eigenvectors = unsafePerformIO $ (ATen.cast2 ATen.eig_tb) t eigenvectors

svd
  :: Tensor dtype shape
  -> Bool
  -> Bool
  -> (Tensor dtype shape, Tensor dtype shape, Tensor dtype shape)
svd t some compute_uv =
  unsafePerformIO $ (ATen.cast3 ATen.svd_tbb) t some compute_uv

cholesky :: Tensor dtype shape -> D.Tri -> Tensor dtype shape
cholesky t upper = unsafePerformIO $ (ATen.cast2 ATen.cholesky_tb) t boolUpper
  where boolUpper = D.isUpper upper

cholesky_solve
  :: Tensor dtype shape -> Tensor dtype shape -> D.Tri -> Tensor dtype shape
cholesky_solve t1 t2 upper =
  unsafePerformIO $ (ATen.cast3 ATen.cholesky_solve_ttb) t1 t2 boolUpper
  where boolUpper = D.isUpper upper

solve
  :: Tensor dtype shape
  -> Tensor dtype shape
  -> (Tensor dtype shape, Tensor dtype shape)
solve b a = unsafePerformIO $ (ATen.cast2 ATen.solve_tt) b a

cholesky_inverse :: Tensor dtype shape -> D.Tri -> Tensor dtype shape
cholesky_inverse t upper =
  unsafePerformIO $ (ATen.cast2 ATen.cholesky_inverse_tb) t boolUpper
  where boolUpper = D.isUpper upper

geqrf :: Tensor dtype shape -> (Tensor dtype shape, Tensor dtype shape)
geqrf t = unsafePerformIO $ (ATen.cast1 ATen.geqrf_t) t

orgqr :: Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape
orgqr b a = unsafePerformIO $ (ATen.cast2 ATen.orgqr_tt) b a

sign :: Tensor dtype shape -> Tensor dtype shape
sign t = unsafePerformIO $ (ATen.cast1 ATen.sign_t) t

type family SetValue (shape :: [Nat]) (i :: Nat) (j :: Nat)  :: [Nat] where
    SetValue '[] _ _ = '[]
    SetValue (x: xs) 0 j = j: xs
    SetValue (x: xs) i j = x: SetValue xs (i-1) j
type family GetValue (shape :: [Nat]) (i :: Nat) :: Nat where
    GetValue '[] _ = TypeError (Text "Can not find a element in the list.")
    GetValue (x: xs) 0 = x
    GetValue (x: xs) i = GetValue xs (i-1)

-- | Transpose
-- >>> :kind! Transpose '[3,2] 0 1
-- Transpose '[3,2] 0 1 :: [Nat]
-- = '[2, 3]
-- >>> :kind! Transpose '[3,2,1] 1 2
-- Transpose '[3,2,1] 1 2 :: [Nat]
-- = '[3, 1, 2]
type family Transpose (shape :: [Nat]) (dim0 :: Nat) (dim1 :: Nat) :: [Nat] where
    Transpose s d0 d1 = (SetValue (SetValue s d0 (GetValue s d1)) d1 (GetValue s d0))

-- | transpose
-- See ../../../../deps/pytorch/aten/src/ATen/native/TensorShape.cpp
-- >>> t = transpose @0 @1 (ones :: Tensor Float '[3,2])
-- >>> (dtype t,shape t)
-- (Float,[2,3])
-- >>> t = transpose @0 @1 (ones :: Tensor Float '[3,2,1])
-- >>> (dtype t,shape t)
-- (Float,[2,3,1])
-- >>> t = transpose @1 @2 (ones :: Tensor Float '[3,2,1])
-- >>> (dtype t,shape t)
-- (Float,[3,1,2])
transpose
  :: forall n m (shape :: [Nat]) dtype
   . (KnownNat n, KnownNat m)
  => Tensor dtype shape
  -> Tensor dtype (Transpose shape n m)
transpose t =
  unsafePerformIO $ (ATen.cast3 ATen.transpose_tll) t (natValI @n) (natValI @m)

-- | transpose special case for a 2D tensor
-- >>> t = transpose2D (ones :: Tensor Float '[3,2])
-- >>> (dtype t,shape t)
-- (Float,[2,3])
transpose2D
  :: forall (i :: Nat) (j :: Nat) dtype
   . Tensor dtype '[i, j]
  -> Tensor dtype '[j, i]
transpose2D t = transpose @0 @1 t

diag :: Tensor dtype shape -> Int -> Tensor dtype shape
diag t index = unsafePerformIO $ (ATen.cast2 ATen.tensor_diag_l) t index

all :: Tensor dtype shape -> Bool
all t = toInt (unsafePerformIO $ (ATen.cast1 ATen.all_t) t) == 1

any :: Tensor dtype shape -> Bool
any t = toInt (unsafePerformIO $ (ATen.cast1 ATen.any_t) t) == 1

all' :: Tensor dtype shape -> Int -> Bool -> Tensor dtype shape
all' t dim keepdim = unsafePerformIO $ (ATen.cast3 ATen.all_tlb) t dim keepdim

any' :: Tensor dtype shape -> Int -> Bool -> Tensor dtype shape
any' t dim keepdim = unsafePerformIO $ (ATen.cast3 ATen.any_tlb) t dim keepdim


---

dropout :: Tensor dtype shape -> Double -> Bool -> Tensor dtype shape
dropout _input _p _train =
  unsafePerformIO $ (ATen.cast3 ATen.dropout_tdb) _input _p _train

feature_dropout :: Tensor dtype shape -> Double -> Bool -> Tensor dtype shape
feature_dropout _input _p _train =
  unsafePerformIO $ (ATen.cast3 ATen.feature_dropout_tdb) _input _p _train

alpha_dropout :: Tensor dtype shape -> Double -> Bool -> Tensor dtype shape
alpha_dropout _input _p _train =
  unsafePerformIO $ (ATen.cast3 ATen.alpha_dropout_tdb) _input _p _train

feature_alpha_dropout
  :: Tensor dtype shape -> Double -> Bool -> Tensor dtype shape
feature_alpha_dropout _input _p _train =
  unsafePerformIO $ (ATen.cast3 ATen.feature_alpha_dropout_tdb) _input _p _train

acos :: Tensor dtype shape -> Tensor dtype shape
acos _self = unsafePerformIO $ (ATen.cast1 ATen.acos_t) _self

avg_pool1d
  :: Tensor dtype shape
  -> Int
  -> Int
  -> Int
  -> Bool
  -> Bool
  -> Tensor dtype shape
avg_pool1d _self _kernel_size _stride _padding _ceil_mode _count_include_pad =
  unsafePerformIO $ (ATen.cast6 ATen.avg_pool1d_tlllbb) _self
                                                        _kernel_size
                                                        _stride
                                                        _padding
                                                        _ceil_mode
                                                        _count_include_pad

adaptive_avg_pool1d :: Tensor dtype shape -> Int -> Tensor dtype shape
adaptive_avg_pool1d _self _output_size =
  unsafePerformIO $ (ATen.cast2 ATen.adaptive_avg_pool1d_tl) _self _output_size

adaptive_max_pool1d
  :: Tensor dtype shape -> Int -> (Tensor dtype shape, Tensor dtype shape)
adaptive_max_pool1d _self _output_size =
  unsafePerformIO $ (ATen.cast2 ATen.adaptive_max_pool1d_tl) _self _output_size

addmv
  :: Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Float
  -> Float
  -> Tensor dtype shape
addmv _self _mat _vec _beta _alpha =
  unsafePerformIO $ (ATen.cast5 ATen.addmv_tttss) _self _mat _vec _beta _alpha

addr
  :: Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Float
  -> Float
  -> Tensor dtype shape
addr _self _vec1 _vec2 _beta _alpha =
  unsafePerformIO $ (ATen.cast5 ATen.addr_tttss) _self _vec1 _vec2 _beta _alpha

affine_grid_generator :: Tensor dtype shape -> [Int] -> Tensor dtype shape
affine_grid_generator _theta _size =
  unsafePerformIO $ (ATen.cast2 ATen.affine_grid_generator_tl) _theta _size

allclose
  :: Tensor dtype shape
  -> Tensor dtype shape
  -> Double
  -> Double
  -> Bool
  -> Bool
allclose _self _other _rtol _atol _equal_nan =
  unsafePerformIO
    $ (ATen.cast5 ATen.allclose_ttddb) _self _other _rtol _atol _equal_nan

argmax :: Tensor dtype shape -> Int -> Bool -> Tensor dtype shape
argmax _self _dim _keepdim =
  unsafePerformIO $ (ATen.cast3 ATen.argmax_tlb) _self _dim _keepdim

argmin :: Tensor dtype shape -> Int -> Bool -> Tensor dtype shape
argmin _self _dim _keepdim =
  unsafePerformIO $ (ATen.cast3 ATen.argmin_tlb) _self _dim _keepdim

as_strided :: Tensor dtype shape -> [Int] -> [Int] -> Int -> Tensor dtype shape
as_strided _self _size _stride _storage_offset =
  unsafePerformIO
    $ (ATen.cast4 ATen.as_strided_tlll) _self _size _stride _storage_offset

asin :: Tensor dtype shape -> Tensor dtype shape
asin _self = unsafePerformIO $ (ATen.cast1 ATen.asin_t) _self

atan :: Tensor dtype shape -> Tensor dtype shape
atan _self = unsafePerformIO $ (ATen.cast1 ATen.atan_t) _self

baddbmm
  :: Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Float
  -> Float
  -> Tensor dtype shape
baddbmm _self _batch1 _batch2 _beta _alpha =
  unsafePerformIO
    $ (ATen.cast5 ATen.baddbmm_tttss) _self _batch1 _batch2 _beta _alpha

batch_norm
  :: Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Bool
  -> Double
  -> Double
  -> Bool
  -> Tensor dtype shape
batch_norm _input _weight _bias _running_mean _running_var _training _momentum _eps _cudnn_enabled
  = unsafePerformIO $ (ATen.cast9 ATen.batch_norm_tttttbddb) _input
                                                             _weight
                                                             _bias
                                                             _running_mean
                                                             _running_var
                                                             _training
                                                             _momentum
                                                             _eps
                                                             _cudnn_enabled

bilinear
  :: Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
bilinear _input1 _input2 _weight _bias =
  unsafePerformIO
    $ (ATen.cast4 ATen.bilinear_tttt) _input1 _input2 _weight _bias

binary_cross_entropy_with_logits
  :: Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Int
  -> Tensor dtype shape
binary_cross_entropy_with_logits _self _target _weight _pos_weight _reduction =
  unsafePerformIO $ (ATen.cast5 ATen.binary_cross_entropy_with_logits_ttttl)
    _self
    _target
    _weight
    _pos_weight
    _reduction

bincount
  :: Tensor dtype shape -> Tensor dtype shape -> Int -> Tensor dtype shape
bincount _self _weights _minlength =
  unsafePerformIO $ (ATen.cast3 ATen.bincount_ttl) _self _weights _minlength

bitwise_not :: Tensor dtype shape -> Tensor dtype shape
bitwise_not _self = unsafePerformIO $ (ATen.cast1 ATen.bitwise_not_t) _self

bmm :: Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape
bmm _self _mat2 = unsafePerformIO $ (ATen.cast2 ATen.bmm_tt) _self _mat2

broadcast_tensors :: [Tensor dtype shape] -> [Tensor dtype shape]
broadcast_tensors _tensors =
  unsafePerformIO $ (ATen.cast1 ATen.broadcast_tensors_l) _tensors

cat :: [Tensor dtype shape] -> Int -> Tensor dtype shape
cat _tensors _dim = unsafePerformIO $ (ATen.cast2 ATen.cat_ll) _tensors _dim

chain_matmul :: [Tensor dtype shape] -> Tensor dtype shape
chain_matmul _matrices =
  unsafePerformIO $ (ATen.cast1 ATen.chain_matmul_l) _matrices

chunk :: Tensor dtype shape -> Int -> Int -> [Tensor dtype shape]
chunk _self _chunks _dim =
  unsafePerformIO $ (ATen.cast3 ATen.chunk_tll) _self _chunks _dim

clamp :: Tensor dtype shape -> Float -> Float -> Tensor dtype shape
clamp _self _min _max =
  unsafePerformIO $ (ATen.cast3 ATen.clamp_tss) _self _min _max

clamp_max :: Tensor dtype shape -> Float -> Tensor dtype shape
clamp_max _self _max =
  unsafePerformIO $ (ATen.cast2 ATen.clamp_max_ts) _self _max

clamp_min :: Tensor dtype shape -> Float -> Tensor dtype shape
clamp_min _self _min =
  unsafePerformIO $ (ATen.cast2 ATen.clamp_min_ts) _self _min

cudnn_is_acceptable :: Tensor dtype shape -> Bool
cudnn_is_acceptable _self =
  unsafePerformIO $ (ATen.cast1 ATen.cudnn_is_acceptable_t) _self

constant_pad_nd :: Tensor dtype shape -> [Int] -> Float -> Tensor dtype shape
constant_pad_nd _self _pad _value =
  unsafePerformIO $ (ATen.cast3 ATen.constant_pad_nd_tls) _self _pad _value

convolution
  :: Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> [Int]
  -> [Int]
  -> [Int]
  -> Bool
  -> [Int]
  -> Int
  -> Tensor dtype shape
convolution _input _weight _bias _stride _padding _dilation _transposed _output_padding _groups
  = unsafePerformIO $ (ATen.cast9 ATen.convolution_tttlllbll) _input
                                                              _weight
                                                              _bias
                                                              _stride
                                                              _padding
                                                              _dilation
                                                              _transposed
                                                              _output_padding
                                                              _groups

conv1d
  :: Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Int
  -> Int
  -> Int
  -> Int
  -> Tensor dtype shape
conv1d _input _weight _bias _stride _padding _dilation _groups =
  unsafePerformIO $ (ATen.cast7 ATen.conv1d_tttllll) _input
                                                     _weight
                                                     _bias
                                                     _stride
                                                     _padding
                                                     _dilation
                                                     _groups

conv3d
  :: Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> (Int, Int, Int)
  -> (Int, Int, Int)
  -> (Int, Int, Int)
  -> Int
  -> Tensor dtype shape
conv3d _input _weight _bias _stride _padding _dilation _groups =
  unsafePerformIO $ (ATen.cast7 ATen.conv3d_tttllll) _input
                                                     _weight
                                                     _bias
                                                     _stride
                                                     _padding
                                                     _dilation
                                                     _groups

conv_tbc
  :: Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Int
  -> Tensor dtype shape
conv_tbc _self _weight _bias _pad =
  unsafePerformIO $ (ATen.cast4 ATen.conv_tbc_tttl) _self _weight _bias _pad

conv_transpose1d
  :: Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Int
  -> Int
  -> Int
  -> Int
  -> Int
  -> Tensor dtype shape
conv_transpose1d _input _weight _bias _stride _padding _output_padding _groups _dilation
  = unsafePerformIO $ (ATen.cast8 ATen.conv_transpose1d_tttlllll)
    _input
    _weight
    _bias
    _stride
    _padding
    _output_padding
    _groups
    _dilation

cosh :: Tensor dtype shape -> Tensor dtype shape
cosh _self = unsafePerformIO $ (ATen.cast1 ATen.cosh_t) _self

cosine_embedding_loss
  :: Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Double
  -> Int
  -> Tensor dtype shape
cosine_embedding_loss _input1 _input2 _target _margin _reduction =
  unsafePerformIO $ (ATen.cast5 ATen.cosine_embedding_loss_tttdl) _input1
                                                                  _input2
                                                                  _target
                                                                  _margin
                                                                  _reduction

cudnn_affine_grid_generator
  :: Tensor dtype shape -> Int -> Int -> Int -> Int -> Tensor dtype shape
cudnn_affine_grid_generator _theta _N _C _H _W =
  unsafePerformIO
    $ (ATen.cast5 ATen.cudnn_affine_grid_generator_tllll) _theta _N _C _H _W

cudnn_batch_norm
  :: Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Bool
  -> Double
  -> Double
  -> ( Tensor dtype shape
     , Tensor dtype shape
     , Tensor dtype shape
     )
cudnn_batch_norm _input _weight _bias _running_mean _running_var _training _exponential_average_factor _epsilon
  = unsafePerformIO $ (ATen.cast8 ATen.cudnn_batch_norm_tttttbdd)
    _input
    _weight
    _bias
    _running_mean
    _running_var
    _training
    _exponential_average_factor
    _epsilon

cudnn_convolution
  :: Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> [Int]
  -> [Int]
  -> [Int]
  -> Int
  -> Bool
  -> Bool
  -> Tensor dtype shape
cudnn_convolution _self _weight _bias _padding _stride _dilation _groups _benchmark _deterministic
  = unsafePerformIO $ (ATen.cast9 ATen.cudnn_convolution_tttllllbb)
    _self
    _weight
    _bias
    _padding
    _stride
    _dilation
    _groups
    _benchmark
    _deterministic

cudnn_convolution_transpose
  :: Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> [Int]
  -> [Int]
  -> [Int]
  -> [Int]
  -> Int
  -> Bool
  -> Bool
  -> Tensor dtype shape
cudnn_convolution_transpose _self _weight _bias _padding _output_padding _stride _dilation _groups _benchmark _deterministic
  = unsafePerformIO $ (ATen.cast10 ATen.cudnn_convolution_transpose_tttlllllbb)
    _self
    _weight
    _bias
    _padding
    _output_padding
    _stride
    _dilation
    _groups
    _benchmark
    _deterministic

cudnn_grid_sampler
  :: Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape
cudnn_grid_sampler _self _grid =
  unsafePerformIO $ (ATen.cast2 ATen.cudnn_grid_sampler_tt) _self _grid

det :: Tensor dtype shape -> Tensor dtype shape
det _self = unsafePerformIO $ (ATen.cast1 ATen.det_t) _self

diag_embed :: Tensor dtype shape -> Int -> Int -> Int -> Tensor dtype shape
diag_embed _self _offset _dim1 _dim2 =
  unsafePerformIO $ (ATen.cast4 ATen.diag_embed_tlll) _self _offset _dim1 _dim2

diagflat :: Tensor dtype shape -> Int -> Tensor dtype shape
diagflat _self _offset =
  unsafePerformIO $ (ATen.cast2 ATen.diagflat_tl) _self _offset

diagonal :: Tensor dtype shape -> Int -> Int -> Int -> Tensor dtype shape
diagonal _self _offset _dim1 _dim2 =
  unsafePerformIO $ (ATen.cast4 ATen.diagonal_tlll) _self _offset _dim1 _dim2

dot :: Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape
dot _self _tensor = unsafePerformIO $ (ATen.cast2 ATen.dot_tt) _self _tensor

einsum :: String -> [Tensor dtype shape] -> Tensor dtype shape
einsum _equation _tensors =
  unsafePerformIO $ (ATen.cast2 ATen.einsum_sl) _equation _tensors

embedding
  :: Tensor dtype shape
  -> Tensor dtype shape
  -> Int
  -> Bool
  -> Bool
  -> Tensor dtype shape
embedding _weight _indices _padding_idx _scale_grad_by_freq _sparse =
  unsafePerformIO $ (ATen.cast5 ATen.embedding_ttlbb) _weight
                                                      _indices
                                                      _padding_idx
                                                      _scale_grad_by_freq
                                                      _sparse

embedding_bag
  :: Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Bool
  -> Int
  -> Bool
  -> Tensor dtype shape
  -> ( Tensor dtype shape
     , Tensor dtype shape
     , Tensor dtype shape
     , Tensor dtype shape
     )
embedding_bag _weight _indices _offsets _scale_grad_by_freq _mode _sparse _per_sample_weights
  = unsafePerformIO $ (ATen.cast7 ATen.embedding_bag_tttblbt)
    _weight
    _indices
    _offsets
    _scale_grad_by_freq
    _mode
    _sparse
    _per_sample_weights

empty_like :: Tensor dtype shape -> Tensor dtype shape
empty_like _self = unsafePerformIO $ (ATen.cast1 ATen.empty_like_t) _self

erfc :: Tensor dtype shape -> Tensor dtype shape
erfc _self = unsafePerformIO $ (ATen.cast1 ATen.erfc_t) _self

expm1 :: Tensor dtype shape -> Tensor dtype shape
expm1 _self = unsafePerformIO $ (ATen.cast1 ATen.expm1_t) _self

flatten :: Tensor dtype shape -> Int -> Int -> Tensor dtype shape
flatten _self _start_dim _end_dim =
  unsafePerformIO $ (ATen.cast3 ATen.flatten_tll) _self _start_dim _end_dim

frac :: Tensor dtype shape -> Tensor dtype shape
frac _self = unsafePerformIO $ (ATen.cast1 ATen.frac_t) _self

full_like :: Tensor dtype shape -> Float -> Tensor dtype shape
full_like _self _fill_value =
  unsafePerformIO $ (ATen.cast2 ATen.full_like_ts) _self _fill_value

grid_sampler
  :: Tensor dtype shape
  -> Tensor dtype shape
  -> Int
  -> Int
  -> Tensor dtype shape
grid_sampler _input _grid _interpolation_mode _padding_mode =
  unsafePerformIO $ (ATen.cast4 ATen.grid_sampler_ttll) _input
                                                        _grid
                                                        _interpolation_mode
                                                        _padding_mode

grid_sampler_2d
  :: Tensor dtype shape
  -> Tensor dtype shape
  -> Int
  -> Int
  -> Tensor dtype shape
grid_sampler_2d _input _grid _interpolation_mode _padding_mode =
  unsafePerformIO $ (ATen.cast4 ATen.grid_sampler_2d_ttll) _input
                                                           _grid
                                                           _interpolation_mode
                                                           _padding_mode

grid_sampler_3d
  :: Tensor dtype shape
  -> Tensor dtype shape
  -> Int
  -> Int
  -> Tensor dtype shape
grid_sampler_3d _input _grid _interpolation_mode _padding_mode =
  unsafePerformIO $ (ATen.cast4 ATen.grid_sampler_3d_ttll) _input
                                                           _grid
                                                           _interpolation_mode
                                                           _padding_mode

hinge_embedding_loss
  :: Tensor dtype shape
  -> Tensor dtype shape
  -> Double
  -> Int
  -> Tensor dtype shape
hinge_embedding_loss _self _target _margin _reduction =
  unsafePerformIO $ (ATen.cast4 ATen.hinge_embedding_loss_ttdl) _self
                                                                _target
                                                                _margin
                                                                _reduction

ger :: Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape
ger _self _vec2 = unsafePerformIO $ (ATen.cast2 ATen.ger_tt) _self _vec2

group_norm
  :: Tensor dtype shape
  -> Int
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Double
  -> Bool
  -> Tensor dtype shape
group_norm _input _num_groups _weight _bias _eps _cudnn_enabled =
  unsafePerformIO $ (ATen.cast6 ATen.group_norm_tlttdb) _input
                                                        _num_groups
                                                        _weight
                                                        _bias
                                                        _eps
                                                        _cudnn_enabled

fft :: Tensor dtype shape -> Int -> Bool -> Tensor dtype shape
fft _self _signal_ndim _normalized =
  unsafePerformIO $ (ATen.cast3 ATen.fft_tlb) _self _signal_ndim _normalized

ifft :: Tensor dtype shape -> Int -> Bool -> Tensor dtype shape
ifft _self _signal_ndim _normalized =
  unsafePerformIO $ (ATen.cast3 ATen.ifft_tlb) _self _signal_ndim _normalized

rfft :: Tensor dtype shape -> Int -> Bool -> Bool -> Tensor dtype shape
rfft _self _signal_ndim _normalized _onesided =
  unsafePerformIO
    $ (ATen.cast4 ATen.rfft_tlbb) _self _signal_ndim _normalized _onesided

irfft
  :: Tensor dtype shape -> Int -> Bool -> Bool -> [Int] -> Tensor dtype shape
irfft _self _signal_ndim _normalized _onesided _signal_sizes =
  unsafePerformIO $ (ATen.cast5 ATen.irfft_tlbbl) _self
                                                  _signal_ndim
                                                  _normalized
                                                  _onesided
                                                  _signal_sizes

index :: Tensor dtype shape -> [Tensor dtype shape] -> Tensor dtype shape
index _self _indices =
  unsafePerformIO $ (ATen.cast2 ATen.index_tl) _self _indices

index_copy
  :: Tensor dtype shape
  -> Int
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
index_copy _self _dim _index _source =
  unsafePerformIO $ (ATen.cast4 ATen.index_copy_tltt) _self _dim _index _source

index_put
  :: Tensor dtype shape
  -> [Tensor dtype shape]
  -> Tensor dtype shape
  -> Bool
  -> Tensor dtype shape
index_put _self _indices _values _accumulate =
  unsafePerformIO
    $ (ATen.cast4 ATen.index_put_tltb) _self _indices _values _accumulate

instance_norm
  :: Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Bool
  -> Double
  -> Double
  -> Bool
  -> Tensor dtype shape
instance_norm _input _weight _bias _running_mean _running_var _use_input_stats _momentum _eps _cudnn_enabled
  = unsafePerformIO $ (ATen.cast9 ATen.instance_norm_tttttbddb)
    _input
    _weight
    _bias
    _running_mean
    _running_var
    _use_input_stats
    _momentum
    _eps
    _cudnn_enabled



-- |
-- >>> t = isclose (ones :: Tensor Float '[3,2]) (ones :: Tensor Float '[3,2]) 0.1 0.1 False
-- >>> (dtype t,shape t)
-- (Bool,[3,2])
isclose
  :: Tensor dtype shape
  -> Tensor dtype shape
  -> Double
  -> Double
  -> Bool
  -> Tensor Bool shape
isclose _self _other _rtol _atol _equal_nan =
  unsafePerformIO
    $ (ATen.cast5 ATen.isclose_ttddb) _self _other _rtol _atol _equal_nan

isnan :: Tensor dtype shape -> Tensor Bool shape
isnan _self = unsafePerformIO $ (ATen.cast1 ATen.isnan_t) _self

is_distributed :: Tensor dtype shape -> Bool
is_distributed _self =
  unsafePerformIO $ (ATen.cast1 ATen.is_distributed_t) _self

is_floating_point :: Tensor dtype shape -> Bool
is_floating_point _self =
  unsafePerformIO $ (ATen.cast1 ATen.is_floating_point_t) _self

is_complex :: Tensor dtype shape -> Bool
is_complex _self = unsafePerformIO $ (ATen.cast1 ATen.is_complex_t) _self

is_nonzero :: Tensor dtype shape -> Bool
is_nonzero _self = unsafePerformIO $ (ATen.cast1 ATen.is_nonzero_t) _self

is_same_size :: Tensor dtype shape -> Tensor dtype shape -> Bool
is_same_size _self _other =
  unsafePerformIO $ (ATen.cast2 ATen.is_same_size_tt) _self _other

is_signed :: Tensor dtype shape -> Bool
is_signed _self = unsafePerformIO $ (ATen.cast1 ATen.is_signed_t) _self

kl_div :: Tensor dtype shape -> Tensor dtype shape -> Int -> Tensor dtype shape
kl_div _self _target _reduction =
  unsafePerformIO $ (ATen.cast3 ATen.kl_div_ttl) _self _target _reduction

kthvalue
  :: Tensor dtype shape
  -> Int
  -> Int
  -> Bool
  -> (Tensor dtype shape, Tensor dtype shape)
kthvalue _self _k _dim _keepdim =
  unsafePerformIO $ (ATen.cast4 ATen.kthvalue_tllb) _self _k _dim _keepdim

layer_norm
  :: Tensor dtype shape
  -> [Int]
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Double
  -> Bool
  -> Tensor dtype shape
layer_norm _input _normalized_shape _weight _bias _eps _cudnn_enable =
  unsafePerformIO $ (ATen.cast6 ATen.layer_norm_tlttdb) _input
                                                        _normalized_shape
                                                        _weight
                                                        _bias
                                                        _eps
                                                        _cudnn_enable

native_layer_norm
  :: Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Int
  -> Int
  -> Double
  -> ( Tensor dtype shape
     , Tensor dtype shape
     , Tensor dtype shape
     )
native_layer_norm _input _weight _bias _M _N _eps =
  unsafePerformIO $ (ATen.cast6 ATen.native_layer_norm_tttlld) _input
                                                               _weight
                                                               _bias
                                                               _M
                                                               _N
                                                               _eps

-- linear :: Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape
-- linear _input _weight _bias = unsafePerformIO $ (ATen.cast3 ATen.linear_ttt) _input _weight _bias

mkldnn_linear
  :: Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
mkldnn_linear _input _weight _bias =
  unsafePerformIO $ (ATen.cast3 ATen.mkldnn_linear_ttt) _input _weight _bias

fbgemm_linear_int8_weight
  :: Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Float
  -> Float
  -> Tensor dtype shape
  -> Tensor dtype shape
fbgemm_linear_int8_weight _input _weight _packed _col_offsets _weight_scale _weight_zero_point _bias
  = unsafePerformIO $ (ATen.cast7 ATen.fbgemm_linear_int8_weight_ttttsst)
    _input
    _weight
    _packed
    _col_offsets
    _weight_scale
    _weight_zero_point
    _bias

fbgemm_linear_quantize_weight
  :: Tensor dtype shape -> (Tensor dtype shape, Tensor dtype shape, Double, Int)
fbgemm_linear_quantize_weight _input =
  unsafePerformIO $ (ATen.cast1 ATen.fbgemm_linear_quantize_weight_t) _input

fbgemm_pack_gemm_matrix_fp16 :: Tensor dtype shape -> Tensor dtype shape
fbgemm_pack_gemm_matrix_fp16 _input =
  unsafePerformIO $ (ATen.cast1 ATen.fbgemm_pack_gemm_matrix_fp16_t) _input

fbgemm_linear_fp16_weight
  :: Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
fbgemm_linear_fp16_weight _input _packed_weight _bias =
  unsafePerformIO $ (ATen.cast3 ATen.fbgemm_linear_fp16_weight_ttt)
    _input
    _packed_weight
    _bias

fbgemm_pack_quantized_matrix
  :: Tensor dtype shape -> Int -> Int -> Tensor dtype shape
fbgemm_pack_quantized_matrix _input _K _N =
  unsafePerformIO
    $ (ATen.cast3 ATen.fbgemm_pack_quantized_matrix_tll) _input _K _N

--fbgemm_is_cpu_supported :: Bool
--fbgemm_is_cpu_supported  = unsafePerformIO $ (cast0 ATen.fbgemm_is_cpu_supported) 

log :: Tensor dtype shape -> Tensor dtype shape
log _self = unsafePerformIO $ (ATen.cast1 ATen.log_t) _self

logdet :: Tensor dtype shape -> Tensor dtype shape
logdet _self = unsafePerformIO $ (ATen.cast1 ATen.logdet_t) _self

logsumexp :: Tensor dtype shape -> Int -> Bool -> Tensor dtype shape
logsumexp _self _dim _keepdim =
  unsafePerformIO $ (ATen.cast3 ATen.logsumexp_tlb) _self _dim _keepdim

margin_ranking_loss
  :: Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Double
  -> Int
  -> Tensor dtype shape
margin_ranking_loss _input1 _input2 _target _margin _reduction =
  unsafePerformIO $ (ATen.cast5 ATen.margin_ranking_loss_tttdl) _input1
                                                                _input2
                                                                _target
                                                                _margin
                                                                _reduction

matrix_power :: Tensor dtype shape -> Int -> Tensor dtype shape
matrix_power _self _n =
  unsafePerformIO $ (ATen.cast2 ATen.matrix_power_tl) _self _n

max_values :: Tensor dtype shape -> Int -> Bool -> Tensor dtype shape
max_values _self _dim _keepdim =
  unsafePerformIO $ (ATen.cast3 ATen.max_values_tlb) _self _dim _keepdim

max_pool1d_with_indices
  :: Tensor dtype shape
  -> Int
  -> Int
  -> Int
  -> Int
  -> Bool
  -> (Tensor dtype shape, Tensor dtype shape)
max_pool1d_with_indices _self _kernel_size _stride _padding _dilation _ceil_mode
  = unsafePerformIO $ (ATen.cast6 ATen.max_pool1d_with_indices_tllllb)
    _self
    _kernel_size
    _stride
    _padding
    _dilation
    _ceil_mode

max_pool1d
  :: Tensor dtype shape
  -> Int
  -> Int
  -> Int
  -> Int
  -> Bool
  -> Tensor dtype shape
max_pool1d _self _kernel_size _stride _padding _dilation _ceil_mode =
  unsafePerformIO $ (ATen.cast6 ATen.max_pool1d_tllllb) _self
                                                        _kernel_size
                                                        _stride
                                                        _padding
                                                        _dilation
                                                        _ceil_mode

max_pool2d
  :: Tensor dtype shape
  -> (Int, Int)
  -> (Int, Int)
  -> (Int, Int)
  -> (Int, Int)
  -> Bool
  -> Tensor dtype shape
max_pool2d _self _kernel_size _stride _padding _dilation _ceil_mode =
  unsafePerformIO $ (ATen.cast6 ATen.max_pool2d_tllllb) _self
                                                        _kernel_size
                                                        _stride
                                                        _padding
                                                        _dilation
                                                        _ceil_mode

mkldnn_max_pool2d
  :: Tensor dtype shape
  -> (Int, Int)
  -> (Int, Int)
  -> (Int, Int)
  -> (Int, Int)
  -> Bool
  -> Tensor dtype shape
mkldnn_max_pool2d _self _kernel_size _stride _padding _dilation _ceil_mode =
  unsafePerformIO $ (ATen.cast6 ATen.mkldnn_max_pool2d_tllllb) _self
                                                               _kernel_size
                                                               _stride
                                                               _padding
                                                               _dilation
                                                               _ceil_mode

quantized_max_pool2d
  :: Tensor dtype shape
  -> (Int, Int)
  -> (Int, Int)
  -> (Int, Int)
  -> (Int, Int)
  -> Tensor dtype shape
quantized_max_pool2d _self _kernel_size _stride _padding _dilation =
  unsafePerformIO $ (ATen.cast5 ATen.quantized_max_pool2d_tllll) _self
                                                                 _kernel_size
                                                                 _stride
                                                                 _padding
                                                                 _dilation

max_pool3d
  :: Tensor dtype shape
  -> (Int, Int, Int)
  -> (Int, Int, Int)
  -> (Int, Int, Int)
  -> (Int, Int, Int)
  -> Bool
  -> Tensor dtype shape
max_pool3d _self _kernel_size _stride _padding _dilation _ceil_mode =
  unsafePerformIO $ (ATen.cast6 ATen.max_pool3d_tllllb) _self
                                                        _kernel_size
                                                        _stride
                                                        _padding
                                                        _dilation
                                                        _ceil_mode

min_values :: Tensor dtype shape -> Int -> Bool -> Tensor dtype shape
min_values _self _dim _keepdim =
  unsafePerformIO $ (ATen.cast3 ATen.min_values_tlb) _self _dim _keepdim

mkldnn_convolution
  :: Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> [Int]
  -> [Int]
  -> [Int]
  -> Int
  -> Tensor dtype shape
mkldnn_convolution _self _weight _bias _padding _stride _dilation _groups =
  unsafePerformIO $ (ATen.cast7 ATen.mkldnn_convolution_tttllll) _self
                                                                 _weight
                                                                 _bias
                                                                 _padding
                                                                 _stride
                                                                 _dilation
                                                                 _groups

miopen_batch_norm
  :: Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Bool
  -> Double
  -> Double
  -> ( Tensor dtype shape
     , Tensor dtype shape
     , Tensor dtype shape
     )
miopen_batch_norm _input _weight _bias _running_mean _running_var _training _exponential_average_factor _epsilon
  = unsafePerformIO $ (ATen.cast8 ATen.miopen_batch_norm_tttttbdd)
    _input
    _weight
    _bias
    _running_mean
    _running_var
    _training
    _exponential_average_factor
    _epsilon

miopen_convolution
  :: Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> [Int]
  -> [Int]
  -> [Int]
  -> Int
  -> Bool
  -> Bool
  -> Tensor dtype shape
miopen_convolution _self _weight _bias _padding _stride _dilation _groups _benchmark _deterministic
  = unsafePerformIO $ (ATen.cast9 ATen.miopen_convolution_tttllllbb)
    _self
    _weight
    _bias
    _padding
    _stride
    _dilation
    _groups
    _benchmark
    _deterministic

miopen_convolution_transpose
  :: Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> [Int]
  -> [Int]
  -> [Int]
  -> [Int]
  -> Int
  -> Bool
  -> Bool
  -> Tensor dtype shape
miopen_convolution_transpose _self _weight _bias _padding _output_padding _stride _dilation _groups _benchmark _deterministic
  = unsafePerformIO $ (ATen.cast10 ATen.miopen_convolution_transpose_tttlllllbb)
    _self
    _weight
    _bias
    _padding
    _output_padding
    _stride
    _dilation
    _groups
    _benchmark
    _deterministic

miopen_depthwise_convolution
  :: Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> [Int]
  -> [Int]
  -> [Int]
  -> Int
  -> Bool
  -> Bool
  -> Tensor dtype shape
miopen_depthwise_convolution _self _weight _bias _padding _stride _dilation _groups _benchmark _deterministic
  = unsafePerformIO $ (ATen.cast9 ATen.miopen_depthwise_convolution_tttllllbb)
    _self
    _weight
    _bias
    _padding
    _stride
    _dilation
    _groups
    _benchmark
    _deterministic

miopen_rnn _input _weight _weight_stride0 _hx _cx _mode _hidden_size _num_layers _batch_first _dropout _train _bidirectional _batch_sizes _dropout_state
  = unsafePerformIO $ (ATen.cast14 ATen.miopen_rnn_tllttlllbdbblt)
    _input
    _weight
    _weight_stride0
    _hx
    _cx
    _mode
    _hidden_size
    _num_layers
    _batch_first
    _dropout
    _train
    _bidirectional
    _batch_sizes
    _dropout_state

mode
  :: Tensor dtype shape
  -> Int
  -> Bool
  -> (Tensor dtype shape, Tensor dtype shape)
mode _self _dim _keepdim =
  unsafePerformIO $ (ATen.cast3 ATen.mode_tlb) _self _dim _keepdim

mv :: Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape
mv _self _vec = unsafePerformIO $ (ATen.cast2 ATen.mv_tt) _self _vec

mvlgamma :: Tensor dtype shape -> Int -> Tensor dtype shape
mvlgamma _self _p = unsafePerformIO $ (ATen.cast2 ATen.mvlgamma_tl) _self _p

narrow :: Tensor dtype shape -> Int -> Int -> Int -> Tensor dtype shape
narrow _self _dim _start _length =
  unsafePerformIO $ (ATen.cast4 ATen.narrow_tlll) _self _dim _start _length

native_batch_norm
  :: Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Bool
  -> Double
  -> Double
  -> ( Tensor dtype shape
     , Tensor dtype shape
     , Tensor dtype shape
     )
native_batch_norm _input _weight _bias _running_mean _running_var _training _momentum _eps
  = unsafePerformIO $ (ATen.cast8 ATen.native_batch_norm_tttttbdd)
    _input
    _weight
    _bias
    _running_mean
    _running_var
    _training
    _momentum
    _eps

batch_norm_stats
  :: Tensor dtype shape -> Double -> (Tensor dtype shape, Tensor dtype shape)
batch_norm_stats _input _eps =
  unsafePerformIO $ (ATen.cast2 ATen.batch_norm_stats_td) _input _eps

batch_norm_elemt
  :: Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Double
  -> Tensor dtype shape
batch_norm_elemt _input _weight _bias _mean _invstd _eps =
  unsafePerformIO $ (ATen.cast6 ATen.batch_norm_elemt_tttttd) _input
                                                              _weight
                                                              _bias
                                                              _mean
                                                              _invstd
                                                              _eps

batch_norm_gather_stats
  :: Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Double
  -> Double
  -> Int
  -> (Tensor dtype shape, Tensor dtype shape)
batch_norm_gather_stats _input _mean _invstd _running_mean _running_var _momentum _eps _count
  = unsafePerformIO $ (ATen.cast8 ATen.batch_norm_gather_stats_tttttddl)
    _input
    _mean
    _invstd
    _running_mean
    _running_var
    _momentum
    _eps
    _count

batch_norm_gather_stats_with_counts
  :: Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Double
  -> Double
  -> [Int]
  -> (Tensor dtype shape, Tensor dtype shape)
batch_norm_gather_stats_with_counts _input _mean _invstd _running_mean _running_var _momentum _eps _counts
  = unsafePerformIO
    $ (ATen.cast8 ATen.batch_norm_gather_stats_with_counts_tttttddl)
        _input
        _mean
        _invstd
        _running_mean
        _running_var
        _momentum
        _eps
        _counts

batch_norm_update_stats
  :: Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Double
  -> (Tensor dtype shape, Tensor dtype shape)
batch_norm_update_stats _input _running_mean _running_var _momentum =
  unsafePerformIO $ (ATen.cast4 ATen.batch_norm_update_stats_tttd)
    _input
    _running_mean
    _running_var
    _momentum

ones_like :: Tensor dtype shape -> Tensor dtype shape
ones_like _self = unsafePerformIO $ (ATen.cast1 ATen.ones_like_t) _self

pairwise_distance
  :: Tensor dtype shape
  -> Tensor dtype shape
  -> Double
  -> Double
  -> Bool
  -> Tensor dtype shape
pairwise_distance _x1 _x2 _p _eps _keepdim =
  unsafePerformIO
    $ (ATen.cast5 ATen.pairwise_distance_ttddb) _x1 _x2 _p _eps _keepdim

cdist
  :: Tensor dtype shape -> Tensor dtype shape -> Double -> Tensor dtype shape
cdist _x1 _x2 _p = unsafePerformIO $ (ATen.cast3 ATen.cdist_ttd) _x1 _x2 _p

pdist :: Tensor dtype shape -> Double -> Tensor dtype shape
pdist _self _p = unsafePerformIO $ (ATen.cast2 ATen.pdist_td) _self _p

cosine_similarity
  :: Tensor dtype shape
  -> Tensor dtype shape
  -> Int
  -> Double
  -> Tensor dtype shape
cosine_similarity _x1 _x2 _dim _eps =
  unsafePerformIO $ (ATen.cast4 ATen.cosine_similarity_ttld) _x1 _x2 _dim _eps

pixel_shuffle :: Tensor dtype shape -> Int -> Tensor dtype shape
pixel_shuffle _self _upscale_factor =
  unsafePerformIO $ (ATen.cast2 ATen.pixel_shuffle_tl) _self _upscale_factor

pin_memory :: Tensor dtype shape -> Tensor dtype shape
pin_memory _self = unsafePerformIO $ (ATen.cast1 ATen.pin_memory_t) _self

pinverse :: Tensor dtype shape -> Double -> Tensor dtype shape
pinverse _self _rcond =
  unsafePerformIO $ (ATen.cast2 ATen.pinverse_td) _self _rcond

poisson_nll_loss
  :: Tensor dtype shape
  -> Tensor dtype shape
  -> Bool
  -> Bool
  -> Double
  -> Int
  -> Tensor dtype shape
poisson_nll_loss _input _target _log_input _full _eps _reduction =
  unsafePerformIO $ (ATen.cast6 ATen.poisson_nll_loss_ttbbdl) _input
                                                              _target
                                                              _log_input
                                                              _full
                                                              _eps
                                                              _reduction

rand_like :: Tensor dtype shape -> IO (Tensor dtype shape)
rand_like _self = (ATen.cast1 ATen.rand_like_t) _self

randn_like :: Tensor dtype shape -> IO (Tensor dtype shape)
randn_like _self = (ATen.cast1 ATen.randn_like_t) _self

reciprocal :: Tensor dtype shape -> Tensor dtype shape
reciprocal _self = unsafePerformIO $ (ATen.cast1 ATen.reciprocal_t) _self

neg :: Tensor dtype shape -> Tensor dtype shape
neg _self = unsafePerformIO $ (ATen.cast1 ATen.neg_t) _self

round :: Tensor dtype shape -> Tensor dtype shape
round _self = unsafePerformIO $ (ATen.cast1 ATen.round_t) _self

prelu :: Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape
prelu _self _weight =
  unsafePerformIO $ (ATen.cast2 ATen.prelu_tt) _self _weight

gelu :: Tensor dtype shape -> Tensor dtype shape
gelu _self = unsafePerformIO $ (ATen.cast1 ATen.gelu_t) _self

hardshrink :: Tensor dtype shape -> Float -> Tensor dtype shape
hardshrink _self _lambd =
  unsafePerformIO $ (ATen.cast2 ATen.hardshrink_ts) _self _lambd

rsqrt :: Tensor dtype shape -> Tensor dtype shape
rsqrt _self = unsafePerformIO $ (ATen.cast1 ATen.rsqrt_t) _self

celu :: Tensor dtype shape -> Float -> Tensor dtype shape
celu _self _alpha = unsafePerformIO $ (ATen.cast2 ATen.celu_ts) _self _alpha

slice :: Tensor dtype shape -> Int -> Int -> Int -> Int -> Tensor dtype shape
slice _self _dim _start _end _step =
  unsafePerformIO $ (ATen.cast5 ATen.slice_tllll) _self _dim _start _end _step

slogdet :: Tensor dtype shape -> (Tensor dtype shape, Tensor dtype shape)
slogdet _self = unsafePerformIO $ (ATen.cast1 ATen.slogdet_t) _self

smm :: Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape
smm _self _mat2 = unsafePerformIO $ (ATen.cast2 ATen.smm_tt) _self _mat2

split :: Tensor dtype shape -> Int -> Int -> [Tensor dtype shape]
split _self _split_size _dim =
  unsafePerformIO $ (ATen.cast3 ATen.split_tll) _self _split_size _dim

split_with_sizes :: Tensor dtype shape -> [Int] -> Int -> [Tensor dtype shape]
split_with_sizes _self _split_sizes _dim =
  unsafePerformIO
    $ (ATen.cast3 ATen.split_with_sizes_tll) _self _split_sizes _dim

sspaddmm
  :: Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Float
  -> Float
  -> Tensor dtype shape
sspaddmm _self _mat1 _mat2 _beta _alpha =
  unsafePerformIO
    $ (ATen.cast5 ATen.sspaddmm_tttss) _self _mat1 _mat2 _beta _alpha

stack :: [Tensor dtype shape] -> Int -> Tensor dtype shape
stack _tensors _dim =
  unsafePerformIO $ (ATen.cast2 ATen.stack_ll) _tensors _dim

stft
  :: Tensor dtype shape
  -> Int
  -> Int
  -> Int
  -> Tensor dtype shape
  -> Bool
  -> Bool
  -> Tensor dtype shape
stft _self _n_fft _hop_length _win_length _window _normalized _onesided =
  unsafePerformIO $ (ATen.cast7 ATen.stft_tllltbb) _self
                                                   _n_fft
                                                   _hop_length
                                                   _win_length
                                                   _window
                                                   _normalized
                                                   _onesided

stride :: Tensor dtype shape -> Int -> Int
stride _self _dim = unsafePerformIO $ (ATen.cast2 ATen.stride_tl) _self _dim

t :: Tensor dtype shape -> Tensor dtype shape
t _self = unsafePerformIO $ (ATen.cast1 ATen.t_t) _self

tan :: Tensor dtype shape -> Tensor dtype shape
tan _self = unsafePerformIO $ (ATen.cast1 ATen.tan_t) _self

tensordot
  :: Tensor dtype shape
  -> Tensor dtype shape
  -> [Int]
  -> [Int]
  -> Tensor dtype shape
tensordot _self _other _dims_self _dims_other =
  unsafePerformIO
    $ (ATen.cast4 ATen.tensordot_ttll) _self _other _dims_self _dims_other

threshold :: Tensor dtype shape -> Float -> Float -> Tensor dtype shape
threshold _self _threshold _value =
  unsafePerformIO $ (ATen.cast3 ATen.threshold_tss) _self _threshold _value

one_hot :: Tensor dtype shape -> Int -> Tensor dtype shape
one_hot _self _num_classes =
  unsafePerformIO $ (ATen.cast2 ATen.one_hot_tl) _self _num_classes

flip :: Tensor dtype shape -> [Int] -> Tensor dtype shape
flip _self _dims = unsafePerformIO $ (ATen.cast2 ATen.flip_tl) _self _dims

roll :: Tensor dtype shape -> Int -> Int -> Tensor dtype shape
roll _self _shifts _dims =
  unsafePerformIO $ (ATen.cast3 ATen.roll_tll) _self _shifts _dims

rot90 :: Tensor dtype shape -> Int -> [Int] -> Tensor dtype shape
rot90 _self _k _dims =
  unsafePerformIO $ (ATen.cast3 ATen.rot90_tll) _self _k _dims

triplet_margin_loss
  :: Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Double
  -> Double
  -> Double
  -> Bool
  -> Int
  -> Tensor dtype shape
triplet_margin_loss _anchor _positive _negative _margin _p _eps _swap _reduction
  = unsafePerformIO $ (ATen.cast8 ATen.triplet_margin_loss_tttdddbl)
    _anchor
    _positive
    _negative
    _margin
    _p
    _eps
    _swap
    _reduction

trunc :: Tensor dtype shape -> Tensor dtype shape
trunc _self = unsafePerformIO $ (ATen.cast1 ATen.trunc_t) _self

unique_dim
  :: Tensor dtype shape
  -> Int
  -> Bool
  -> Bool
  -> Bool
  -> (Tensor dtype shape, Tensor dtype shape, Tensor dtype shape)
unique_dim _self _dim _sorted _return_inverse _return_counts =
  unsafePerformIO $ (ATen.cast5 ATen.unique_dim_tlbbb) _self
                                                       _dim
                                                       _sorted
                                                       _return_inverse
                                                       _return_counts

unique_consecutive
  :: Tensor dtype shape
  -> Bool
  -> Bool
  -> Int
  -> (Tensor dtype shape, Tensor dtype shape, Tensor dtype shape)
unique_consecutive _self _return_inverse _return_counts _dim =
  unsafePerformIO $ (ATen.cast4 ATen.unique_consecutive_tbbl) _self
                                                              _return_inverse
                                                              _return_counts
                                                              _dim

unique_dim_consecutive
  :: Tensor dtype shape
  -> Int
  -> Bool
  -> Bool
  -> (Tensor dtype shape, Tensor dtype shape, Tensor dtype shape)
unique_dim_consecutive _self _dim _return_inverse _return_counts =
  unsafePerformIO $ (ATen.cast4 ATen.unique_dim_consecutive_tlbb)
    _self
    _dim
    _return_inverse
    _return_counts

unsqueeze :: Tensor dtype shape -> Int -> Tensor dtype shape
unsqueeze _self _dim =
  unsafePerformIO $ (ATen.cast2 ATen.unsqueeze_tl) _self _dim

where'
  :: Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
where' _condition _self _other =
  unsafePerformIO $ (ATen.cast3 ATen.where_ttt) _condition _self _other

where_ :: Tensor dtype shape -> [Tensor dtype shape]
where_ _condition = unsafePerformIO $ (ATen.cast1 ATen.where_t) _condition

norm_except_dim :: Tensor dtype shape -> Int -> Int -> Tensor dtype shape
norm_except_dim _v _pow _dim =
  unsafePerformIO $ (ATen.cast3 ATen.norm_except_dim_tll) _v _pow _dim

zeros_like :: Tensor dtype shape -> Tensor dtype shape
zeros_like _self = unsafePerformIO $ (ATen.cast1 ATen.zeros_like_t) _self

native_norm :: Tensor dtype shape -> Float -> Tensor dtype shape
native_norm _self _p =
  unsafePerformIO $ (ATen.cast2 ATen.native_norm_ts) _self _p

clone :: Tensor dtype shape -> Tensor dtype shape
clone _self = unsafePerformIO $ (ATen.cast1 ATen.clone_t) _self

s_native_addmm
  :: Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Float
  -> Float
  -> Tensor dtype shape
s_native_addmm _self _mat1 _mat2 _beta _alpha =
  unsafePerformIO
    $ (ATen.cast5 ATen.s_native_addmm_tttss) _self _mat1 _mat2 _beta _alpha

addmm
  :: Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Float
  -> Float
  -> Tensor dtype shape
addmm _self _mat1 _mat2 _beta _alpha =
  unsafePerformIO $ (ATen.cast5 ATen.addmm_tttss) _self _mat1 _mat2 _beta _alpha

hspmm :: Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape
hspmm _mat1 _mat2 = unsafePerformIO $ (ATen.cast2 ATen.hspmm_tt) _mat1 _mat2

numel :: Tensor dtype shape -> Int
numel _self = unsafePerformIO $ (ATen.cast1 ATen.numel_t) _self

unbind :: Tensor dtype shape -> Int -> [Tensor dtype shape]
unbind _self _dim = unsafePerformIO $ (ATen.cast2 ATen.unbind_tl) _self _dim

mkldnn_reorder_conv2d_weight
  :: Tensor dtype shape
  -> (Int, Int)
  -> (Int, Int)
  -> (Int, Int)
  -> Int
  -> Tensor dtype shape
mkldnn_reorder_conv2d_weight _self _padding _stride _dilation _groups =
  unsafePerformIO $ (ATen.cast5 ATen.mkldnn_reorder_conv2d_weight_tllll)
    _self
    _padding
    _stride
    _dilation
    _groups

--quantize_linear :: Tensor dtype shape -> Double -> Int -> DType -> Tensor dtype shape
--quantize_linear _self _scale _zero_point _dtype = unsafePerformIO $ (ATen.cast4 ATen.quantize_linear_tdls) _self _scale _zero_point _dtype

--quantize_linear_per_channel :: Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> [Int] -> DType -> Tensor dtype shape
--quantize_linear_per_channel _self _scales _zero_points _axis _dtype = unsafePerformIO $ (ATen.cast5 ATen.quantize_linear_per_channel_tttls) _self _scales _zero_points _axis _dtype

dequantize :: Tensor dtype shape -> Tensor dtype shape
dequantize _self = unsafePerformIO $ (ATen.cast1 ATen.dequantize_t) _self

q_scale :: Tensor dtype shape -> Double
q_scale _self = unsafePerformIO $ (ATen.cast1 ATen.q_scale_t) _self

q_zero_point :: Tensor dtype shape -> Int
q_zero_point _self = unsafePerformIO $ (ATen.cast1 ATen.q_zero_point_t) _self

int_repr :: Tensor dtype shape -> Tensor dtype shape
int_repr _self = unsafePerformIO $ (ATen.cast1 ATen.int_repr_t) _self

fake_quantize_per_tensor_affine
  :: Tensor dtype shape -> Double -> Int -> Int -> Int -> Tensor dtype shape
fake_quantize_per_tensor_affine _self _scale _zero_point _quant_min _quant_max
  = unsafePerformIO $ (ATen.cast5 ATen.fake_quantize_per_tensor_affine_tdlll)
    _self
    _scale
    _zero_point
    _quant_min
    _quant_max

meshgrid :: [Tensor dtype shape] -> [Tensor dtype shape]
meshgrid _tensors = unsafePerformIO $ (ATen.cast1 ATen.meshgrid_l) _tensors

cartesian_prod :: [Tensor dtype shape] -> Tensor dtype shape
cartesian_prod _tensors =
  unsafePerformIO $ (ATen.cast1 ATen.cartesian_prod_l) _tensors

combinations :: Tensor dtype shape -> Int -> Bool -> Tensor dtype shape
combinations _self _r _with_replacement =
  unsafePerformIO
    $ (ATen.cast3 ATen.combinations_tlb) _self _r _with_replacement

lstm_cell
  :: Tensor dtype shape
  -> [Tensor dtype shape]
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> (Tensor dtype shape, Tensor dtype shape)
lstm_cell _input _hx _w_ih _w_hh _b_ih _b_hh =
  unsafePerformIO
    $ (ATen.cast6 ATen.lstm_cell_tltttt) _input _hx _w_ih _w_hh _b_ih _b_hh

gru_cell
  :: Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
gru_cell _input _hx _w_ih _w_hh _b_ih _b_hh =
  unsafePerformIO
    $ (ATen.cast6 ATen.gru_cell_tttttt) _input _hx _w_ih _w_hh _b_ih _b_hh

rnn_tanh_cell
  :: Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
rnn_tanh_cell _input _hx _w_ih _w_hh _b_ih _b_hh =
  unsafePerformIO $ (ATen.cast6 ATen.rnn_tanh_cell_tttttt) _input
                                                           _hx
                                                           _w_ih
                                                           _w_hh
                                                           _b_ih
                                                           _b_hh

rnn_relu_cell
  :: Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
rnn_relu_cell _input _hx _w_ih _w_hh _b_ih _b_hh =
  unsafePerformIO $ (ATen.cast6 ATen.rnn_relu_cell_tttttt) _input
                                                           _hx
                                                           _w_ih
                                                           _w_hh
                                                           _b_ih
                                                           _b_hh

--quantized_lstm :: Tensor dtype shape -> [Tensor dtype shape] -> [Tensor dtype shape] -> Bool -> Int -> Double -> Bool -> Bool -> Bool -> DType -> (Tensor dtype shape,Tensor dtype shape,Tensor dtype shape)
--quantized_lstm _input _hx _params _has_biases _num_layers _dropout _train _bidirectional _batch_first _dtype = unsafePerformIO $ (ATen.cast10 ATen.quantized_lstm_tllbldbbbs) _input _hx _params _has_biases _num_layers _dropout _train _bidirectional _batch_first _dtype

quantized_lstm_cell
  :: Tensor dtype shape
  -> [Tensor dtype shape]
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Float
  -> Float
  -> Float
  -> Float
  -> ( Tensor dtype shape
     , Tensor dtype shape
     )
quantized_lstm_cell _input _hx _w_ih _w_hh _b_ih _b_hh _packed_ih _packed_hh _col_offsets_ih _col_offsets_hh _scale_ih _scale_hh _zero_point_ih _zero_point_hh
  = unsafePerformIO $ (ATen.cast14 ATen.quantized_lstm_cell_tlttttttttssss)
    _input
    _hx
    _w_ih
    _w_hh
    _b_ih
    _b_hh
    _packed_ih
    _packed_hh
    _col_offsets_ih
    _col_offsets_hh
    _scale_ih
    _scale_hh
    _zero_point_ih
    _zero_point_hh

quantized_gru_cell
  :: Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Float
  -> Float
  -> Float
  -> Float
  -> Tensor dtype shape
quantized_gru_cell _input _hx _w_ih _w_hh _b_ih _b_hh _packed_ih _packed_hh _col_offsets_ih _col_offsets_hh _scale_ih _scale_hh _zero_point_ih _zero_point_hh
  = unsafePerformIO $ (ATen.cast14 ATen.quantized_gru_cell_ttttttttttssss)
    _input
    _hx
    _w_ih
    _w_hh
    _b_ih
    _b_hh
    _packed_ih
    _packed_hh
    _col_offsets_ih
    _col_offsets_hh
    _scale_ih
    _scale_hh
    _zero_point_ih
    _zero_point_hh

quantized_rnn_relu_cell
  :: Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Float
  -> Float
  -> Float
  -> Float
  -> Tensor dtype shape
quantized_rnn_relu_cell _input _hx _w_ih _w_hh _b_ih _b_hh _packed_ih _packed_hh _col_offsets_ih _col_offsets_hh _scale_ih _scale_hh _zero_point_ih _zero_point_hh
  = unsafePerformIO $ (ATen.cast14 ATen.quantized_rnn_relu_cell_ttttttttttssss)
    _input
    _hx
    _w_ih
    _w_hh
    _b_ih
    _b_hh
    _packed_ih
    _packed_hh
    _col_offsets_ih
    _col_offsets_hh
    _scale_ih
    _scale_hh
    _zero_point_ih
    _zero_point_hh

quantized_rnn_tanh_cell
  :: Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Float
  -> Float
  -> Float
  -> Float
  -> Tensor dtype shape
quantized_rnn_tanh_cell _input _hx _w_ih _w_hh _b_ih _b_hh _packed_ih _packed_hh _col_offsets_ih _col_offsets_hh _scale_ih _scale_hh _zero_point_ih _zero_point_hh
  = unsafePerformIO $ (ATen.cast14 ATen.quantized_rnn_tanh_cell_ttttttttttssss)
    _input
    _hx
    _w_ih
    _w_hh
    _b_ih
    _b_hh
    _packed_ih
    _packed_hh
    _col_offsets_ih
    _col_offsets_hh
    _scale_ih
    _scale_hh
    _zero_point_ih
    _zero_point_hh

masked_scatter
  :: Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
masked_scatter _self _mask _source =
  unsafePerformIO $ (ATen.cast3 ATen.masked_scatter_ttt) _self _mask _source

index_add
  :: Tensor dtype shape
  -> Int
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
index_add _self _dim _index _source =
  unsafePerformIO $ (ATen.cast4 ATen.index_add_tltt) _self _dim _index _source

scatter_add
  :: Tensor dtype shape
  -> Int
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
scatter_add _self _dim _index _src =
  unsafePerformIO $ (ATen.cast4 ATen.scatter_add_tltt) _self _dim _index _src

addbmm
  :: Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Float
  -> Float
  -> Tensor dtype shape
addbmm _self _batch1 _batch2 _beta _alpha =
  unsafePerformIO
    $ (ATen.cast5 ATen.addbmm_tttss) _self _batch1 _batch2 _beta _alpha

cross :: Tensor dtype shape -> Tensor dtype shape -> Int -> Tensor dtype shape
cross _self _other _dim =
  unsafePerformIO $ (ATen.cast3 ATen.cross_ttl) _self _other _dim

triu :: Tensor dtype shape -> Int -> Tensor dtype shape
triu _self _diagonal =
  unsafePerformIO $ (ATen.cast2 ATen.triu_tl) _self _diagonal

tril :: Tensor dtype shape -> Int -> Tensor dtype shape
tril _self _diagonal =
  unsafePerformIO $ (ATen.cast2 ATen.tril_tl) _self _diagonal

trace :: Tensor dtype shape -> Tensor dtype shape
trace _self = unsafePerformIO $ (ATen.cast1 ATen.trace_t) _self

take :: Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape
take _self _index = unsafePerformIO $ (ATen.cast2 ATen.take_tt) _self _index

index_select
  :: Tensor dtype shape -> Int -> Tensor dtype shape -> Tensor dtype shape
index_select _self _dim _index =
  unsafePerformIO $ (ATen.cast3 ATen.index_select_tlt) _self _dim _index

masked_select :: Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape
masked_select _self _mask =
  unsafePerformIO $ (ATen.cast2 ATen.masked_select_tt) _self _mask

nonzero :: Tensor dtype shape -> Tensor dtype shape
nonzero _self = unsafePerformIO $ (ATen.cast1 ATen.nonzero_t) _self

nonzero_numpy :: Tensor dtype shape -> [Tensor dtype shape]
nonzero_numpy _self = unsafePerformIO $ (ATen.cast1 ATen.nonzero_numpy_t) _self

gather
  :: Tensor dtype shape
  -> Int
  -> Tensor dtype shape
  -> Bool
  -> Tensor dtype shape
gather _self _dim _index _sparse_grad =
  unsafePerformIO $ (ATen.cast4 ATen.gather_tltb) _self _dim _index _sparse_grad

addcmul
  :: Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Float
  -> Tensor dtype shape
addcmul _self _tensor1 _tensor2 _value =
  unsafePerformIO
    $ (ATen.cast4 ATen.addcmul_ttts) _self _tensor1 _tensor2 _value

addcdiv
  :: Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Float
  -> Tensor dtype shape
addcdiv _self _tensor1 _tensor2 _value =
  unsafePerformIO
    $ (ATen.cast4 ATen.addcdiv_ttts) _self _tensor1 _tensor2 _value

lstsq
  :: Tensor dtype shape
  -> Tensor dtype shape
  -> (Tensor dtype shape, Tensor dtype shape)
lstsq _self _A = unsafePerformIO $ (ATen.cast2 ATen.lstsq_tt) _self _A

triangular_solve
  :: Tensor dtype shape
  -> Tensor dtype shape
  -> Bool
  -> Bool
  -> Bool
  -> (Tensor dtype shape, Tensor dtype shape)
triangular_solve _self _A _upper _transpose _unitriangular =
  unsafePerformIO $ (ATen.cast5 ATen.triangular_solve_ttbbb) _self
                                                             _A
                                                             _upper
                                                             _transpose
                                                             _unitriangular

qr :: Tensor dtype shape -> Bool -> (Tensor dtype shape, Tensor dtype shape)
qr _self _some = unsafePerformIO $ (ATen.cast2 ATen.qr_tb) _self _some

ormqr
  :: Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Bool
  -> Bool
  -> Tensor dtype shape
ormqr _self _input2 _input3 _left _transpose =
  unsafePerformIO
    $ (ATen.cast5 ATen.ormqr_tttbb) _self _input2 _input3 _left _transpose

lu_solve
  :: Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
lu_solve _self _LU_data _LU_pivots =
  unsafePerformIO $ (ATen.cast3 ATen.lu_solve_ttt) _self _LU_data _LU_pivots

lgamma :: Tensor dtype shape -> Tensor dtype shape
lgamma _self = unsafePerformIO $ (ATen.cast1 ATen.lgamma_t) _self

digamma :: Tensor dtype shape -> Tensor dtype shape
digamma _self = unsafePerformIO $ (ATen.cast1 ATen.digamma_t) _self

polygamma :: Int -> Tensor dtype shape -> Tensor dtype shape
polygamma _n _self = unsafePerformIO $ (ATen.cast2 ATen.polygamma_lt) _n _self

erfinv :: Tensor dtype shape -> Tensor dtype shape
erfinv _self = unsafePerformIO $ (ATen.cast1 ATen.erfinv_t) _self

dist :: Tensor dtype shape -> Tensor dtype shape -> Float -> Tensor dtype shape
dist _self _other _p =
  unsafePerformIO $ (ATen.cast3 ATen.dist_tts) _self _other _p

atan2 :: Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape
atan2 _self _other = unsafePerformIO $ (ATen.cast2 ATen.atan2_tt) _self _other

histc :: Tensor dtype shape -> Int -> Float -> Float -> Tensor dtype shape
histc _self _bins _min _max =
  unsafePerformIO $ (ATen.cast4 ATen.histc_tlss) _self _bins _min _max

-- |
-- >>> t = minAll (ones :: Tensor Float '[2,2])
-- >>> (dtype t,shape t)
-- (Float,[])
minAll :: Tensor dtype shape -> Tensor dtype '[]
minAll _self = unsafePerformIO $ (ATen.cast1 ATen.min_t) _self

maxAll :: Tensor dtype shape -> Tensor dtype '[]
maxAll _self = unsafePerformIO $ (ATen.cast1 ATen.max_t) _self

medianAll :: Tensor dtype shape -> Tensor dtype '[]
medianAll _self = unsafePerformIO $ (ATen.cast1 ATen.median_t) _self

sort
  :: Tensor dtype shape
  -> Int
  -> Bool
  -> (Tensor dtype shape, Tensor dtype shape)
sort _self _dim _descending =
  unsafePerformIO $ (ATen.cast3 ATen.sort_tlb) _self _dim _descending

argsort :: Tensor dtype shape -> Int -> Bool -> Tensor dtype shape
argsort _self _dim _descending =
  unsafePerformIO $ (ATen.cast3 ATen.argsort_tlb) _self _dim _descending

topk
  :: Tensor dtype shape
  -> Int
  -> Int
  -> Bool
  -> Bool
  -> (Tensor dtype shape, Tensor dtype shape)
topk _self _k _dim _largest _sorted =
  unsafePerformIO $ (ATen.cast5 ATen.topk_tllbb) _self _k _dim _largest _sorted

renorm :: Tensor dtype shape -> Float -> Int -> Float -> Tensor dtype shape
renorm _self _p _dim _maxnorm =
  unsafePerformIO $ (ATen.cast4 ATen.renorm_tsls) _self _p _dim _maxnorm

equal :: Tensor dtype shape -> Tensor dtype shape -> Bool
equal _self _other = unsafePerformIO $ (ATen.cast2 ATen.equal_tt) _self _other

alias :: Tensor dtype shape -> Tensor dtype shape
alias _self = unsafePerformIO $ (ATen.cast1 ATen.alias_t) _self

-- |
-- >>> t = l1Loss @D.ReduceNone @Float @[2,2] ones ones
-- >>> (dtype t,shape t)
-- (Float,[2,2])
-- >>> t = l1Loss @D.ReduceSum @Float @[2,2] ones ones
-- >>> (dtype t,shape t)
-- (Float,[])
l1Loss
  :: forall reduction dtype shape
   . (KnownReduction reduction)
  => Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype (ConditionalReduction shape reduction)
l1Loss self target = unsafePerformIO
  $ (ATen.cast3 ATen.l1_loss_ttl) self target (reductionVal @reduction)

-- | WIP, see https://github.com/pytorch/pytorch/blob/master/aten/src/THNN/generic/MultiMarginCriterion.c
-- >>> t = multiMarginLoss @D.ReduceNone @Float @3 @2 ones ones 1 1 ones :: Tensor Float '[3]
-- >>> (dtype t,shape t)
-- (Float,[3])
-- >>> t = multiMarginLoss @D.ReduceSum @Float @3 @2 ones ones 1 1 ones :: Tensor Float '[]
-- >>> (dtype t,shape t)
-- (Float,[])
multiMarginLoss
  :: forall reduction dtype n m
   . (KnownReduction reduction)
  => Tensor dtype [n, m]
  -> Tensor Int '[n]
  -> Int
  -> Float
  -> Tensor dtype '[]
  -> Tensor dtype (ConditionalReduction '[n] reduction)
multiMarginLoss self target p margin weight =
  unsafePerformIO $ (ATen.cast6 ATen.multi_margin_loss_ttsstl) self
                                                               target
                                                               p
                                                               margin
                                                               weight
                                                               (reductionVal @reduction)

-- | WIP, see https://github.com/pytorch/pytorch/blob/master/aten/src/THNN/generic/MultiLabelMarginCriterion.c
multilabel_margin_loss
  :: Tensor dtype shape -> Tensor dtype shape -> Int -> Tensor dtype shape
multilabel_margin_loss _self _target _reduction =
  unsafePerformIO
    $ (ATen.cast3 ATen.multilabel_margin_loss_ttl) _self _target _reduction

multilabel_margin_loss_forward
  :: Tensor dtype shape
  -> Tensor dtype shape
  -> Int
  -> (Tensor dtype shape, Tensor dtype shape)
multilabel_margin_loss_forward _self _target _reduction =
  unsafePerformIO $ (ATen.cast3 ATen.multilabel_margin_loss_forward_ttl)
    _self
    _target
    _reduction

-- | The negative log likelihood loss.
-- >>> input = randn @Float @[3, 5]
-- >>> target = 
-- >>> t = nllLoss @D.ReduceNone @Float @[2, 2] 
nllLoss
  :: forall reduction dtype shape
   . (KnownReduction reduction)
  => Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Int
  -> Tensor dtype shape
nllLoss input target weight ignoreIndex =
  unsafePerformIO $ (ATen.cast5 ATen.nll_loss_tttll) input
                                                     target
                                                     weight
                                                     (reductionVal @reduction)
                                                     ignoreIndex

nll_loss_forward
  :: Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Int
  -> Int
  -> (Tensor dtype shape, Tensor dtype shape)
nll_loss_forward _self _target _weight _reduction _ignore_index =
  unsafePerformIO $ (ATen.cast5 ATen.nll_loss_forward_tttll) _self
                                                             _target
                                                             _weight
                                                             _reduction
                                                             _ignore_index

nll_loss2d
  :: Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Int
  -> Int
  -> Tensor dtype shape
nll_loss2d _self _target _weight _reduction _ignore_index =
  unsafePerformIO $ (ATen.cast5 ATen.nll_loss2d_tttll) _self
                                                       _target
                                                       _weight
                                                       _reduction
                                                       _ignore_index

nll_loss2d_forward
  :: Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype shape
  -> Int
  -> Int
  -> (Tensor dtype shape, Tensor dtype shape)
nll_loss2d_forward _self _target _weight _reduction _ignore_index =
  unsafePerformIO $ (ATen.cast5 ATen.nll_loss2d_forward_tttll) _self
                                                               _target
                                                               _weight
                                                               _reduction
                                                               _ignore_index

-- |
-- >>> t = smooth_l1_loss @D.ReduceNone (ones :: Tensor Float '[2,2]) (ones :: Tensor Float '[2,2])
-- >>> (dtype t,shape t)
-- (Float,[2,2])
-- >>> t = smooth_l1_loss @D.ReduceSum (ones :: Tensor Float '[2,2]) (ones :: Tensor Float '[2,2])
-- >>> (dtype t,shape t)
-- (Float,[])
smooth_l1_loss
  :: forall reduction dtype shape
   . (KnownReduction reduction)
  => Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype (ConditionalReduction shape reduction)
smooth_l1_loss self target =
  unsafePerformIO $ (ATen.cast3 ATen.smooth_l1_loss_ttl)
    self
    target
    (reductionVal @reduction)

-- |
-- >>> t = soft_margin_loss @D.ReduceNone (ones :: Tensor Float '[2,2]) (ones :: Tensor Float '[2,2])
-- >>> (dtype t,shape t)
-- (Float,[2,2])
-- >>> t = soft_margin_loss @D.ReduceSum (ones :: Tensor Float '[2,2]) (ones :: Tensor Float '[2,2])
-- >>> (dtype t,shape t)
-- (Float,[])
soft_margin_loss
  :: forall reduction dtype shape
   . (KnownReduction reduction)
  => Tensor dtype shape
  -> Tensor dtype shape
  -> Tensor dtype (ConditionalReduction shape reduction)
soft_margin_loss self target =
  unsafePerformIO $ (ATen.cast3 ATen.soft_margin_loss_ttl)
    self
    target
    (reductionVal @reduction)

-- |
-- >>> t = elu (ones :: Tensor Float '[2,2]) 1 1 1
-- >>> (dtype t,shape t)
-- (Float,[2,2])
elu :: Tensor dtype shape -> Float -> Float -> Float -> Tensor dtype shape
elu self alpha scale input_scale =
  unsafePerformIO $ (ATen.cast4 ATen.elu_tsss) self alpha scale input_scale

glu :: Tensor dtype shape -> Int -> Tensor dtype shape
glu _self _dim = unsafePerformIO $ (ATen.cast2 ATen.glu_tl) _self _dim

hardtanh :: Tensor dtype shape -> Float -> Float -> Tensor dtype shape
hardtanh _self _min_val _max_val =
  unsafePerformIO $ (ATen.cast3 ATen.hardtanh_tss) _self _min_val _max_val

leaky_relu :: Tensor dtype shape -> Float -> Tensor dtype shape
leaky_relu _self _negative_slope =
  unsafePerformIO $ (ATen.cast2 ATen.leaky_relu_ts) _self _negative_slope

log_sigmoid :: Tensor dtype shape -> Tensor dtype shape
log_sigmoid _self = unsafePerformIO $ (ATen.cast1 ATen.log_sigmoid_t) _self

log_sigmoid_forward
  :: Tensor dtype shape -> (Tensor dtype shape, Tensor dtype shape)
log_sigmoid_forward _self =
  unsafePerformIO $ (ATen.cast1 ATen.log_sigmoid_forward_t) _self

softplus :: Tensor dtype shape -> Float -> Float -> Tensor dtype shape
softplus _self _beta _threshold =
  unsafePerformIO $ (ATen.cast3 ATen.softplus_tss) _self _beta _threshold

softshrink :: Tensor dtype shape -> Float -> Tensor dtype shape
softshrink _self _lambd =
  unsafePerformIO $ (ATen.cast2 ATen.softshrink_ts) _self _lambd

adaptive_avg_pool2d :: Tensor dtype shape -> (Int, Int) -> Tensor dtype shape
adaptive_avg_pool2d _self _output_size =
  unsafePerformIO $ (ATen.cast2 ATen.adaptive_avg_pool2d_tl) _self _output_size

mkldnn_adaptive_avg_pool2d
  :: Tensor dtype shape -> (Int, Int) -> Tensor dtype shape
mkldnn_adaptive_avg_pool2d _self _output_size =
  unsafePerformIO
    $ (ATen.cast2 ATen.mkldnn_adaptive_avg_pool2d_tl) _self _output_size

adaptive_avg_pool3d
  :: Tensor dtype shape -> (Int, Int, Int) -> Tensor dtype shape
adaptive_avg_pool3d _self _output_size =
  unsafePerformIO $ (ATen.cast2 ATen.adaptive_avg_pool3d_tl) _self _output_size

adaptive_max_pool2d
  :: Tensor dtype shape
  -> (Int, Int)
  -> (Tensor dtype shape, Tensor dtype shape)
adaptive_max_pool2d _self _output_size =
  unsafePerformIO $ (ATen.cast2 ATen.adaptive_max_pool2d_tl) _self _output_size

adaptive_max_pool3d
  :: Tensor dtype shape
  -> (Int, Int, Int)
  -> (Tensor dtype shape, Tensor dtype shape)
adaptive_max_pool3d _self _output_size =
  unsafePerformIO $ (ATen.cast2 ATen.adaptive_max_pool3d_tl) _self _output_size

avg_pool2d
  :: Tensor dtype shape
  -> (Int, Int)
  -> (Int, Int)
  -> (Int, Int)
  -> Bool
  -> Bool
  -> Int
  -> Tensor dtype shape
avg_pool2d _self _kernel_size _stride _padding _ceil_mode _count_include_pad _divisor_override
  = unsafePerformIO $ (ATen.cast7 ATen.avg_pool2d_tlllbbl) _self
                                                           _kernel_size
                                                           _stride
                                                           _padding
                                                           _ceil_mode
                                                           _count_include_pad
                                                           _divisor_override

avg_pool3d
  :: Tensor dtype shape
  -> (Int, Int, Int)
  -> (Int, Int, Int)
  -> (Int, Int, Int)
  -> Bool
  -> Bool
  -> Int
  -> Tensor dtype shape
avg_pool3d _self _kernel_size _stride _padding _ceil_mode _count_include_pad _divisor_override
  = unsafePerformIO $ (ATen.cast7 ATen.avg_pool3d_tlllbbl) _self
                                                           _kernel_size
                                                           _stride
                                                           _padding
                                                           _ceil_mode
                                                           _count_include_pad
                                                           _divisor_override

fractional_max_pool2d
  :: Tensor dtype shape
  -> (Int, Int)
  -> (Int, Int)
  -> Tensor dtype shape
  -> (Tensor dtype shape, Tensor dtype shape)
fractional_max_pool2d _self _kernel_size _output_size _random_samples =
  unsafePerformIO $ (ATen.cast4 ATen.fractional_max_pool2d_tllt)
    _self
    _kernel_size
    _output_size
    _random_samples

fractional_max_pool3d
  :: Tensor dtype shape
  -> (Int, Int, Int)
  -> (Int, Int, Int)
  -> Tensor dtype shape
  -> (Tensor dtype shape, Tensor dtype shape)
fractional_max_pool3d _self _kernel_size _output_size _random_samples =
  unsafePerformIO $ (ATen.cast4 ATen.fractional_max_pool3d_tllt)
    _self
    _kernel_size
    _output_size
    _random_samples

max_pool2d_with_indices
  :: Tensor dtype shape
  -> (Int, Int)
  -> (Int, Int)
  -> (Int, Int)
  -> (Int, Int)
  -> Bool
  -> (Tensor dtype shape, Tensor dtype shape)
max_pool2d_with_indices _self _kernel_size _stride _padding _dilation _ceil_mode
  = unsafePerformIO $ (ATen.cast6 ATen.max_pool2d_with_indices_tllllb)
    _self
    _kernel_size
    _stride
    _padding
    _dilation
    _ceil_mode

max_pool3d_with_indices
  :: Tensor dtype shape
  -> (Int, Int, Int)
  -> (Int, Int, Int)
  -> (Int, Int, Int)
  -> (Int, Int, Int)
  -> Bool
  -> (Tensor dtype shape, Tensor dtype shape)
max_pool3d_with_indices _self _kernel_size _stride _padding _dilation _ceil_mode
  = unsafePerformIO $ (ATen.cast6 ATen.max_pool3d_with_indices_tllllb)
    _self
    _kernel_size
    _stride
    _padding
    _dilation
    _ceil_mode

max_unpool2d
  :: Tensor dtype shape
  -> Tensor dtype shape
  -> (Int, Int)
  -> Tensor dtype shape
max_unpool2d _self _indices _output_size =
  unsafePerformIO
    $ (ATen.cast3 ATen.max_unpool2d_ttl) _self _indices _output_size

max_unpool3d
  :: Tensor dtype shape
  -> Tensor dtype shape
  -> (Int, Int, Int)
  -> (Int, Int, Int)
  -> (Int, Int, Int)
  -> Tensor dtype shape
max_unpool3d _self _indices _output_size _stride _padding =
  unsafePerformIO $ (ATen.cast5 ATen.max_unpool3d_ttlll) _self
                                                         _indices
                                                         _output_size
                                                         _stride
                                                         _padding

reflection_pad1d :: Tensor dtype shape -> (Int, Int) -> Tensor dtype shape
reflection_pad1d _self _padding =
  unsafePerformIO $ (ATen.cast2 ATen.reflection_pad1d_tl) _self _padding

reflection_pad2d
  :: Tensor dtype shape -> (Int, Int, Int, Int) -> Tensor dtype shape
reflection_pad2d _self _padding =
  unsafePerformIO $ (ATen.cast2 ATen.reflection_pad2d_tl) _self _padding

replication_pad1d :: Tensor dtype shape -> (Int, Int) -> Tensor dtype shape
replication_pad1d _self _padding =
  unsafePerformIO $ (ATen.cast2 ATen.replication_pad1d_tl) _self _padding

replication_pad2d
  :: Tensor dtype shape -> (Int, Int, Int, Int) -> Tensor dtype shape
replication_pad2d _self _padding =
  unsafePerformIO $ (ATen.cast2 ATen.replication_pad2d_tl) _self _padding

replication_pad3d
  :: Tensor dtype shape -> (Int, Int, Int, Int, Int, Int) -> Tensor dtype shape
replication_pad3d _self _padding =
  unsafePerformIO $ (ATen.cast2 ATen.replication_pad3d_tl) _self _padding

upsample_linear1d :: Tensor dtype shape -> Int -> Bool -> Tensor dtype shape
upsample_linear1d _self _output_size _align_corners =
  unsafePerformIO $ (ATen.cast3 ATen.upsample_linear1d_tlb) _self
                                                            _output_size
                                                            _align_corners

upsample_bilinear2d
  :: Tensor dtype shape -> (Int, Int) -> Bool -> Tensor dtype shape
upsample_bilinear2d _self _output_size _align_corners =
  unsafePerformIO $ (ATen.cast3 ATen.upsample_bilinear2d_tlb) _self
                                                              _output_size
                                                              _align_corners

upsample_bicubic2d
  :: Tensor dtype shape -> (Int, Int) -> Bool -> Tensor dtype shape
upsample_bicubic2d _self _output_size _align_corners =
  unsafePerformIO $ (ATen.cast3 ATen.upsample_bicubic2d_tlb) _self
                                                             _output_size
                                                             _align_corners

upsample_trilinear3d
  :: Tensor dtype shape -> (Int, Int, Int) -> Bool -> Tensor dtype shape
upsample_trilinear3d _self _output_size _align_corners =
  unsafePerformIO $ (ATen.cast3 ATen.upsample_trilinear3d_tlb) _self
                                                               _output_size
                                                               _align_corners

upsample_nearest1d :: Tensor dtype shape -> Int -> Tensor dtype shape
upsample_nearest1d _self _output_size =
  unsafePerformIO $ (ATen.cast2 ATen.upsample_nearest1d_tl) _self _output_size

upsample_nearest2d :: Tensor dtype shape -> (Int, Int) -> Tensor dtype shape
upsample_nearest2d _self _output_size =
  unsafePerformIO $ (ATen.cast2 ATen.upsample_nearest2d_tl) _self _output_size

upsample_nearest3d
  :: Tensor dtype shape -> (Int, Int, Int) -> Tensor dtype shape
upsample_nearest3d _self _output_size =
  unsafePerformIO $ (ATen.cast2 ATen.upsample_nearest3d_tl) _self _output_size

thnn_conv2d
  :: Tensor dtype shape
  -> Tensor dtype shape
  -> (Int, Int)
  -> Tensor dtype shape
  -> (Int, Int)
  -> (Int, Int)
  -> Tensor dtype shape
thnn_conv2d _self _weight _kernel_size _bias _stride _padding =
  unsafePerformIO $ (ATen.cast6 ATen.thnn_conv2d_ttltll) _self
                                                         _weight
                                                         _kernel_size
                                                         _bias
                                                         _stride
                                                         _padding

thnn_conv2d_forward
  :: Tensor dtype shape
  -> Tensor dtype shape
  -> (Int, Int)
  -> Tensor dtype shape
  -> (Int, Int)
  -> (Int, Int)
  -> ( Tensor dtype shape
     , Tensor dtype shape
     , Tensor dtype shape
     )
thnn_conv2d_forward _self _weight _kernel_size _bias _stride _padding =
  unsafePerformIO $ (ATen.cast6 ATen.thnn_conv2d_forward_ttltll) _self
                                                                 _weight
                                                                 _kernel_size
                                                                 _bias
                                                                 _stride
                                                                 _padding

thnn_conv_depthwise2d
  :: Tensor dtype shape
  -> Tensor dtype shape
  -> (Int, Int)
  -> Tensor dtype shape
  -> (Int, Int)
  -> (Int, Int)
  -> (Int, Int)
  -> Tensor dtype shape
thnn_conv_depthwise2d _self _weight _kernel_size _bias _stride _padding _dilation
  = unsafePerformIO $ (ATen.cast7 ATen.thnn_conv_depthwise2d_ttltlll)
    _self
    _weight
    _kernel_size
    _bias
    _stride
    _padding
    _dilation

thnn_conv_depthwise2d_forward
  :: Tensor dtype shape
  -> Tensor dtype shape
  -> (Int, Int)
  -> Tensor dtype shape
  -> (Int, Int)
  -> (Int, Int)
  -> (Int, Int)
  -> Tensor dtype shape
thnn_conv_depthwise2d_forward _self _weight _kernel_size _bias _stride _padding _dilation
  = unsafePerformIO $ (ATen.cast7 ATen.thnn_conv_depthwise2d_forward_ttltlll)
    _self
    _weight
    _kernel_size
    _bias
    _stride
    _padding
    _dilation

thnn_conv3d
  :: Tensor dtype shape
  -> Tensor dtype shape
  -> (Int, Int, Int)
  -> Tensor dtype shape
  -> (Int, Int, Int)
  -> (Int, Int, Int)
  -> Tensor dtype shape
thnn_conv3d _self _weight _kernel_size _bias _stride _padding =
  unsafePerformIO $ (ATen.cast6 ATen.thnn_conv3d_ttltll) _self
                                                         _weight
                                                         _kernel_size
                                                         _bias
                                                         _stride
                                                         _padding

thnn_conv3d_forward
  :: Tensor dtype shape
  -> Tensor dtype shape
  -> (Int, Int, Int)
  -> Tensor dtype shape
  -> (Int, Int, Int)
  -> (Int, Int, Int)
  -> ( Tensor dtype shape
     , Tensor dtype shape
     , Tensor dtype shape
     )
thnn_conv3d_forward _self _weight _kernel_size _bias _stride _padding =
  unsafePerformIO $ (ATen.cast6 ATen.thnn_conv3d_forward_ttltll) _self
                                                                 _weight
                                                                 _kernel_size
                                                                 _bias
                                                                 _stride
                                                                 _padding

conv_dilated2d
  :: Tensor dtype shape
  -> Tensor dtype shape
  -> (Int, Int)
  -> Tensor dtype shape
  -> (Int, Int)
  -> (Int, Int)
  -> (Int, Int)
  -> Tensor dtype shape
conv_dilated2d _self _weight _kernel_size _bias _stride _padding _dilation =
  unsafePerformIO $ (ATen.cast7 ATen.conv_dilated2d_ttltlll) _self
                                                             _weight
                                                             _kernel_size
                                                             _bias
                                                             _stride
                                                             _padding
                                                             _dilation

conv_dilated3d
  :: Tensor dtype shape
  -> Tensor dtype shape
  -> (Int, Int, Int)
  -> Tensor dtype shape
  -> (Int, Int, Int)
  -> (Int, Int, Int)
  -> (Int, Int, Int)
  -> Tensor dtype shape
conv_dilated3d _self _weight _kernel_size _bias _stride _padding _dilation =
  unsafePerformIO $ (ATen.cast7 ATen.conv_dilated3d_ttltlll) _self
                                                             _weight
                                                             _kernel_size
                                                             _bias
                                                             _stride
                                                             _padding
                                                             _dilation

col2im
  :: Tensor dtype shape
  -> (Int, Int)
  -> (Int, Int)
  -> (Int, Int)
  -> (Int, Int)
  -> (Int, Int)
  -> Tensor dtype shape
col2im _self _output_size _kernel_size _dilation _padding _stride =
  unsafePerformIO $ (ATen.cast6 ATen.col2im_tlllll) _self
                                                    _output_size
                                                    _kernel_size
                                                    _dilation
                                                    _padding
                                                    _stride

im2col
  :: Tensor dtype shape
  -> (Int, Int)
  -> (Int, Int)
  -> (Int, Int)
  -> (Int, Int)
  -> Tensor dtype shape
im2col _self _kernel_size _dilation _padding _stride =
  unsafePerformIO $ (ATen.cast5 ATen.im2col_tllll) _self
                                                   _kernel_size
                                                   _dilation
                                                   _padding
                                                   _stride

