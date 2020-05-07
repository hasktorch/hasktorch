{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
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

module Torch.Typed.Functional where

import           Prelude                 hiding ( all
                                                , any
                                                , sin
                                                , sinh
                                                , cos
                                                , cosh
                                                , tan
                                                , tanh
                                                , asin
                                                , asinh
                                                , acos
                                                , acosh
                                                , atan
                                                , atanh
                                                , abs
                                                , max
                                                , min
                                                , exp
                                                , log
                                                , round
                                                , isNaN
                                                , floor
                                                , ceil
                                                )
import           Data.Finite
import qualified Data.Int                      as I
import           Torch.HList
import           Data.Kind                      ( Constraint
                                                , Type
                                                )
import           Data.Maybe
import           Data.Proxy
import           Data.Reflection
import           Control.Arrow                  ( (&&&) )
import           GHC.Generics                   ( Generic )
import           GHC.Natural                    ( Natural )
import           GHC.TypeLits
import           GHC.TypeLits.Extra
import           System.IO.Unsafe
import           Data.Singletons.Prelude.List   ( Product )

import           Foreign.ForeignPtr
import qualified Torch.Internal.Const                    as ATen
import qualified Torch.Internal.Type                     as ATen
import qualified Torch.Internal.Cast                     as ATen
import qualified Torch.Internal.Class                    as ATen
import qualified Torch.Internal.Managed.Cast             as ATen.Managed
import qualified Torch.Internal.Managed.Native           as ATen.Managed
import qualified Torch.Internal.Managed.Type.Tensor      as ATen.Managed
import qualified Torch.Internal.Managed.Type.Scalar      as ATen.Managed
import qualified Torch.Internal.Managed.Type.Tuple       as ATen.Managed

import qualified Torch.Tensor                  as D
import qualified Torch.TensorFactories         as D
import qualified Torch.TensorOptions           as D
import qualified Torch.DType                   as D
import qualified Torch.Device                  as D
import qualified Torch.Scalar                  as D
import           Torch.Functional               ( Reduction(..)
                                                , Tri(..)
                                                , isUpper
                                                , kOne
                                                )
import           Torch.Typed.Aux
import           Torch.Typed.Factories
import           Torch.Typed.Tensor


type family SumDType (dtype :: D.DType) :: D.DType where
  SumDType D.Bool = D.Int64
  SumDType D.UInt8 = D.Int64
  SumDType D.Int8 = D.Int64
  SumDType D.Int16 = D.Int64
  SumDType D.Int32 = D.Int64
  SumDType D.Int64 = D.Int64
  SumDType D.Half = D.Half
  SumDType D.Float = D.Float
  SumDType D.Double = D.Double

type family SumDTypeIsValid (device :: (D.DeviceType, Nat)) (dtype :: D.DType) :: Constraint where
  SumDTypeIsValid '( 'D.CPU, 0)    dtype = DTypeIsNotHalf '( 'D.CPU, 0) dtype
  SumDTypeIsValid '( 'D.CUDA, _)   dtype = ()
  SumDTypeIsValid '(deviceType, _) dtype = UnsupportedDTypeForDevice deviceType dtype

-- | sumAll
--
-- >>> dtype &&& shape &&& (\t' -> D.asValue (toDynamic t') :: Int) $ sumAll (ones :: CPUTensor 'D.Bool '[2, 3])
-- (Int64,([],6))
-- >>> dtype &&& shape &&& (\t' -> D.asValue (toDynamic t') :: Int) $ sumAll (ones :: CPUTensor 'D.UInt8 '[2, 3])
-- (Int64,([],6))
-- >>> dtype &&& shape &&& (\t' -> D.asValue (toDynamic t') :: Int) $ sumAll (ones :: CPUTensor 'D.Int8 '[2, 3])
-- (Int64,([],6))
-- >>> dtype &&& shape &&& (\t' -> D.asValue (toDynamic t') :: Int) $ sumAll (ones :: CPUTensor 'D.Int16 '[2, 3])
-- (Int64,([],6))
-- >>> dtype &&& shape &&& (\t' -> D.asValue (toDynamic t') :: Int) $ sumAll (ones :: CPUTensor 'D.Int32 '[2, 3])
-- (Int64,([],6))
-- >>> dtype &&& shape &&& (\t' -> D.asValue (toDynamic t') :: Int) $ sumAll (ones :: CPUTensor 'D.Int64 '[2, 3])
-- (Int64,([],6))
-- >>> dtype &&& shape &&& (\t' -> D.asValue (toDynamic t') :: Float) $ sumAll (ones :: CPUTensor 'D.Float '[2, 3])
-- (Float,([],6.0))
-- >>> dtype &&& shape &&& (\t' -> D.asValue (toDynamic t') :: Double) $ sumAll (ones :: CPUTensor 'D.Double '[2, 3])
-- (Double,([],6.0))
sumAll
  :: forall shape dtype' dtype device
   . ( SumDTypeIsValid device dtype
     , dtype' ~ SumDType dtype
     )
  => Tensor device dtype  shape -- ^ input
  -> Tensor device dtype' '[] -- ^ output
sumAll input = unsafePerformIO $ ATen.cast1 ATen.Managed.sum_t input

-- | sumDim
--
-- >>> dtype &&& shape $ sumDim @0 (ones :: CPUTensor 'D.Float '[3,4,5])
-- (Float,[4,5])
-- >>> sumDim @1 (ones :: CPUTensor 'D.Float '[2,4])
-- Tensor Float [2] [ 4.0000   ,  4.0000   ]
sumDim
  :: forall d shape shape' dtype dtype' device
   . ( KnownNat d
     , shape' ~ DropValue shape d
     , SumDTypeIsValid device dtype
     , dtype' ~ SumDType dtype
     )
  => Tensor device dtype  shape -- ^ input
  -> Tensor device dtype' shape' -- ^ output
sumDim input = unsafePerformIO $ ATen.cast2 ATen.Managed.sum_tl input (natValI @d)

-- | abs
--
-- >>> dtype &&& shape $ abs (ones :: CPUTensor 'D.Float '[2,2])
-- (Float,[2,2])
abs
  :: forall shape dtype device
   . (DTypeIsNotHalf device dtype, DTypeIsNotBool device dtype)
  => Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
abs input = unsafePerformIO $ ATen.cast1 ATen.Managed.abs_t input

-- | ceil
--
-- >>> dtype &&& shape $ ceil (ones :: CPUTensor 'D.Float '[2,2])
-- (Float,[2,2])
ceil
  :: forall shape dtype device
   . (StandardFloatingPointDTypeValidation device dtype)
  => Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
ceil input = unsafePerformIO $ ATen.cast1 ATen.Managed.ceil_t input

-- | floor
--
-- >>> dtype &&& shape $ floor (ones :: CPUTensor 'D.Float '[2,2])
-- (Float,[2,2])
floor
  :: forall shape dtype device
   . (StandardFloatingPointDTypeValidation device dtype)
  => Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
floor input = unsafePerformIO $ ATen.cast1 ATen.Managed.floor_t input

-- TODO: better error messages, "Couldn't match type ‘'False’ with ‘'True’" isn't great
type family AllDimsPositive (shape :: [Nat]) :: Constraint where
  AllDimsPositive '[] = ()
  AllDimsPositive (x ': xs) = (1 <= x, AllDimsPositive xs)

type family AggregationDTypeIsValid (device :: (D.DeviceType, Nat)) (dtype :: D.DType) :: Constraint where
  AggregationDTypeIsValid '( 'D.CPU, 0)    dtype = DTypeIsNotHalf '( 'D.CPU, 0) dtype
  AggregationDTypeIsValid '( 'D.CUDA, _)   dtype = ()
  AggregationDTypeIsValid '(deviceType, _) dtype = UnsupportedDTypeForDevice deviceType dtype

-- | min
--
-- >>> dtype &&& shape $ min (ones :: CPUTensor 'D.Float '[2,2])
-- (Float,[])
min
  :: forall shape dtype device
   . ( AggregationDTypeIsValid device dtype
     , AllDimsPositive shape
     )
  => Tensor device dtype shape -- ^ input
  -> Tensor device dtype '[] -- ^ output
min input = unsafePerformIO $ ATen.cast1 ATen.Managed.min_t input

-- | max
--
-- >>> dtype &&& shape $ max (ones :: CPUTensor 'D.Float '[2,2])
-- (Float,[])
max
  :: forall shape dtype device
  . ( AggregationDTypeIsValid device dtype
    , AllDimsPositive shape
    )
  => Tensor device dtype shape -- ^ input
  -> Tensor device dtype '[] -- ^ output
max input = unsafePerformIO $ ATen.cast1 ATen.Managed.max_t input

-- | median
--
-- >>> dtype &&& shape $ median (ones :: CPUTensor 'D.Float '[2,2])
-- (Float,[])
median
  :: forall shape dtype device
  . ( AggregationDTypeIsValid device dtype
    , AllDimsPositive shape
    )
  => Tensor device dtype shape -- ^ input
  -> Tensor device dtype '[] -- ^ output
median input = unsafePerformIO $ ATen.cast1 ATen.Managed.median_t input

-- | mean
--
-- >>> dtype &&& shape $ mean (ones :: CPUTensor 'D.Float '[2,2])
-- (Float,[])
mean
  :: forall shape dtype device
  . ( AggregationDTypeIsValid device dtype
    , AllDimsPositive shape
    )
  => Tensor device dtype shape -- ^ input
  -> Tensor device dtype '[] -- ^ output
mean input = unsafePerformIO $ ATen.cast1 ATen.Managed.mean_t input

-- | addScalar
-- TODO: what dtypes is this defined for?
-- TODO: what scalar types is this defined for?
--
-- >>> dtype &&& shape $ addScalar 1 (ones :: CPUTensor 'D.Float '[2,2])
-- (Float,[2,2])
addScalar
  :: forall a shape dtype device
   . D.Scalar a
  => a -- ^ scalar input
  -> Tensor device dtype shape -- ^ tensor input
  -> Tensor device dtype shape -- ^ output
addScalar a input = unsafePerformIO $ ATen.cast2 ATen.Managed.add_ts input a

-- | subScalar
-- TODO: what dtypes is this defined for?
-- TODO: what scalar types is this defined for?
--
-- >>> dtype &&& shape $ subScalar 1 (ones :: CPUTensor 'D.Float '[2,2])
-- (Float,[2,2])
subScalar
  :: forall a shape dtype device
   . D.Scalar a
  => a -- ^ scalar input
  -> Tensor device dtype shape -- ^ tensor input
  -> Tensor device dtype shape -- ^ output
subScalar a input = unsafePerformIO $ ATen.cast2 ATen.Managed.sub_ts input a

-- | mulScalar
-- TODO: what dtypes is this defined for?
-- TODO: what scalar types is this defined for?
--
-- >>> dtype &&& shape $ mulScalar 2 (ones :: CPUTensor 'D.Float '[2,2])
-- (Float,[2,2])
mulScalar
  :: forall a shape dtype device
   . D.Scalar a
  => a -- ^ scalar input
  -> Tensor device dtype shape -- ^ tensor input
  -> Tensor device dtype shape -- ^ output
mulScalar a input = unsafePerformIO $ ATen.cast2 ATen.Managed.mul_ts input a

-- | divScalar
-- TODO: what dtypes is this defined for?
-- TODO: what scalar types is this defined for?
--
-- >>> dtype &&& shape $ divScalar 2 (ones :: CPUTensor 'D.Float '[2,2])
-- (Float,[2,2])
divScalar
  :: forall a shape dtype device
   . D.Scalar a
  => a -- ^ scalar input
  -> Tensor device dtype shape -- ^ tensor input
  -> Tensor device dtype shape -- ^ output
divScalar a input = unsafePerformIO $ ATen.cast2 ATen.Managed.div_ts input a

-- | erf
--
-- >>> dtype &&& shape $ erf (ones :: CPUTensor 'D.Float '[3,2])
-- (Float,[3,2])
erf
  :: forall shape dtype device
   . (StandardFloatingPointDTypeValidation device dtype)
  => Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
erf input = unsafePerformIO $ ATen.cast1 ATen.Managed.erf_t input

-- | exp
--
-- >>> dtype &&& shape $ exp (ones :: CPUTensor 'D.Float '[3,2])
-- (Float,[3,2])
exp
  :: forall shape dtype device
   . (StandardFloatingPointDTypeValidation device dtype)
  => Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
exp input = unsafePerformIO $ ATen.cast1 ATen.Managed.exp_t input

-- | log1p
--
-- >>> dtype &&& shape $ log1p (ones :: CPUTensor 'D.Float '[3,2])
-- (Float,[3,2])
log1p
  :: forall shape dtype device
   . (StandardFloatingPointDTypeValidation device dtype)
  => Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
log1p input = unsafePerformIO $ ATen.cast1 ATen.Managed.log1p_t input

-- | log2
-- >>> dtype &&& shape $ log2 (ones :: CPUTensor 'D.Float '[3,2])
-- (Float,[3,2])
log2
  :: forall shape dtype device
   . (StandardFloatingPointDTypeValidation device dtype)
  => Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
log2 input = unsafePerformIO $ ATen.cast1 ATen.Managed.log2_t input

-- | log10
--
-- >>> dtype &&& shape $ log10 (ones :: CPUTensor 'D.Float '[3,2])
-- (Float,[3,2])
log10
  :: forall shape dtype device
   . (StandardFloatingPointDTypeValidation device dtype)
  => Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
log10 input = unsafePerformIO $ ATen.cast1 ATen.Managed.log10_t input

-- | pow
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
--
-- >>> dtype &&& shape $ pow 2 (ones :: CPUTensor 'D.Float '[3,2])
-- (Float,[3,2])
pow
  :: forall a shape dtype device
   . D.Scalar a
  => a -- ^ power
  -> Tensor device dtype shape -- ^ input tensor
  -> Tensor device dtype shape -- ^ output tensor
pow a input = unsafePerformIO $ ATen.cast2 ATen.Managed.pow_ts input a

-- | relu activation function
--
-- >>> dtype &&& shape $ relu (ones :: CPUTensor 'D.Float '[3,2])
-- (Float,[3,2])
relu
  :: forall shape dtype device
   . (StandardFloatingPointDTypeValidation device dtype)
  => Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
relu input = unsafePerformIO $ ATen.cast1 ATen.Managed.relu_t input

-- | selu
--
-- >>> dtype &&& shape $ selu (ones :: CPUTensor 'D.Float '[3,2])
-- (Float,[3,2])
selu
  :: forall shape dtype device
   . (StandardFloatingPointDTypeValidation device dtype)
  => Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
selu input = unsafePerformIO $ ATen.cast1 ATen.Managed.selu_t input

-- | mish
-- `mish` is a smooth activation function, see https://arxiv.org/abs/1908.08681 for details.
--
-- >>> dtype &&& shape &&& (\t -> D.asValue (toDynamic t) :: [[Float]]) $ mish (ones :: CPUTensor 'D.Float '[3,2])
-- (Float,([3,2],[[0.86509836,0.86509836],[0.86509836,0.86509836],[0.86509836,0.86509836]]))
mish
  :: forall shape dtype device
   . ( StandardFloatingPointDTypeValidation device dtype
     , BasicArithmeticDTypeIsValid device dtype
     , shape ~ Broadcast shape shape
     )
  => Tensor device dtype shape
  -> Tensor device dtype shape
mish = mul =<< tanh . softplus (1 :: Float) 20

-- | sigmoid
--
-- >>> dtype &&& shape $ sigmoid (ones :: CPUTensor 'D.Float '[3,2])
-- (Float,[3,2])
sigmoid
  :: forall shape dtype device
   . (StandardFloatingPointDTypeValidation device dtype)
  => Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
sigmoid input = unsafePerformIO $ ATen.cast1 ATen.Managed.sigmoid_t input

-- | sin
--
-- >>> dtype &&& shape $ sin (ones :: CPUTensor 'D.Float '[3,2])
-- (Float,[3,2])
sin
  :: forall shape dtype device
   . (StandardFloatingPointDTypeValidation device dtype)
  => Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
sin input = unsafePerformIO $ ATen.cast1 ATen.Managed.sin_t input

-- | sinh
--
-- >>> dtype &&& shape $ sinh (ones :: CPUTensor 'D.Float '[3,2])
-- (Float,[3,2])
sinh
  :: forall shape dtype device
   . (StandardFloatingPointDTypeValidation device dtype)
  => Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
sinh input = unsafePerformIO $ ATen.cast1 ATen.Managed.sinh_t input

-- | cos
--
-- >>> dtype &&& shape $ cos (ones :: CPUTensor 'D.Float '[3,2])
-- (Float,[3,2])
cos
  :: forall shape dtype device
   . (StandardFloatingPointDTypeValidation device dtype)
  => Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
cos input = unsafePerformIO $ ATen.cast1 ATen.Managed.cos_t input

-- | sqrt
sqrt
  :: forall shape dtype device
   . (StandardFloatingPointDTypeValidation device dtype)
  => Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
sqrt input = unsafePerformIO $ ATen.cast1 ATen.Managed.sqrt_t input

-- | tanh
tanh
  :: forall shape dtype device
   . (StandardFloatingPointDTypeValidation device dtype)
  => Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
tanh input = unsafePerformIO $ ATen.cast1 ATen.Managed.tanh_t input

type family SqueezeAll (shape :: [Nat]) :: [Nat] where
  SqueezeAll '[] = '[]
  SqueezeAll (1: xs) = SqueezeAll xs
  SqueezeAll (x: xs) = x ': SqueezeAll xs

-- | squeezeAll
-- | Note: this function is unsafe; dimensions not known statically are retained in the type,
-- | but may be squeezed out if they turn out 1 at run-time.
--
-- >>> dtype &&& shape $ squeezeAll (ones :: CPUTensor 'D.Float '[2,1,2,1,2])
-- (Float,[2,2,2])
squeezeAll
  :: forall shape shape' dtype device
   . (shape' ~ SqueezeAll shape)
  => Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape' -- ^ output
squeezeAll input = unsafePerformIO $ ATen.cast1 ATen.Managed.squeeze_t input

-- | ConditionalReduction
--
-- >>> :kind! ConditionalReduction '[3,2] ReduceNone
-- ConditionalReduction '[3,2] ReduceNone :: [Nat]
-- = '[3, 2]
-- >>> :kind! ConditionalReduction '[3,2] ReduceMean
-- ConditionalReduction '[3,2] ReduceMean :: [Nat]
-- = '[]
type family ConditionalReduction (shape :: [Nat]) (reduction :: Reduction) :: [Nat] where
    ConditionalReduction shape ReduceNone = shape
    ConditionalReduction shape _ = '[]

class KnownReduction reduction where
    reductionVal :: Int

instance KnownReduction ReduceNone where
    reductionVal = 0
instance KnownReduction ReduceMean where
    reductionVal = 1
instance KnownReduction ReduceSum where
    reductionVal = 2

-- | binary cross entropy
--
-- >>> t = ones :: CPUTensor 'D.Float '[2,2]
-- >>> dtype &&& shape $ binaryCrossEntropy @ReduceNone t t t
-- (Float,[2,2])
-- >>> dtype &&& shape $ binaryCrossEntropy @ReduceMean t t t 
-- (Float,[])
-- >>> dtype &&& shape $ binaryCrossEntropy @ReduceSum t t t
-- (Float,[])
binaryCrossEntropy
  :: forall (reduction :: Reduction) shape shape' dtype device
   . ( KnownReduction reduction
     , shape' ~ ConditionalReduction shape reduction
     , StandardFloatingPointDTypeValidation device dtype
     )
  => Tensor device dtype shape -- ^ weight
  -> Tensor device dtype shape -- ^ prediction
  -> Tensor device dtype shape -- ^ target
  -> Tensor device dtype shape' -- ^ output
binaryCrossEntropy weight prediction target = unsafePerformIO $ ATen.cast4
  ATen.Managed.binary_cross_entropy_tttl
  prediction
  target
  weight
  (reductionVal @reduction)

-- | mseLoss
--
-- >>> t = ones :: CPUTensor 'D.Float '[2,2]
-- >>> dtype &&& shape $ mseLoss @ReduceNone t t
-- (Float,[2,2])
-- >>> dtype &&& shape $ mseLoss @ReduceMean t t
-- (Float,[])
-- >>> dtype &&& shape $ mseLoss @ReduceSum t t
-- (Float,[])
mseLoss
  :: forall (reduction :: Reduction) shape shape' dtype device
   . ( KnownReduction reduction
     , shape' ~ ConditionalReduction shape reduction
     , StandardFloatingPointDTypeValidation device dtype
     )
  => Tensor device dtype shape -- ^ prediction
  -> Tensor device dtype shape -- ^ target
  -> Tensor device dtype shape' -- ^ output
mseLoss prediction target = unsafePerformIO $ ATen.cast3
  ATen.Managed.mse_loss_ttl
  prediction
  target
  (reductionVal @reduction)

-- | softmax
--
-- >>> t = ones :: CPUTensor 'D.Float '[2,2]
-- >>> dtype &&& shape $ softmax @0 t
-- (Float,[2,2])
-- >>> dtype &&& shape $ softmax @1 t
-- (Float,[2,2])
softmax
  :: forall dim shape dtype device
   . ( KnownNat dim, DimOutOfBoundCheck shape dim
     , KnownDType dtype
     , StandardFloatingPointDTypeValidation device dtype
     )
  => Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
softmax input = unsafePerformIO
  $ ATen.cast3 ATen.Managed.softmax_tls input (natValI @dim) (dtypeVal @dtype)

-- | logSoftmax
--
-- >>> t = ones :: CPUTensor 'D.Float '[2,2]
-- >>> dtype &&& shape $ logSoftmax @0 t
-- (Float,[2,2])
-- >>> dtype &&& shape $ logSoftmax @1 t
-- (Float,[2,2])
logSoftmax
  :: forall dim shape dtype device
   . ( KnownNat dim, DimOutOfBoundCheck shape dim
     , KnownDType dtype
     , StandardFloatingPointDTypeValidation device dtype
     )
  => Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
logSoftmax input = unsafePerformIO
  $ ATen.cast3 ATen.Managed.log_softmax_tls input (natValI @dim) (dtypeVal @dtype)

type family Square (shape :: [Nat]) :: [Nat] where
  Square (n:n:'[])   = '[n,n]
  Square (b:n:n:'[]) = '[b,n,n]
  Square _           = TypeError (Text "This shape must be square matrix or batch + square matrix.")

type family VectorOfSquare (shape :: [Nat]) :: [Nat] where
  VectorOfSquare (n:n:'[])   = '[n]
  VectorOfSquare (b:n:n:'[]) = '[b,n]
  VectorOfSquare _           = TypeError (Text "This shape must be square matrix or batch + square matrix.")

type family FstSquareDim (shape :: [Nat]) :: Nat where
  FstSquareDim (n:m:'[])   = n
  FstSquareDim (b:n:m:'[]) = n
  FstSquareDim _           = TypeError (Text "Can not get first dimention of matrix or batch + matrix.")

type family InverseShapeIsValid (device :: (D.DeviceType, Nat)) (shape :: [Nat]) :: Constraint where
  InverseShapeIsValid '( 'D.CPU, 0)  _     = ()
  InverseShapeIsValid '( 'D.CUDA, _) shape = AllDimsPositive shape

type family InverseDTypeIsValid (device :: (D.DeviceType, Nat)) (dtype :: D.DType) :: Constraint where
  InverseDTypeIsValid '( 'D.CPU, 0)            dtype = ( DTypeIsFloatingPoint '( 'D.CPU, 0) dtype
                                                       , DTypeIsNotHalf '( 'D.CPU, 0) dtype
                                                       )
  InverseDTypeIsValid '( 'D.CUDA, deviceIndex) dtype = ( DTypeIsFloatingPoint '( 'D.CUDA, deviceIndex) dtype
                                                       , DTypeIsNotHalf '( 'D.CUDA, deviceIndex) dtype
                                                       )
  InverseDTypeIsValid '(deviceType, _)         dtype = UnsupportedDTypeForDevice deviceType dtype

-- | inverse
-- TODO: if rank < n for any tensors in the batch, then this will not work. we can't decide this statically, but we should prevent runtime errors. therefore, return Maybe?
--
-- >>> t <- randn :: IO (CPUTensor 'D.Float '[3,2,2])
-- >>> dtype &&& shape $ inverse t
-- (Float,[3,2,2])
-- >>> t <- randn :: IO (CPUTensor 'D.Float '[2,2])
-- >>> dtype &&& shape $ inverse t
-- (Float,[2,2])
inverse
  :: forall shape shape' dtype device
   . ( shape' ~ Square shape
     , InverseShapeIsValid device shape
     , InverseDTypeIsValid device dtype
     )
  => Tensor device dtype shape -- ^ inverse
  -> Tensor device dtype shape' -- ^ output
inverse input = unsafePerformIO $ ATen.cast1 ATen.Managed.inverse_t input

type family SymeigDTypeIsValid (device :: (D.DeviceType, Nat)) (dtype :: D.DType) :: Constraint where
  SymeigDTypeIsValid '( 'D.CPU, 0)            dtype = ( DTypeIsFloatingPoint '( 'D.CPU, 0) dtype
                                                      , DTypeIsNotHalf '( 'D.CPU, 0) dtype
                                                      )
  SymeigDTypeIsValid '( 'D.CUDA, deviceIndex) dtype = ( DTypeIsFloatingPoint '( 'D.CUDA, deviceIndex) dtype
                                                      , DTypeIsNotHalf '( 'D.CUDA, deviceIndex) dtype
                                                      )
  SymeigDTypeIsValid '(deviceType, _)         dtype = UnsupportedDTypeForDevice deviceType dtype

-- | symeig
--
-- >>> t <- rand :: IO (CPUTensor 'D.Float '[3,2,2])
-- >>> (eigenVals,eigenVecs) = symeig Upper t
-- >>> dtype &&& shape $ eigenVals
-- (Float,[3,2])
-- >>> :t eigenVals
-- eigenVals :: Tensor '( 'D.CPU, 0) 'D.Float '[3, 2]
-- >>> dtype &&& shape $ eigenVecs
-- (Float,[3,2,2])
-- >>> :t eigenVecs
-- eigenVecs :: Tensor '( 'D.CPU, 0) 'D.Float '[3, 2, 2]
-- >>> (eigenVals,eigenVecs) = symeig Lower t
-- >>> dtype &&& shape $ eigenVals
-- (Float,[3,2])
-- >>> dtype &&& shape $ eigenVecs
-- (Float,[3,2,2])
symeig
  :: forall shape shape' shape'' dtype device
   . ( shape' ~ VectorOfSquare shape
     , shape'' ~ Square shape
     , SymeigDTypeIsValid device dtype
     )
  => Tri -- ^ upper or lower triagonal
  -> Tensor device dtype shape -- ^ input
  -> ( Tensor device dtype shape'
     , Tensor device dtype shape''
     ) -- ^ eigenvalues and eigenvectors
symeig upper input = unsafePerformIO
  $ ATen.cast3 ATen.Managed.symeig_tbb input True boolUpper
  where boolUpper = isUpper upper

-- | symeigvalues
--
-- >>> t <- rand :: IO (CPUTensor 'D.Float '[3,2,2])
-- >>> eigenVals = symeigvalues Upper t
-- >>> dtype &&& shape $ eigenVals
-- (Float,[3,2])
-- >>> :t eigenVals
-- eigenVals :: Tensor '( 'D.CPU, 0) 'D.Float '[3, 2]
symeigvalues
  :: forall shape shape' dtype device
   . ( shape' ~ VectorOfSquare shape
     , SymeigDTypeIsValid device dtype
     )
  => Tri -- ^ upper or lower triagonal
  -> Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape'
symeigvalues upper input = fst symeig'
  where
    boolUpper = isUpper upper
    symeig' :: ( Tensor device dtype shape', Tensor device dtype shape'')
    symeig' = unsafePerformIO $ ATen.cast3 ATen.Managed.symeig_tbb input False boolUpper

data EigenVectors = EnableEigenVectors | DisableEigenVectors

class KnownEigenVectors a where
  enableEigenVectors :: Bool

instance KnownEigenVectors EnableEigenVectors where
  enableEigenVectors = True
instance KnownEigenVectors DisableEigenVectors where
  enableEigenVectors = False

type family ConditionalEigenVectors (eigenvectors :: EigenVectors) (n:: Nat) :: [Nat] where
  ConditionalEigenVectors EnableEigenVectors  n = '[n, n]
  ConditionalEigenVectors DisableEigenVectors _ = '[0]

type family EigDTypeIsValid (device :: (D.DeviceType, Nat)) (dtype :: D.DType) :: Constraint where
  EigDTypeIsValid '( 'D.CPU, 0)            dtype = ( DTypeIsFloatingPoint '( 'D.CPU, 0) dtype
                                                   , DTypeIsNotHalf '( 'D.CPU, 0) dtype
                                                   )
  EigDTypeIsValid '( 'D.CUDA, deviceIndex) dtype = ( DTypeIsFloatingPoint '( 'D.CUDA, deviceIndex) dtype
                                                   , DTypeIsNotHalf '( 'D.CUDA, deviceIndex) dtype
                                                   )
  EigDTypeIsValid '(deviceType, _)         dtype = UnsupportedDTypeForDevice deviceType dtype

-- | eig
-- >>> t <- rand :: IO (CPUTensor 'D.Float '[3,3])
-- >>> (eigenVals,eigenVecs) = eig @EnableEigenVectors t
-- >>> dtype &&& shape $ eigenVals
-- (Float,[3,2])
-- >>> :t eigenVals
-- eigenVals :: Tensor '( 'D.CPU, 0) 'D.Float '[3, 2]
-- >>> dtype &&& shape $ eigenVecs
-- (Float,[3,3])
-- >>> :t eigenVecs
-- eigenVecs :: Tensor '( 'D.CPU, 0) 'D.Float '[3, 3]
-- >>> (eigenVals,eigenVecs) = eig @DisableEigenVectors t
-- >>> dtype &&& shape $ eigenVals
-- (Float,[3,2])
-- >>> dtype &&& shape $ eigenVecs
-- (Float,[0])
-- >>> :t eigenVecs
-- eigenVecs :: Tensor '( 'D.CPU, 0) 'D.Float '[0]
eig
  :: forall eigenvectors n shape dtype device
   . ( KnownNat n
     , KnownEigenVectors eigenvectors
     , shape ~ ConditionalEigenVectors eigenvectors n
     , EigDTypeIsValid device dtype
     )
  => Tensor device dtype '[n, n] -- ^ input matrix
  -> ( Tensor device dtype '[n, 2]
     , Tensor device dtype shape
     ) -- ^ eigenvalues and eigenvectors
eig input =
  unsafePerformIO $ ATen.cast2 ATen.Managed.eig_tb input (enableEigenVectors @eigenvectors)

type family SVDShapes (shape :: [Nat]) (reduced :: ReducedSVD) :: ([Nat], [Nat], [Nat]) where
  SVDShapes '[0, n]    'ThinSVD = '( '[0, 0],          '[0],          '[n, n])
  SVDShapes '[m, n]    'ThinSVD = '( '[m, Min m n],    '[Min m n],    '[n, Min m n])
  SVDShapes '[m, n]    'FullSVD = '( '[m, m],          '[Min m n],    '[n, n])
  SVDShapes '[b, 0, n] 'ThinSVD = '( '[b, 0, 0],       '[b, 0],       '[b, n, n])
  SVDShapes '[b, m, n] 'ThinSVD = '( '[b, m, Min m n], '[b, Min m n], '[b, n, Min m n])
  SVDShapes '[b, m, n] 'FullSVD = '( '[b, m, m],       '[b, Min m n], '[b, n, n])
  SVDShapes _          _        = TypeError (Text "A singular value decomposition can only be computed for 2D matrices for at most one batch dimension.")

data ReducedSVD = ThinSVD | FullSVD

class KnownReducedSVD (reduced :: ReducedSVD) where
  reducedSVD :: Bool

instance KnownReducedSVD ThinSVD where
  reducedSVD = True
instance KnownReducedSVD FullSVD where
  reducedSVD = False

type family SVDDTypeIsValid (device :: (D.DeviceType, Nat)) (dtype :: D.DType) :: Constraint where
  SVDDTypeIsValid '( 'D.CPU, 0)            dtype = ( DTypeIsFloatingPoint '( 'D.CPU, 0) dtype
                                                   , DTypeIsNotHalf '( 'D.CPU, 0) dtype
                                                   )
  SVDDTypeIsValid '( 'D.CUDA, deviceIndex) dtype = ( DTypeIsFloatingPoint '( 'D.CUDA, deviceIndex) dtype
                                                   , DTypeIsNotHalf '( 'D.CUDA, deviceIndex) dtype
                                                   )
  SVDDTypeIsValid '(deviceType, _)         dtype = UnsupportedDTypeForDevice deviceType dtype


-- | Singular Value Decomposition
-- TODO: When `compute_uv` is `False`, backward cannot be performed since `u` and `v` from the forward pass are required for the backward operation. There is no way to encode in the types at this point in time. Thus, only `True` is supported currently.
--
-- This function returns a tuple `(u, s, v)`
-- which is the singular value decomposition of a input real matrix
-- or batches of real matrices input such that
-- `input = U×diag(S)×V^T`.
--
-- >>> a <- randn :: IO (CPUTensor 'D.Float '[3, 5])
-- >>> (u, s, v) = svd @'ThinSVD a
-- >>> dtype &&& shape $ u
-- (Float,[3,3])
-- >>> dtype &&& shape $ s
-- (Float,[3])
-- >>> dtype &&& shape $ v
-- (Float,[5,3])
-- >>> (u, s, v) = svd @'FullSVD a
-- >>> dtype &&& shape $ u
-- (Float,[3,3])
-- >>> dtype &&& shape $ s
-- (Float,[3])
-- >>> dtype &&& shape $ v
-- (Float,[5,5])
-- >>> a <- randn :: IO (CPUTensor 'D.Float '[5, 3])
-- >>> (u, s, v) = svd @'ThinSVD a
-- >>> dtype &&& shape $ u
-- (Float,[5,3])
-- >>> dtype &&& shape $ s
-- (Float,[3])
-- >>> dtype &&& shape $ v
-- (Float,[3,3])
-- >>> (u, s, v) = svd @'FullSVD a
-- >>> dtype &&& shape $ u
-- (Float,[5,5])
-- >>> dtype &&& shape $ s
-- (Float,[3])
-- >>> dtype &&& shape $ v
-- (Float,[3,3])
svd
  :: forall reduced shape shapeU shapeS shapeV dtype device
   . ( KnownReducedSVD reduced
     , '(shapeU, shapeS, shapeV) ~ SVDShapes shape reduced
     , SVDDTypeIsValid device dtype
     )
  => Tensor device dtype shape -- ^ (batched) input real matrix
  -> ( Tensor device dtype shapeU
     , Tensor device dtype shapeS
     , Tensor device dtype shapeV
     ) -- ^ (batched) output tuple of `u`, `s`, and `v`
svd input =
  unsafePerformIO $ ATen.cast3 ATen.Managed.svd_tbb input (reducedSVD @reduced) True

type family CholeskyDTypeIsValid (device :: (D.DeviceType, Nat)) (dtype :: D.DType) :: Constraint where
  CholeskyDTypeIsValid '( 'D.CPU, 0)            dtype = ( DTypeIsFloatingPoint '( 'D.CPU, 0) dtype
                                                        , DTypeIsNotHalf '( 'D.CPU, 0) dtype
                                                        )
  CholeskyDTypeIsValid '( 'D.CUDA, deviceIndex) dtype = ( DTypeIsFloatingPoint '( 'D.CUDA, deviceIndex) dtype
                                                        , DTypeIsNotHalf '( 'D.CUDA, deviceIndex) dtype
                                                        )
  CholeskyDTypeIsValid '(deviceType, _)         dtype = UnsupportedDTypeForDevice deviceType dtype

-- | cholesky
-- TODO: cholesky can throw if the input is not positive-definite.
-- Computes the Cholesky decomposition of a symmetric positive-definite matrix.
-- The operation supports batching.
--
-- >>> t <- rand :: IO (CPUTensor 'D.Float '[2,2])
-- >>> u = cholesky Upper (t `matmul` transpose2D t)
-- >>> dtype &&& shape $ u
-- (Float,[2,2])
-- >>> :t u
-- u :: Tensor '( 'D.CPU, 0) 'D.Float '[2, 2]
cholesky
  :: forall shape shape' dtype device
   . ( shape' ~ Square shape
     , CholeskyDTypeIsValid device dtype
     )
  => Tri -- ^ indicate whether to return an upper or lower triangular matrix.
  -> Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape' -- ^ output
cholesky upper input = unsafePerformIO $ ATen.cast2 ATen.Managed.cholesky_tb input boolUpper
  where boolUpper = isUpper upper

-- | choleskyInverse
-- Computes the inverse of a symmetric positive-definite matrix
-- using its Cholesky factor, returned, e.g., by `cholesky`.
-- Unlike `cholesky`, this operation does not support batching.
-- The inverse is computed using the LAPACK routine `?potri`.
-- 
-- >>> t <- rand :: IO (CPUTensor 'D.Float '[2,2])
-- >>> tri = Upper
-- >>> u = cholesky tri (t `matmul` transpose2D t)
-- >>> dtype &&& shape $ choleskyInverse tri u
-- (Float,[2,2])
choleskyInverse
  :: forall n dtype device
   . ( 1 <= n
     , CholeskyDTypeIsValid device dtype
     )
  => Tri -- ^ decides whether the upper or the lower triangular part of the input tensor is used
  -> Tensor device dtype '[n, n] -- ^ the input 2-D tensor `u`, an upper or lower triangular Cholesky factor
  -> Tensor device dtype '[n, n] -- ^ the output 2-D tensor
choleskyInverse upper input = unsafePerformIO
  $ ATen.cast2 ATen.Managed.cholesky_inverse_tb input boolUpper
  where boolUpper = isUpper upper

-- | choleskySolve
-- Solves the system of linear equations represented by `a c = b`
-- using the Cholesky factor matrix `u` of `a` (returned, e.g., by `cholesky`),
-- where `a` is a positive semidefinite matrix.
-- The operation supports batching.
--
-- >>> t <- rand :: IO (CPUTensor 'D.Float '[3,3])
-- >>> a = t `matmul` transpose2D t
-- >>> b <- rand :: IO (CPUTensor 'D.Float '[3,2])
-- >>> tri = Upper
-- >>> u = cholesky tri a
-- >>> dtype &&& shape $ choleskySolve tri b u
-- (Float,[3,2])
choleskySolve
  :: forall m_k m_m dtype device
   . ( Square m_m ~ m_m
     , FstSquareDim m_m ~ FstSquareDim m_k
     , 1 <= FstSquareDim m_m
     , CholeskyDTypeIsValid device dtype
     )
  => Tri -- ^ decides whether the upper or the lower triangular part of the input tensor `u` is used
  -> Tensor device dtype m_k -- ^ the (batched) RHS tensor `b`
  -> Tensor device dtype m_m -- ^ the (batched) input 2-D tensor `u`, an upper or lower triangular Cholesky factor
  -> Tensor device dtype m_k -- ^ the (batched) output 2-D tensor
choleskySolve upper b u = unsafePerformIO
  $ ATen.cast3 ATen.Managed.cholesky_solve_ttb b u boolUpper
  where boolUpper = isUpper upper

type family SolveDTypeIsValid (device :: (D.DeviceType, Nat)) (dtype :: D.DType) :: Constraint where
  SolveDTypeIsValid '( 'D.CPU, 0)            dtype = ( DTypeIsFloatingPoint '( 'D.CPU, 0) dtype
                                                     , DTypeIsNotHalf '( 'D.CPU, 0) dtype
                                                     )
  SolveDTypeIsValid '( 'D.CUDA, deviceIndex) dtype = ( DTypeIsFloatingPoint '( 'D.CUDA, deviceIndex) dtype
                                                     , DTypeIsNotHalf '( 'D.CUDA, deviceIndex) dtype
                                                     )
  SolveDTypeIsValid '(deviceType, _)         dtype = UnsupportedDTypeForDevice deviceType dtype

-- | solve
-- Solves the system of linear equations represented by `a c = b` and also returns the LU decomposition of `a`.
-- `a` has to be a positive semidefinite matrix.
-- The operation supports batching.
--
-- >>> t <- rand :: IO (CPUTensor 'D.Float '[10,10])
-- >>> a = t `matmul` transpose2D t
-- >>> b <- rand :: IO (CPUTensor 'D.Float '[10,3])
-- >>> (c,lu) = solve b a
-- >>> dtype &&& shape $ c
-- (Float,[10,3])
-- >>> dtype &&& shape $ lu
-- (Float,[10,10])
-- >>> :t c
-- c :: Tensor '( 'D.CPU, 0) 'D.Float '[10, 3]
-- >>> :t lu
-- lu :: Tensor '( 'D.CPU, 0) 'D.Float '[10, 10]
solve
  :: forall m_k m_m dtype device
   . ( Square m_m ~ m_m
     , FstSquareDim m_m ~ FstSquareDim m_k
     , 1 <= FstSquareDim m_m
     , SolveDTypeIsValid device dtype
     )
  => Tensor device dtype m_k -- ^ the (batched) RHS tensor `b`
  -> Tensor device dtype m_m -- ^ the (batched) positive semidefinite matrix `a`
  -> ( Tensor device dtype m_k
     , Tensor device dtype m_m
     ) -- ^ the (batched) outputs c and lu
solve b a = unsafePerformIO $ ATen.cast2 ATen.Managed.solve_tt b a

-- | geqrf
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
-- `geqrf` computes a QR decomposition of the given `input` matrix,
-- but without constructing `Q` and `R` as explicit separate matrices.
-- Rather, this function directly calls the underlying LAPACK function `?geqrf`
-- which produces a tuple `(a, tau)` of intermediate results as defined in
-- the LAPACK documentation for `?geqrf`.
--
-- You can use `orgqr` on `(a, tau)` to compute the real orthogonal matrix `Q`,
-- but in general you may just want to use `qr` instead.
--
-- See the LAPACK documentation for `?geqrf` for further details,
-- https://software.intel.com/en-us/node/521004.
-- 
-- >>> (a, tau) = geqrf (ones :: CPUTensor 'D.Float '[3,4])
-- >>> dtype &&& shape $ a
-- (Float,[3,4])
-- >>> dtype &&& shape $ tau
-- (Float,[3])
-- >>> (a, tau) = geqrf (ones :: CPUTensor 'D.Float '[4,3])
-- >>> dtype &&& shape $ a
-- (Float,[4,3])
-- >>> dtype &&& shape $ tau
-- (Float,[3])
geqrf
  :: forall m n dtype device
   . Tensor device dtype '[m, n] -- ^ input matrix
  -> ( Tensor device dtype '[m, n]
     , Tensor device dtype '[Min m n]
     ) -- ^ tuple `(a, tau)` of output matrices
geqrf input = unsafePerformIO $ ATen.cast1 ATen.Managed.geqrf_t input

-- | orgqr
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
-- Computes the orthogonal matrix `Q` of a QR factorization
-- from the `(a, tau)` tuple returned by `geqrf`.
-- 
-- This directly calls the underlying LAPACK function `?orgqr`.
-- See the LAPACK documentation for `?orgqr` for further details,
-- https://software.intel.com/en-us/mkl-developer-reference-c-orgqr.
-- 
-- >>> dtype &&& shape $ orgqr (ones :: CPUTensor 'D.Float '[3,4]) (ones :: CPUTensor 'D.Float '[3])
-- (Float,[3,4])
orgqr
  :: forall m n dtype device
   . Tensor device dtype '[m, n]
  -> Tensor device dtype '[Min m n]
  -> Tensor device dtype '[m, n]
orgqr a tau = unsafePerformIO $ ATen.cast2 ATen.Managed.orgqr_tt a tau

-- | sign
-- works for all dtypes
--
-- >>> dtype &&& shape $ sign (ones :: CPUTensor 'D.Float '[3,2])
-- (Float,[3,2])
sign
  :: forall shape dtype device
   . Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
sign input = unsafePerformIO $ ATen.cast1 ATen.Managed.sign_t input

type family SetValue (shape :: [Nat]) (i :: Nat) (j :: Nat)  :: [Nat] where
  SetValue '[]     _ _ = '[]
  SetValue (x: xs) 0 j = j: xs
  SetValue (x: xs) i j = x: SetValue xs (i-1) j

type family GetValue (shape :: [Nat]) (i :: Nat) :: Nat where
  GetValue '[]     _ = TypeError (Text "Can not find a element in the list.")
  GetValue (x: xs) 0 = x
  GetValue (x: xs) i = GetValue xs (i-1)

-- | Transpose
--
-- >>> :kind! Transpose '[3,2] 0 1
-- Transpose '[3,2] 0 1 :: [Nat]
-- = '[2, 3]
-- >>> :kind! Transpose '[3,2,1] 1 2
-- Transpose '[3,2,1] 1 2 :: [Nat]
-- = '[3, 1, 2]
type family Transpose (shape :: [Nat]) (dim0 :: Nat) (dim1 :: Nat) :: [Nat] where
  Transpose s d0 d1 = (SetValue (SetValue s d0 (GetValue s d1)) d1 (GetValue s d0))

-- | transpose
-- See "../../../../deps/pytorch/aten/src/ATen/native/TensorShape.cpp".
--
-- >>> dtype &&& shape $ transpose @0 @1 (ones :: CPUTensor 'D.Float '[3,2])
-- (Float,[2,3])
-- >>> dtype &&& shape $ transpose @0 @1 (ones :: CPUTensor 'D.Float '[3,2,1])
-- (Float,[2,3,1])
-- >>> dtype &&& shape $ transpose @1 @2 (ones :: CPUTensor 'D.Float '[3,2,1])
-- (Float,[3,1,2])
transpose
  :: forall n m shape shape' dtype device
   . ( KnownNat n, KnownNat m
     , shape' ~ Transpose shape n m
     )
  => Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape' -- ^ output
transpose input =
  unsafePerformIO $ ATen.cast3 ATen.Managed.transpose_tll input (natValI @n) (natValI @m)

-- | transpose2d, special case for a 2D tensor
--
-- >>> dtype &&& shape $ transpose2D (ones :: CPUTensor 'D.Float '[3,2])
-- (Float,[2,3])
transpose2D
  :: forall (i :: Nat) (j :: Nat) dtype device
   . Tensor device dtype '[i, j] -- ^ input
  -> Tensor device dtype '[j, i] -- ^ output
transpose2D = transpose @0 @1

-- diag :: Tensor device dtype shape -> Int -> Tensor device dtype shape
-- diag t index = unsafePerformIO $ (ATen.cast2 ATen.Managed.tensor_diag_l) t index

-- | all
-- See https://pytorch.org/docs/stable/tensors.html#torch.BoolTensor.all.
--
-- >>> t = all (fromJust [False, False] :: CPUTensor 'D.Bool '[2])
-- >>> toInt t == 1
-- False
--
-- >>> t = all (fromJust [False, True] :: CPUTensor 'D.Bool '[2])
-- >>> toInt t == 1
-- False
--
-- >>> t = all (fromJust [True, True] :: CPUTensor 'D.Bool '[2])
-- >>> toInt t == 1
-- True
all
  :: forall shape device
   . Tensor device 'D.Bool shape -- ^ input
  -> Tensor device 'D.Bool '[] -- ^ output
all input = unsafePerformIO $ ATen.cast1 ATen.Managed.all_t input

-- | any
-- See https://pytorch.org/docs/stable/tensors.html#torch.BoolTensor.any.
--
-- >>> t = any (fromJust [False, False] :: CPUTensor 'D.Bool '[2])
-- >>> toInt t == 1
-- False
--
-- >>> t = any (fromJust [False, True] :: CPUTensor 'D.Bool '[2])
-- >>> toInt t == 1
-- True
--
-- >>> t = any (fromJust [True, True] :: CPUTensor 'D.Bool '[2])
-- >>> toInt t == 1
-- True
any
  :: forall shape device
   . Tensor device 'D.Bool shape -- ^ input
  -> Tensor device 'D.Bool '[] -- ^ output
any input = unsafePerformIO $ ATen.cast1 ATen.Managed.any_t input

data KeepOrDropDim = KeepDim | DropDim

class KnownKeepOrDropDim keepOrDropDim where
  keepOrDropDimVal :: Bool

instance KnownKeepOrDropDim KeepDim where
  keepOrDropDimVal = True
instance KnownKeepOrDropDim DropDim where
  keepOrDropDimVal = False

type family ConditionalDropDimension (shape :: [Nat]) (dim :: Nat) (keepOrDropDim :: KeepOrDropDim) :: [Nat] where
  ConditionalDropDimension '[]      _ _             = TypeError (Text "The specified dimension is not available.")
  ConditionalDropDimension (x : xs) 0 KeepDim       = 1 ': xs
  ConditionalDropDimension (x : xs) 0 DropDim       = xs
  ConditionalDropDimension (x : xs) i keepOrDropDim = x ': ConditionalDropDimension xs (i - 1) keepOrDropDim

-- | allDim
-- See https://pytorch.org/docs/stable/tensors.html#torch.BoolTensor.all.
--
-- >>> t = fromJust [[True, True], [True, False], [True, True], [True, True]] :: CPUTensor 'D.Bool '[4, 2]
--
-- >>> dtype &&& shape &&& (\t' -> D.asValue (toDynamic t') :: [Bool]) $ allDim @1 @DropDim t
-- (Bool,([4],[True,False,True,True]))
--
-- >>> dtype &&& shape &&& (\t' -> D.asValue (toDynamic t') :: [[Bool]]) $ allDim @1 @KeepDim t
-- (Bool,([4,1],[[True],[False],[True],[True]]))
--
-- >>> dtype &&& shape &&& (\t' -> D.asValue (toDynamic t') :: [Bool]) $ allDim @0 @DropDim t
-- (Bool,([2],[True,False]))
--
-- >>> dtype &&& shape &&& (\t' -> D.asValue (toDynamic t') :: [[Bool]]) $ allDim @0 @KeepDim t
-- (Bool,([1,2],[[True,False]]))
allDim
  :: forall dim keepOrDropDim shape' shape device
   . ( KnownNat dim
     , KnownKeepOrDropDim keepOrDropDim
     , shape' ~ ConditionalDropDimension shape dim keepOrDropDim
     )
  => Tensor device 'D.Bool shape -- ^ input
  -> Tensor device 'D.Bool shape' -- ^ output
allDim input = unsafePerformIO
  $ ATen.cast3 ATen.Managed.all_tlb input (natValI @dim) (keepOrDropDimVal @keepOrDropDim)

-- | anyDim
-- See https://pytorch.org/docs/stable/tensors.html#torch.BoolTensor.any.
--
-- >>> t = fromJust [[True, True], [True, False], [True, True], [True, True]] :: CPUTensor 'D.Bool '[4, 2]
--
-- >>> dtype &&& shape &&& (\t' -> D.asValue (toDynamic t') :: [Bool]) $ anyDim @1 @DropDim t
-- (Bool,([4],[True,True,True,True]))
--
-- >>> dtype &&& shape &&& (\t' -> D.asValue (toDynamic t') :: [[Bool]]) $ anyDim @1 @KeepDim t
-- (Bool,([4,1],[[True],[True],[True],[True]]))
--
-- >>> dtype &&& shape &&& (\t' -> D.asValue (toDynamic t') :: [Bool]) $ anyDim @0 @DropDim t
-- (Bool,([2],[True,True]))
--
-- >>> dtype &&& shape &&& (\t' -> D.asValue (toDynamic t') :: [[Bool]]) $ anyDim @0 @KeepDim t
-- (Bool,([1,2],[[True,True]]))
anyDim
  :: forall dim keepOrDropDim shape' shape device
   . ( KnownNat dim
     , KnownKeepOrDropDim keepOrDropDim
     , shape' ~ ConditionalDropDimension shape dim keepOrDropDim
     )
  => Tensor device 'D.Bool shape -- ^ input
  -> Tensor device 'D.Bool shape' -- ^ output
anyDim input = unsafePerformIO $ ATen.cast3 ATen.Managed.any_tlb input (natValI @dim) (keepOrDropDimVal @keepOrDropDim)

-- | dropout
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
-- TODO: get rid of IO by exposing the RNG state
-- TODO: can we use D.Scalar for the dropout probability?
--
-- >>> t = ones :: CPUTensor 'D.Float '[3,2]
-- >>> t' <- dropout 0.5 False t
-- >>> dtype &&& shape $ t'
-- (Float,[3,2])
-- >>> t'' <- dropout 0.5 False t
-- >>> t ==. t''
-- Tensor Bool [3,2] [[ 1,  1],
--                    [ 1,  1],
--                    [ 1,  1]]
-- >>> t''' <- dropout 0.0 True t
-- >>> t ==. t'''
-- Tensor Bool [3,2] [[ 1,  1],
--                    [ 1,  1],
--                    [ 1,  1]]
-- >>> t'''' <- dropout 1.0 True t
-- >>> t''''
-- Tensor Float [3,2] [[ 0.0000,  0.0000],
--                     [ 0.0000,  0.0000],
--                     [ 0.0000,  0.0000]]
dropout
  :: forall shape dtype device
   . Double -- ^ dropout probability
  -> Bool -- ^ whether or not to activate dropout
  -> Tensor device dtype shape -- ^ input
  -> IO (Tensor device dtype shape) -- ^ output
dropout p train input = ATen.cast3 ATen.Managed.dropout_tdb input p train

-- | featureDropout
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
-- TODO: why not IO?
-- TODO: can we use D.Scalar for the dropout probability?
--
-- >>> c = featureDropout 0.1 True (ones :: CPUTensor 'D.Float '[2,2])
-- >>> dtype &&& shape $ c
-- (Float,[2,2])
featureDropout
  :: forall shape dtype device
   . Double -- ^ dropout probability
  -> Bool -- ^ whether or not to activate dropout
  -> Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
featureDropout p train input =
  unsafePerformIO $ ATen.cast3 ATen.Managed.feature_dropout_tdb input p train

-- | alphaDropout
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
-- TODO: why not IO?
-- TODO: can we use D.Scalar for the dropout probability?
--
-- >>> c = alphaDropout 0.1 True (ones :: CPUTensor 'D.Float '[2,2])
-- >>> dtype &&& shape $ c
-- (Float,[2,2])
alphaDropout
  :: forall shape dtype device
   . Double -- ^ dropout probability
  -> Bool -- ^ whether or not to activate dropout
  -> Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
alphaDropout p train input =
  unsafePerformIO $ ATen.cast3 ATen.Managed.alpha_dropout_tdb input p train

-- | featureAlphaDropout
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
-- TODO: why not IO?
-- TODO: can we use D.Scalar for the dropout probability?
--
-- >>> c = featureAlphaDropout 0.1 True (ones :: CPUTensor 'D.Float '[2,2])
-- >>> dtype &&& shape $ c
-- (Float,[2,2])
featureAlphaDropout
  :: forall shape dtype device
   . Double -- ^ dropout probability
  -> Bool -- ^ whether or not to activate dropout
  -> Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
featureAlphaDropout p train input =
  unsafePerformIO $ ATen.cast3 ATen.Managed.feature_alpha_dropout_tdb input p train

-- | acos
--
-- >>> dtype &&& shape $ acos (ones :: CPUTensor 'D.Float '[3,2])
-- (Float,[3,2])
acos
  :: forall shape dtype device
   . (StandardFloatingPointDTypeValidation device dtype)
  => Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
acos input = unsafePerformIO $ ATen.cast1 ATen.Managed.acos_t input

-- | avgPool1d
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
--
-- >>> t = avgPool1d @1 @1 @0 (ones :: CPUTensor 'D.Float '[1,3,4])
-- >>> shape t
-- [1,3,4]
-- >>> :t t
-- t :: Tensor '( 'D.CPU, 0) 'D.Float '[1, 3, 4]
avgPool1d
  :: forall kernelSize
            stride
            padding
            channelSize
            inputSize
            batchSize
            outputSize
            dtype
            device
   . ( All KnownNat '[kernelSize, stride, padding, channelSize, inputSize, batchSize]
     , ConvSideCheck inputSize kernelSize stride padding outputSize
     )
  => Tensor device dtype '[batchSize, channelSize, inputSize] -- ^ input
  -> Tensor device dtype '[batchSize, channelSize, outputSize] -- ^ output
avgPool1d input = unsafePerformIO $ ATen.cast6 ATen.Managed.avg_pool1d_tlllbb
  input
  (natValI @kernelSize)
  (natValI @stride)
  (natValI @padding)
  False
  True

-- | adaptiveAvgPool1d
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
--
-- >>> t = adaptiveAvgPool1d @8 (ones :: CPUTensor 'D.Float '[1,3,16])
-- >>> shape t
-- [1,3,8]
-- >>> :t t
-- t :: Tensor '( 'D.CPU, 0) 'D.Float '[1, 3, 8]
adaptiveAvgPool1d
  :: forall outputSize channelSize inputSize batchSize dtype device
   . (All KnownNat '[channelSize, inputSize, batchSize, outputSize])
  => Tensor device dtype '[batchSize, channelSize, inputSize] -- ^ input
  -> Tensor device dtype '[batchSize, channelSize, outputSize] -- ^ output
adaptiveAvgPool1d input = unsafePerformIO
  $ ATen.cast2 ATen.Managed.adaptive_avg_pool1d_tl input (natValI @outputSize)

-- | adaptiveMaxPool1d
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
--
-- >>> tt = adaptiveMaxPool1d @8 (ones :: CPUTensor 'D.Float '[1,3,16])
-- >>> shape . fst $ tt
-- [1,3,8]
-- >>> :t tt
-- tt 
--   :: (Tensor '( 'D.CPU, 0) 'D.Float '[1, 3, 8],
--       Tensor '( 'D.CPU, 0) 'D.Int64 '[1, 3, 8])
adaptiveMaxPool1d
  :: forall outputSize channelSize inputSize batchSize dtype device
   . (All KnownNat '[channelSize, inputSize, batchSize, outputSize])
  => Tensor device dtype '[batchSize, channelSize, inputSize] -- ^ input
  -> ( Tensor device dtype    '[batchSize, channelSize, outputSize]
     , Tensor device 'D.Int64 '[batchSize, channelSize, outputSize]
     ) -- ^ output
adaptiveMaxPool1d input = unsafePerformIO
  $ ATen.cast2 ATen.Managed.adaptive_max_pool1d_tl input (natValI @outputSize)

-- | addmv
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
-- TODO: can we use D.Scalar for beta and alpha?
--
-- >>> t = addmv 1 1 (ones :: CPUTensor 'D.Float '[3,2]) (zeros :: CPUTensor 'D.Float '[2]) (ones :: CPUTensor 'D.Float '[])
-- >>> dtype &&& shape $ t
-- (Float,[3])
-- >>> :t t
-- t :: Tensor '( 'D.CPU, 0) 'D.Float '[3]
addmv
  :: forall shape' shape n m dtype device
   . ( KnownNat n
     , KnownNat m
     , shape' ~ Broadcast shape '[n]
     )
  => Float -- ^ beta
  -> Float -- ^ alpha
  -> Tensor device dtype '[n,m] -- ^ matrix
  -> Tensor device dtype '[m] -- ^ vector
  -> Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape' -- ^ output
addmv beta alpha mat vec input = unsafePerformIO $ ATen.cast5 ATen.Managed.addmv_tttss input mat vec beta alpha

-- | addr
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
-- TODO: can we use D.Scalar for beta and alpha?
--
-- >>> t = addr 1 1 (ones :: CPUTensor 'D.Float '[3]) (zeros :: CPUTensor 'D.Float '[2]) (ones :: CPUTensor 'D.Float '[])
-- >>> dtype &&& shape $ t
-- (Float,[3,2])
-- >>> :t t
-- t :: Tensor '( 'D.CPU, 0) 'D.Float '[3, 2]
addr
  :: forall shape' shape n m dtype device
   . ( KnownNat n
     , KnownNat m
     , shape' ~ Broadcast shape '[n,m]
     )
  => Float -- ^ beta
  -> Float -- ^ alpha
  -> Tensor device dtype '[n] -- ^ first input vector
  -> Tensor device dtype '[m] -- ^ second input vector
  -> Tensor device dtype shape -- ^ input tensor
  -> Tensor device dtype shape' -- ^ output tensor
addr beta alpha vec1 vec2 input = unsafePerformIO $ ATen.cast5 ATen.Managed.addr_tttss input vec1 vec2 beta alpha

-- affine_grid_generator :: Tensor device dtype shape -> [Int] -> Tensor device dtype shape
-- affine_grid_generator _theta _size = unsafePerformIO $ (ATen.cast2 ATen.Managed.affine_grid_generator_tl) _theta _size

-- | allclose
--
-- >>> allclose 0.1 0.1 True (ones :: CPUTensor 'D.Float '[3,3]) (ones :: CPUTensor 'D.Float '[3,3])
-- True
allclose
  :: forall shape dtype device
   . Double -- ^ relative tolerance
  -> Double -- ^ absolute tolerance
  -> Bool -- ^ whether or not NaN equals NaN
  -> Tensor device dtype shape -- ^ input tensor
  -> Tensor device dtype shape -- ^ other input tensor
  -> Bool -- ^ output
allclose rtol atol equalNaN input other =
  unsafePerformIO $ ATen.cast5 ATen.Managed.allclose_ttddb input other rtol atol equalNaN

-- | argmax
-- See https://pytorch.org/docs/stable/torch.html#torch.argmax.
--
-- >>> t = fromJust [[0, 1], [-1, 2], [0, 1], [0, -2]] :: CPUTensor 'D.Float '[4, 2]
--
-- >>> dtype &&& shape &&& (\t' -> D.asValue (toDynamic t') :: [Int]) $ argmax @1 @DropDim t
-- (Int64,([4],[1,1,1,0]))
--
-- >>> dtype &&& shape &&& (\t' -> D.asValue (toDynamic t') :: [[Int]]) $ argmax @1 @KeepDim t
-- (Int64,([4,1],[[1],[1],[1],[0]]))
--
-- >>> dtype &&& shape &&& (\t' -> D.asValue (toDynamic t') :: [Int]) $ argmax @0 @DropDim t
-- (Int64,([2],[3,1]))
--
-- >>> dtype &&& shape &&& (\t' -> D.asValue (toDynamic t') :: [[Int]]) $ argmax @0 @KeepDim t
-- (Int64,([1,2],[[3,1]]))
argmax
  :: forall dim keepOrDropDim shape' shape dtype device
   . ( KnownNat dim
     , KnownKeepOrDropDim keepOrDropDim
     , shape' ~ ConditionalDropDimension shape dim keepOrDropDim
     )
  => Tensor device dtype    shape -- ^ input
  -> Tensor device 'D.Int64 shape' -- ^ output
argmax input = unsafePerformIO $ ATen.cast3 ATen.Managed.argmax_tlb
                                       input
                                       (natValI @dim)
                                       (keepOrDropDimVal @keepOrDropDim)

-- | argmin
-- See https://pytorch.org/docs/stable/torch.html#torch.argmin.
--
-- >>> t = fromJust [[0, 1], [-1, 2], [0, 1], [0, -2]] :: CPUTensor 'D.Float '[4, 2]
--
-- >>> dtype &&& shape &&& (\t' -> D.asValue (toDynamic t') :: [Int]) $ argmin @1 @DropDim t
-- (Int64,([4],[0,0,0,1]))
--
-- >>> dtype &&& shape &&& (\t' -> D.asValue (toDynamic t') :: [[Int]]) $ argmin @1 @KeepDim t
-- (Int64,([4,1],[[0],[0],[0],[1]]))
--
-- >>> dtype &&& shape &&& (\t' -> D.asValue (toDynamic t') :: [Int]) $ argmin @0 @DropDim t
-- (Int64,([2],[1,3]))
--
-- >>> dtype &&& shape &&& (\t' -> D.asValue (toDynamic t') :: [[Int]]) $ argmin @0 @KeepDim t
-- (Int64,([1,2],[[1,3]]))
argmin
  :: forall dim keepOrDropDim shape' shape dtype device
  . ( KnownNat dim
    , KnownKeepOrDropDim keepOrDropDim
    , shape' ~ ConditionalDropDimension shape dim keepOrDropDim
    )
  => Tensor device dtype    shape -- ^ input
  -> Tensor device 'D.Int64 shape' -- ^ output
argmin input = unsafePerformIO $ ATen.cast3 ATen.Managed.argmin_tlb
                                       input
                                       (natValI @dim)
                                       (keepOrDropDimVal @keepOrDropDim)

-- as_strided :: Tensor device dtype shape -> [Int] -> [Int] -> Int -> Tensor device dtype shape
-- as_strided _input _size _stride _storage_offset = unsafePerformIO $ (ATen.cast4 ATen.Managed.as_strided_tlll) _input _size _stride _storage_offset

-- | asin
--
-- >>> dtype &&& shape $ asin (ones :: CPUTensor 'D.Float '[3,2])
-- (Float,[3,2])
asin
  :: forall shape dtype device
   . (StandardFloatingPointDTypeValidation device dtype)
  => Tensor device dtype shape
  -> Tensor device dtype shape
asin input = unsafePerformIO $ ATen.cast1 ATen.Managed.asin_t input

-- | atan
--
-- >>> dtype &&& shape $ atan (ones :: CPUTensor 'D.Float '[3,2])
-- (Float,[3,2])
atan
  :: forall shape dtype device
   . (StandardFloatingPointDTypeValidation device dtype)
  => Tensor device dtype shape
  -> Tensor device dtype shape
atan input = unsafePerformIO $ ATen.cast1 ATen.Managed.atan_t input

-- | baddbmm
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
--
-- >>> t = baddbmm 1 1 (ones :: CPUTensor 'D.Float '[5,3,2]) (zeros :: CPUTensor 'D.Float '[5,2,4]) (ones :: CPUTensor 'D.Float '[])
-- >>> dtype &&& shape $ t
-- (Float,[5,3,4])
-- >>> :t t
-- t :: Tensor '( 'D.CPU, 0) 'D.Float '[5, 3, 4]
baddbmm
  :: forall shape' shape  batchSize n m k dtype device
   . ( KnownNat n
     , KnownNat m
     , KnownNat k
     , shape' ~ Broadcast shape '[batchSize, n, m]
     )
  => Float -- ^ beta
  -> Float -- ^ alpha
  -> Tensor device dtype '[batchSize, n, k] -- ^ first batch
  -> Tensor device dtype '[batchSize, k, m] -- ^ second batch
  -> Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape' -- ^ output
baddbmm beta alpha batch1 batch2 input = unsafePerformIO $ ATen.cast5 ATen.Managed.baddbmm_tttss input batch1 batch2 beta alpha

-- batch_norm :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Bool -> Double -> Double -> Bool -> Tensor device dtype shape
-- batch_norm _input _weight _bias _running_mean _running_var _training _momentum _eps _cudnn_enabled = unsafePerformIO $ (ATen.cast9 ATen.Managed.batch_norm_tttttbddb) _input _weight _bias _running_mean _running_var _training _momentum _eps _cudnn_enabled

-- bilinear :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape
-- bilinear _input1 _input2 _weight _bias = unsafePerformIO $ (ATen.cast4 ATen.Managed.bilinear_tttt) _input1 _input2 _weight _bias

-- binary_cross_entropy_with_logits :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Int -> Tensor device dtype shape
-- binary_cross_entropy_with_logits _input _target _weight _pos_weight _reduction = unsafePerformIO $ (ATen.cast5 ATen.Managed.binary_cross_entropy_with_logits_ttttl) _input _target _weight _pos_weight _reduction

-- bincount :: Tensor device dtype shape -> Tensor device dtype shape -> Int -> Tensor device dtype shape
-- bincount _input _weights _minlength = unsafePerformIO $ (ATen.cast3 ATen.Managed.bincount_ttl) _input _weights _minlength

-- | bitwise_not
--
-- >>> dtype &&& shape $ bitwiseNot (ones :: CPUTensor 'D.Bool [3,3])
-- (Bool,[3,3])
bitwiseNot
  :: forall shape device
   . Tensor device 'D.Bool shape
  -> Tensor device 'D.Bool shape
bitwiseNot input = unsafePerformIO $ ATen.cast1 ATen.Managed.bitwise_not_t input

-- | batched matrix multiplication
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
--
-- >>> dtype &&& shape $ bmm (ones :: CPUTensor 'D.Float '[5,3,2]) (zeros :: CPUTensor 'D.Float '[5,2,4])
-- (Float,[5,3,4])
bmm
  :: forall batchSize n m k dtype device
   . Tensor device dtype '[batchSize, n, k] -- ^ input
  -> Tensor device dtype '[batchSize, k, m] -- ^ other input
  -> Tensor device dtype '[batchSize, n, m] -- ^ output
bmm input other = unsafePerformIO $ ATen.cast2 ATen.Managed.bmm_tt input other

-- | BroadcastTensorsImpl
--
-- >>> type Ty = BroadcastTensorsImpl '[] 'Nothing
-- >>> :kind! Ty
-- Ty :: Maybe ([Nat], D.DType, (D.DeviceType, Nat))
-- = 'Nothing
-- >>> type Ty = BroadcastTensorsImpl '[Tensor '( 'D.CPU, 0) 'D.Float '[1, 3], Tensor '( 'D.CPU, 0) 'D.Float '[2, 1]] 'Nothing
-- >>> :kind! Ty
-- Ty :: Maybe ([Nat], D.DType, (D.DeviceType, Nat))
-- = 'Just '( '[2, 3], 'D.Float, '( 'D.CPU, 0))
-- >>> type Ty = BroadcastTensorsImpl '[Tensor '( 'D.CPU, 0) 'D.Float '[1, 3], Tensor '( 'D.CPU, 0) 'D.Float '[2, 1], Tensor '( 'D.CPU, 0) 'D.Float '[5, 1, 1]] 'Nothing
-- >>> :kind! Ty
-- Ty :: Maybe ([Nat], D.DType, (D.DeviceType, Nat))
-- = 'Just '( '[5, 2, 3], 'D.Float, '( 'D.CPU, 0))
-- >>> type Ty = BroadcastTensorsImpl '[Tensor '( 'D.CPU, 0) 'D.Float '[1, 3], Tensor '( 'D.CPU, 0) 'D.Float '[2, 1], Tensor '( 'D.CPU, 0) 'D.Float '[1, 5, 1]] 'Nothing
-- >>> :kind! Ty
-- Ty :: Maybe ([Nat], D.DType, (D.DeviceType, Nat))
-- = 'Nothing
type family BroadcastTensorsImpl (tensors :: [a]) (acc :: Maybe ([Nat], D.DType, (D.DeviceType, Nat))) :: Maybe ([Nat], D.DType, (D.DeviceType, Nat)) where
  BroadcastTensorsImpl '[]                                    'Nothing                                = 'Nothing
  BroadcastTensorsImpl '[]                                    ('Just '(reverseShape, dtype, device))  = 'Just '(Reverse reverseShape, dtype, device)
  BroadcastTensorsImpl (Tensor device dtype shape ': tensors) 'Nothing                                = BroadcastTensorsImpl tensors ('Just '(Reverse shape, dtype, device))
  BroadcastTensorsImpl (Tensor device dtype shape ': tensors) ('Just '(reverseShape', dtype, device)) = BroadcastTensorsImpl tensors (MaybeTriple (ComputeBroadcast (Reverse shape) reverseShape') ('Just dtype) ('Just device))
  BroadcastTensorsImpl (Tensor device dtype shape ': _)       ('Just _)                               = Nothing

type family BroadcastTensorsCheck (tensors :: [a]) (result :: Maybe ([Nat], D.DType, (D.DeviceType, Nat))) :: [a] where
  BroadcastTensorsCheck tensors 'Nothing                        = TypeError (    Text "Cannot broadcast tensors due to incompatible shapes and/or dtypes: "
                                                                            :<>: ShowType tensors
                                                                            )
  BroadcastTensorsCheck tensors ('Just '(shape, dtype, device)) = HReplicateR (ListLength tensors) (Tensor device dtype shape)

type BroadcastTensors tensors
  = BroadcastTensorsCheck tensors (BroadcastTensorsImpl tensors 'Nothing)

-- | broadcast tensors
-- TODO: broadcastTensors returns garbage data and is hence broken
-- See https://pytorch.org/docs/stable/_modules/torch/functional.html#broadcast_tensors.
--
-- >>> x = ones :: CPUTensor 'D.Float '[1, 3]
-- >>> y = ones :: CPUTensor 'D.Float '[2, 1]
-- >>> z = ones :: CPUTensor 'D.Float '[5, 1, 1]
-- 
-- -- >>> x' :. y' :. z' :. HNil = broadcastTensors (x :. y :. z :. HNil)
-- -- >>> :type x'
-- -- x' :: Tensor '( 'D.CPU, 0) 'D.Float '[5, 2, 3]
-- -- >>> dtype &&& shape &&& (\t -> D.asValue (toDynamic t) :: [[[Float]]]) $ x'
-- -- >>> :type y'
-- -- y' :: Tensor '( 'D.CPU, 0) 'D.Float '[5, 2, 3]
-- -- >>> dtype &&& shape &&& (\t -> D.asValue (toDynamic t) :: [[[Float]]]) $ y'
-- -- >>> :type z'
-- -- z' :: Tensor '( 'D.CPU, 0) 'D.Float '[5, 2, 3]
-- -- >>> dtype &&& shape &&& (\t -> D.asValue (toDynamic t) :: [[[Float]]]) $ z'
broadcastTensors
  :: forall tensors tensors'
   . ( tensors' ~ BroadcastTensors tensors
     , ATen.Castable (HList tensors)  [D.ATenTensor]
     , ATen.Castable (HList tensors') [D.ATenTensor]
     )
  => HList tensors -- ^ input list of tensors
  -> HList tensors' -- ^ output list of tensors
broadcastTensors tensors = unsafePerformIO $ ATen.cast1 ATen.Managed.broadcast_tensors_l tensors

type family CatImpl (dim :: Nat) (tensors :: [a]) (acc :: Maybe ([Nat], D.DType, (D.DeviceType, Nat))) :: Maybe ([Nat], D.DType, (D.DeviceType, Nat)) where
  CatImpl _   '[]                                    acc = acc
  CatImpl dim (Tensor device dtype shape ': tensors) acc = CatImpl dim tensors (MaybeTriple (ComputeCatShape dim shape acc) (ComputeCatDType dtype acc) (ComputeCatDevice device acc))

type family ComputeCatShape (dim :: Nat) (shape :: [Nat]) (acc :: Maybe ([Nat], D.DType, (D.DeviceType, Nat))) :: Maybe [Nat] where
  ComputeCatShape 0   (x ': xs) Nothing                          = Just (x ': xs)
  ComputeCatShape dim (x ': xs) Nothing                          = AppendToMaybe x (ComputeCatShape (dim - 1) xs Nothing)
  ComputeCatShape 0   (x ': xs) (Just '(y ': xs, _, _))          = Just ((x + y) ': xs)
  ComputeCatShape dim (x ': xs) (Just '(x ': ys, dtype, device)) = AppendToMaybe x (ComputeCatShape (dim - 1) xs (Just '(ys, dtype, device)))
  ComputeCatShape _   _         _                                = Nothing

type family ComputeCatDType (dtype :: D.DType) (acc :: Maybe ([Nat], D.DType, (D.DeviceType, Nat))) :: Maybe D.DType where
  ComputeCatDType dtype Nothing               = Just dtype
  ComputeCatDType dtype (Just '(_, dtype, _)) = Just dtype
  ComputeCatDType _     _                     = Nothing

type family ComputeCatDevice (device :: (D.DeviceType, Nat)) (acc :: Maybe ([Nat], D.DType, (D.DeviceType, Nat))) :: Maybe (D.DeviceType, Nat) where
  ComputeCatDevice device Nothing                = Just device
  ComputeCatDevice device (Just '(_, _, device)) = Just device
  ComputeCatDevice _      _                      = Nothing

type family CatCheck (res :: Maybe ([Nat], D.DType, (D.DeviceType, Nat))) :: ([Nat], D.DType, (D.DeviceType, Nat)) where
  CatCheck 'Nothing                        = TypeError (Text "Concatenation impossible.")
  CatCheck ('Just '(shape, dtype, device)) = '(shape, dtype, device)

-- | Cat
--
-- >>> type Ty = Cat 0 '[Tensor '( 'D.CPU, 0) 'D.Float '[1]]
-- >>> :kind! Ty
-- Ty :: ([Nat], D.DType, (D.DeviceType, Nat))
-- = '( '[1], 'D.Float, '( 'D.CPU, 0))
-- >>> type Ty = Cat 0 '[Tensor '( 'D.CPU, 0) 'D.Float '[1], Tensor '( 'D.CPU, 0) 'D.Float '[2]]
-- >>> :kind! Ty
-- Ty :: ([Nat], D.DType, (D.DeviceType, Nat))
-- = '( '[3], 'D.Float, '( 'D.CPU, 0))
-- >>> type Ty = Cat 0 '[Tensor '( 'D.CPU, 0) 'D.Float '[1, 3], Tensor '( 'D.CPU, 0) 'D.Float '[2, 3]]
-- >>> :kind! Ty
-- Ty :: ([Nat], D.DType, (D.DeviceType, Nat))
-- = '( '[3, 3], 'D.Float, '( 'D.CPU, 0))
-- >>> type Ty = Cat 1 '[Tensor '( 'D.CPU, 0) 'D.Float '[3, 1], Tensor '( 'D.CPU, 0) 'D.Float '[3, 2]]
-- >>> :kind! Ty
-- Ty :: ([Nat], D.DType, (D.DeviceType, Nat))
-- = '( '[3, 3], 'D.Float, '( 'D.CPU, 0))
-- >>> type Ty = Cat 1 '[Tensor '( 'D.CPU, 0) 'D.Float '[2, 5, 4, 2], Tensor '( 'D.CPU, 0) 'D.Float '[2, 1, 4, 2], Tensor '( 'D.CPU, 0) 'D.Float '[2, 3, 4, 2], Tensor '( 'D.CPU, 0) 'D.Float '[2, 1, 4, 2]]
-- >>> :kind! Ty
-- Ty :: ([Nat], D.DType, (D.DeviceType, Nat))
-- = '( '[2, 10, 4, 2], 'D.Float, '( 'D.CPU, 0))
type Cat dim tensors = CatCheck (CatImpl dim tensors Nothing)

-- | cat
--
-- >>> t = ones :: CPUTensor 'D.Float '[2,2]
-- >>> t' = cat @0 (t :. HNil)
-- >>> :type t'
-- t' :: Tensor '( 'D.CPU, 0) 'D.Float '[2, 2]
-- >>> dtype &&& shape &&& (\t'' -> D.asValue (toDynamic t'') :: [[Float]]) $ t'
-- (Float,([2,2],[[1.0,1.0],[1.0,1.0]]))
-- >>> t' = cat @1 (t :. HNil)
-- >>> :type t'
-- t' :: Tensor '( 'D.CPU, 0) 'D.Float '[2, 2]
-- >>> dtype &&& shape &&& (\t'' -> D.asValue (toDynamic t'') :: [[Float]]) $ t'
-- (Float,([2,2],[[1.0,1.0],[1.0,1.0]]))
-- >>> t' = cat @0 (t :. t :. t :. HNil)
-- >>> :type t'
-- t' :: Tensor '( 'D.CPU, 0) 'D.Float '[6, 2]
-- >>> dtype &&& shape &&& (\t'' -> D.asValue (toDynamic t'') :: [[Float]]) $ t'
-- (Float,([6,2],[[1.0,1.0],[1.0,1.0],[1.0,1.0],[1.0,1.0],[1.0,1.0],[1.0,1.0]]))
-- >>> t' = cat @1 (t :. t :. t :. HNil)
-- >>> :type t'
-- t' :: Tensor '( 'D.CPU, 0) 'D.Float '[2, 6]
-- >>> dtype &&& shape &&& (\t'' -> D.asValue (toDynamic t'') :: [[Float]]) $ t'
-- (Float,([2,6],[[1.0,1.0,1.0,1.0,1.0,1.0],[1.0,1.0,1.0,1.0,1.0,1.0]]))
cat
  :: forall dim shape dtype device tensors
   . ( KnownNat dim
     , '(shape, dtype, device) ~ Cat dim tensors
     , ATen.Castable (HList tensors) [D.ATenTensor]
     )
  => HList tensors -- ^ input list of tensors
  -> Tensor device dtype shape -- ^ output tensor
cat tensors = unsafePerformIO $ ATen.cast2 ATen.Managed.cat_ll tensors (natValI @dim :: Int)

-- chain_matmul :: [Tensor device dtype shape] -> Tensor device dtype shape
-- chain_matmul _matrices = unsafePerformIO $ (ATen.cast1 ATen.Managed.chain_matmul_l) _matrices

type family ChunkImpl (chunkShapes :: Maybe [[Nat]]) (dtype :: D.DType) (device :: (D.DeviceType, Nat)) :: Maybe a where
  ChunkImpl (Just '[])               _     _      = Just '[]
  ChunkImpl (Just (shape ': shapes)) dtype device = AppendToMaybe (Tensor device dtype shape) (ChunkImpl (Just shapes) dtype device)
  ChunkImpl Nothing                  _     _      = Nothing

type family ChunkCheck (shape :: [Nat]) (dim :: Nat) (result :: Maybe a) :: a where
  ChunkCheck shape dim Nothing       = DimOutOfBound shape dim
  ChunkCheck _     _   (Just result) = result

type family ComputeChunksChunkGo (n' :: Nat) (r :: Nat) (cmp :: Ordering) (cmp' :: Ordering) :: [Nat] where
  ComputeChunksChunkGo n' r GT _  = n' ': ComputeChunksChunkGo n' (r - n') (CmpNat (r - n') n') (CmpNat (r - n') 0)
  ComputeChunksChunkGo n' r EQ _  = n' ': ComputeChunksChunkGo n' (r - n') (CmpNat (r - n') n') (CmpNat (r - n') 0)
  ComputeChunksChunkGo n' r _  GT = '[r]
  ComputeChunksChunkGo n' _ _  _  = '[]

type family ComputeChunksChunkGo0 (n' :: Nat) (chunks :: Nat) :: [Nat] where
  ComputeChunksChunkGo0 _  0      = '[]
  ComputeChunksChunkGo0 n' chunks = n' ': (ComputeChunksChunkGo0 n' (chunks - 1))

type family ComputeChunks (n :: Maybe Nat) (chunks :: Nat) :: Maybe [Nat] where
  ComputeChunks (Just n) chunks = Just (ComputeChunks' n chunks (Mod n chunks))
  ComputeChunks Nothing  _      = Nothing

type family ComputeChunks' (n :: Nat) (chunks :: Nat) (m :: Nat) :: [Nat] where
  ComputeChunks' n chunks 0 = ComputeChunksChunkGo0 (Div n chunks) chunks
  ComputeChunks' n chunks _ = ComputeChunksChunkGo (Div (n + chunks - 1) chunks) n (CmpNat n (Div (n + chunks - 1) chunks)) (CmpNat n 0)

type family ChunkShapesImpl (chunks :: Maybe [Nat]) (dim :: Nat) (shape :: [Nat]) :: Maybe [[Nat]] where
  ChunkShapesImpl (Just (n ': ns)) dim shape = AppendToMaybe' (ReplaceDim dim shape n) (ChunkShapesImpl (Just ns) dim shape)
  ChunkShapesImpl (Just '[])       _   _     = Just '[]
  ChunkShapesImpl Nothing          _   _     = Nothing

type ChunkShapes chunks dim shape = ChunkShapesImpl (ComputeChunks (ExtractDim dim shape) chunks) dim shape

type Chunk chunks dim shape dtype device = ChunkCheck shape dim (ChunkImpl (ChunkShapes chunks dim shape) dtype device)

-- | chunk
--
-- -- >>> :type chunk @3 @1 (ones :: CPUTensor 'D.Float '[2, 2])
-- -- chunk @3 @1 (ones :: CPUTensor 'D.Float '[2, 2])
-- --   :: HList
-- --        '[Tensor '( 'D.CPU, 0) 'D.Float '[2, 1],
-- --          Tensor '( 'D.CPU, 0) 'D.Float '[2, 1]]
-- >>> t0 :. t1 :. HNil = chunk @3 @1 (ones :: CPUTensor 'D.Float '[2, 2])
-- >>> dtype &&& shape $ t0
-- (Float,[2,1])
-- >>> dtype &&& shape $ t1
-- (Float,[2,1])
--
-- -- >>> :type chunk @3 @1 (ones :: CPUTensor 'D.Float '[1, 0, 3])
-- -- chunk @3 @1 (ones :: CPUTensor 'D.Float '[1, 0, 3])
-- --   :: HList
-- --        '[Tensor '( 'D.CPU, 0) 'D.Float '[1, 0, 3],
-- --          Tensor '( 'D.CPU, 0) 'D.Float '[1, 0, 3],
-- --          Tensor '( 'D.CPU, 0) 'D.Float '[1, 0, 3]]
-- >>> t0 :. t1 :. t2 :. HNil = chunk @3 @1 (ones :: CPUTensor 'D.Float '[1, 0, 3])
-- >>> dtype &&& shape $ t0
-- (Float,[1,0,3])
-- >>> dtype &&& shape $ t1
-- (Float,[1,0,3])
-- >>> dtype &&& shape $ t2
-- (Float,[1,0,3])
--
-- -- >>> :type chunk @6 @0 (ones :: CPUTensor 'D.Float '[19, 4])
-- -- chunk @6 @0 (ones :: CPUTensor 'D.Float '[19, 4])
-- --   :: HList
-- --        '[Tensor '( 'D.CPU, 0) 'D.Float '[4, 4],
-- --          Tensor '( 'D.CPU, 0) 'D.Float '[4, 4],
-- --          Tensor '( 'D.CPU, 0) 'D.Float '[4, 4],
-- --          Tensor '( 'D.CPU, 0) 'D.Float '[4, 4],
-- --          Tensor '( 'D.CPU, 0) 'D.Float '[3, 4]]
-- >>> t0 :. t1 :. t2 :. t3 :. t4 :. HNil = chunk @6 @0 (ones :: CPUTensor 'D.Float '[19, 4])
-- >>> dtype &&& shape $ t0
-- (Float,[4,4])
-- >>> dtype &&& shape $ t1
-- (Float,[4,4])
-- >>> dtype &&& shape $ t2
-- (Float,[4,4])
-- >>> dtype &&& shape $ t3
-- (Float,[4,4])
-- >>> dtype &&& shape $ t4
-- (Float,[3,4])
chunk
  :: forall chunks dim shape dtype device tensorChunks
   . ( KnownNat chunks
     , KnownNat dim
     , tensorChunks ~ Chunk chunks dim shape dtype device
     , ATen.Castable (HList tensorChunks) [D.ATenTensor]
     )
  => Tensor device dtype shape -- ^ input tensor
  -> HList tensorChunks -- ^ output list of tensors
chunk input = unsafePerformIO
  $ ATen.cast3 ATen.Managed.chunk_tll input (natValI @chunks :: Int) (natValI @dim :: Int)

-- | clamp
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
-- TODO: can we use D.Scalar for the minimum and maximum values?
--
-- >>> dtype &&& shape $ clamp 0 1 (ones :: CPUTensor 'D.Float '[3,2])
-- (Float,[3,2])
clamp
  :: forall shape dtype device
   . Float -- ^ minimum value
  -> Float -- ^ maximum value
  -> Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
clamp min max input = unsafePerformIO $ ATen.cast3 ATen.Managed.clamp_tss input min max

-- | clampMax
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
-- TODO: can we use D.Scalar for the maximum value?
--
-- >>> dtype &&& shape $ clampMax 1 (ones :: CPUTensor 'D.Float '[3,2])
-- (Float,[3,2])
clampMax
  :: forall shape dtype device
   . Float -- ^ maximum value
  -> Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
clampMax max input = unsafePerformIO $ ATen.cast2 ATen.Managed.clamp_max_ts input max

-- | clampMin
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
-- TODO: can we use D.Scalar for the minimum value?
--
-- >>> dtype &&& shape $ clampMin 0 (ones :: CPUTensor 'D.Float '[3,2])
-- (Float,[3,2])
clampMin
  :: forall shape dtype device
   . Float -- ^ minimum value
  -> Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
clampMin min input = unsafePerformIO $ ATen.cast2 ATen.Managed.clamp_min_ts input min

-- | cudnnIsAcceptable
-- TODO: calling this probably makes only sense when the device is CUDA
cudnnIsAcceptable
  :: forall shape dtype device
   . Tensor device dtype shape -- ^ input
  -> Bool -- ^ output
cudnnIsAcceptable input =
  unsafePerformIO $ ATen.cast1 ATen.Managed.cudnn_is_acceptable_t input

-- constant_pad_nd :: Tensor device dtype shape -> [Int] -> Float -> Tensor device dtype shape
-- constant_pad_nd _input _pad _value = unsafePerformIO $ (ATen.cast3 ATen.Managed.constant_pad_nd_tls) _input _pad _value

constantPadNd1d
  :: forall (pad :: (Nat, Nat)) n dtype device
   . (All KnownNat '[Torch.Typed.Aux.Fst pad, Torch.Typed.Aux.Snd pad, n])
  => Float
  -> Tensor device dtype '[n]
  -> Tensor device dtype '[n + Torch.Typed.Aux.Fst pad + Torch.Typed.Aux.Snd pad]
constantPadNd1d value input = unsafePerformIO $ ATen.cast3
  ATen.Managed.constant_pad_nd_tls
  input
  ([natValI @(Torch.Typed.Aux.Fst pad), natValI @(Torch.Typed.Aux.Snd pad)] :: [Int])
  value

-- convolution :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> [Int] -> [Int] -> [Int] -> Bool -> [Int] -> Int -> Tensor device dtype shape
-- convolution _input _weight _bias _stride _padding _dilation _transposed _output_padding _groups = unsafePerformIO $ (ATen.cast9 ATen.Managed.convolution_tttlllbll) _input _weight _bias _stride _padding _dilation _transposed _output_padding _groups

type ConvSideCheck (inputSize :: Nat) (kernelSize :: Nat) (stride :: Nat) (padding :: Nat) (outputSize :: Nat) =
  (
    -- kernel size and stride must be > 0
    1 <= kernelSize, 1 <= stride
    -- kernel size can't be greater than actual input size
    -- ToDo: Do not use '>=' on constraint to avoid reduction-stack-overflow.
  , (kernelSize - 1) <= (inputSize + (2 * padding))
    -- output size must be greater than 0
  , 1 <= outputSize
    -- output formulation:
  , outputSize ~ ConvOutputSize inputSize kernelSize stride padding
  )

-- | ConvOutputSize
--
-- >>> :kind! ConvOutputSize 4 1 1 0
-- ConvOutputSize 4 1 1 0 :: Nat
-- = 4
type family ConvOutputSize (inputSize :: Nat) (kernelSize :: Nat) (stride :: Nat) (padding :: Nat) :: Nat where
  ConvOutputSize inputSize kernelSize stride padding = (Div ((inputSize + (2 * padding)) - kernelSize) stride) + 1

-- | conv1d
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
--
-- >>> t = conv1d @1 @0 (ones :: CPUTensor 'D.Float '[10, 3, 1]) (ones :: CPUTensor 'D.Float '[10]) (ones :: CPUTensor 'D.Float '[1, 3, 4])
-- >>> :type t
-- t :: Tensor '( 'D.CPU, 0) 'D.Float '[1, 10, 4]
-- >>> dtype &&& shape &&& (\t' -> D.asValue (toDynamic t') :: [[[Float]]]) $ t
-- (Float,([1,10,4],[[[4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0]]]))
conv1d
  :: forall (stride :: Nat)
            (padding :: Nat)
            inputChannelSize outputChannelSize
            kernelSize
            inputSize
            batchSize
            outputSize
            dtype
            device
   . ( All KnownNat '[ stride
                     , padding
                     , inputChannelSize, outputChannelSize
                     , kernelSize
                     , inputSize
                     , batchSize
                     , outputSize
                     ]
     , ConvSideCheck inputSize kernelSize stride padding outputSize
     )
  => Tensor device dtype '[outputChannelSize, inputChannelSize, kernelSize] -- ^ weight
  -> Tensor device dtype '[outputChannelSize] -- ^ bias
  -> Tensor device dtype '[batchSize, inputChannelSize, inputSize] -- ^ input
  -> Tensor device dtype '[batchSize, outputChannelSize, outputSize] -- ^ output
conv1d weight bias input = unsafePerformIO $ ATen.cast7 ATen.Managed.conv1d_tttllll
                                                   input
                                                   weight
                                                   bias
                                                   (natValI @stride :: Int)
                                                   (natValI @padding :: Int)
                                                   (1 :: Int)
                                                   (1 :: Int)

-- | conv2d
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
--
-- >>> t = conv2d @'(1, 1) @'(0, 0) (ones :: CPUTensor 'D.Float '[10, 3, 1, 1]) (ones :: CPUTensor 'D.Float '[10]) (ones :: CPUTensor 'D.Float '[1, 3, 4, 5])
-- >>> :type t
-- t :: Tensor '( 'D.CPU, 0) 'D.Float '[1, 10, 4, 5]
-- >>> dtype &&& shape &&& (\t' -> D.asValue (toDynamic t') :: [[[[Float]]]]) $ t
-- (Float,([1,10,4,5],[[[[4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0]],[[4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0]],[[4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0]],[[4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0]],[[4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0]],[[4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0]],[[4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0]],[[4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0]],[[4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0]],[[4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0]]]]))
conv2d
  :: forall (stride :: (Nat, Nat))
            (padding :: (Nat, Nat))
            inputChannelSize outputChannelSize
            kernelSize0 kernelSize1
            inputSize0 inputSize1
            batchSize
            outputSize0 outputSize1
            dtype
            device
   . ( All KnownNat '[ Torch.Typed.Aux.Fst stride, Torch.Typed.Aux.Snd stride
                     , Torch.Typed.Aux.Fst padding, Torch.Typed.Aux.Snd padding
                     , inputChannelSize, outputChannelSize
                     , kernelSize0, kernelSize1
                     , inputSize0, inputSize1
                     , batchSize
                     , outputSize0, outputSize1
                     ]
     , ConvSideCheck inputSize0 kernelSize0 (Torch.Typed.Aux.Fst stride) (Torch.Typed.Aux.Fst padding) outputSize0
     , ConvSideCheck inputSize1 kernelSize1 (Torch.Typed.Aux.Snd stride) (Torch.Typed.Aux.Snd padding) outputSize1
     )
  => Tensor device dtype '[outputChannelSize, inputChannelSize, kernelSize0, kernelSize1] -- ^ weight
  -> Tensor device dtype '[outputChannelSize] -- ^ bias
  -> Tensor device dtype '[batchSize, inputChannelSize, inputSize0, inputSize1] -- ^ input
  -> Tensor device dtype '[batchSize, outputChannelSize, outputSize0, outputSize1] -- ^ output
conv2d weight bias input = unsafePerformIO $ ATen.cast7
  ATen.Managed.conv2d_tttllll
  input
  weight
  bias
  ([natValI @(Torch.Typed.Aux.Fst stride),  natValI @(Torch.Typed.Aux.Snd stride)]  :: [Int])
  ([natValI @(Torch.Typed.Aux.Fst padding), natValI @(Torch.Typed.Aux.Snd padding)] :: [Int])
  ([1, 1] :: [Int])
  (1 :: Int)

-- | conv3d
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
--
-- >>> t = conv3d @'(1, 1, 1) @'(0, 0, 0) (ones :: CPUTensor 'D.Float '[10, 3, 1, 1, 1]) (ones :: CPUTensor 'D.Float '[10]) (ones :: CPUTensor 'D.Float '[1, 3, 4, 5, 6])
-- >>> :type t
-- t :: Tensor '( 'D.CPU, 0) 'D.Float '[1, 10, 4, 5, 6]
-- >>> dtype &&& shape &&& (\t' -> D.asValue (toDynamic t') :: [[[[[Float]]]]]) $ t
-- (Float,([1,10,4,5,6],[[[[[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0]],[[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0]],[[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0]],[[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0]]],[[[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0]],[[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0]],[[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0]],[[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0]]],[[[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0]],[[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0]],[[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0]],[[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0]]],[[[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0]],[[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0]],[[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0]],[[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0]]],[[[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0]],[[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0]],[[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0]],[[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0]]],[[[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0]],[[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0]],[[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0]],[[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0]]],[[[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0]],[[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0]],[[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0]],[[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0]]],[[[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0]],[[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0]],[[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0]],[[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0]]],[[[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0]],[[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0]],[[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0]],[[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0]]],[[[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0]],[[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0]],[[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0]],[[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0],[4.0,4.0,4.0,4.0,4.0,4.0]]]]]))
conv3d
  :: forall (stride :: (Nat, Nat, Nat))
            (padding :: (Nat, Nat, Nat))
            inputChannelSize outputChannelSize
            kernelSize0 kernelSize1 kernelSize2
            inputSize0 inputSize1 inputSize2
            batchSize
            outputSize0 outputSize1 outputSize2
            dtype
            device
   . ( All KnownNat '[ Fst3 stride, Snd3 stride, Trd3 stride
                     , Fst3 padding, Snd3 padding, Trd3 padding
                     , inputChannelSize, outputChannelSize
                     , kernelSize0, kernelSize1, kernelSize2
                     , inputSize0, inputSize1, inputSize2
                     , batchSize
                     ]
     , ConvSideCheck inputSize0 kernelSize0 (Fst3 stride) (Fst3 padding) outputSize0
     , ConvSideCheck inputSize1 kernelSize1 (Snd3 stride) (Snd3 padding) outputSize1
     , ConvSideCheck inputSize2 kernelSize2 (Trd3 stride) (Trd3 padding) outputSize2
     )
  => Tensor device dtype '[outputChannelSize, inputChannelSize, kernelSize0, kernelSize1, kernelSize2] -- ^ weight
  -> Tensor device dtype '[outputChannelSize] -- ^ bias
  -> Tensor device dtype '[batchSize, inputChannelSize, inputSize0, inputSize1, inputSize2] -- ^ input
  -> Tensor device dtype '[batchSize, outputChannelSize, outputSize0, outputSize1, outputSize2] -- ^ output
conv3d weight bias input = unsafePerformIO $ ATen.cast7
  ATen.Managed.conv3d_tttllll
  input
  weight
  bias
  ([natValI @(Fst3 stride), natValI @(Snd3 stride), natValI @(Trd3 stride)] :: [Int])
  ([natValI @(Fst3 padding), natValI @(Snd3 padding), natValI @(Trd3 padding)] :: [Int])
  ([1, 1, 1] :: [Int])
  (1 :: Int)

-- | convTBC
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
-- 1D convolution over an input of shape `[timeSize, batchSize, inputChannels]`.

-- >>> dtype &&& shape $ convTBC @1 (ones :: CPUTensor 'D.Float '[1,4,5]) (ones :: CPUTensor 'D.Float '[5]) (ones :: CPUTensor 'D.Float '[3,3,4])
-- (Float,[5,3,5])
-- >>> dtype &&& shape $ convTBC @0 (ones :: CPUTensor 'D.Float '[1,4,5]) (ones :: CPUTensor 'D.Float '[5]) (ones :: CPUTensor 'D.Float '[2,3,4])
-- (Float,[3,3,5])
-- >>> dtype &&& shape $ convTBC @0 (ones :: CPUTensor 'D.Float '[2,4,5]) (ones :: CPUTensor 'D.Float '[5]) (ones :: CPUTensor 'D.Float '[2,3,4])
-- (Float,[2,3,5])
convTBC
  :: forall padding timeSize batchSize kernelSize inputChannels outputChannels dtype device
   . (KnownNat padding)
  => Tensor device dtype '[kernelSize, inputChannels, outputChannels]
  -> Tensor device dtype '[outputChannels]
  -> Tensor device dtype '[timeSize, batchSize, inputChannels]
  -> Tensor device dtype '[timeSize+padding*2+1-kernelSize, batchSize, outputChannels]
convTBC weight bias input =
  unsafePerformIO $ ATen.cast4 ATen.Managed.conv_tbc_tttl input weight bias (natValI @padding)

-- conv_transpose1d :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Int -> Int -> Int -> Int -> Int -> Tensor device dtype shape
-- conv_transpose1d _input _weight _bias _stride _padding _output_padding _groups _dilation = unsafePerformIO $ (ATen.cast8 ATen.Managed.conv_transpose1d_tttlllll) _input _weight _bias _stride _padding _output_padding _groups _dilation

-- | cosh
--
-- >>> dtype &&& shape $ cosh (ones :: CPUTensor 'D.Float '[3,2])
-- (Float,[3,2])
cosh
  :: forall shape dtype device
   . (StandardFloatingPointDTypeValidation device dtype)
  => Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
cosh input = unsafePerformIO $ ATen.cast1 ATen.Managed.cosh_t input

-- cosine_embedding_loss :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Double -> Int -> Tensor device dtype shape
-- cosine_embedding_loss _input1 _input2 _target _margin _reduction = unsafePerformIO $ (ATen.cast5 ATen.Managed.cosine_embedding_loss_tttdl) _input1 _input2 _target _margin _reduction

-- cudnn_affine_grid_generator :: Tensor device dtype shape -> Int -> Int -> Int -> Int -> Tensor device dtype shape
-- cudnn_affine_grid_generator _theta _N _C _H _W = unsafePerformIO $ (ATen.cast5 ATen.Managed.cudnn_affine_grid_generator_tllll) _theta _N _C _H _W

-- cudnn_batch_norm :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Bool -> Double -> Double -> (Tensor device dtype shape,Tensor device dtype shape,Tensor device dtype shape)
-- cudnn_batch_norm _input _weight _bias _running_mean _running_var _training _exponential_average_factor _epsilon = unsafePerformIO $ (ATen.cast8 ATen.Managed.cudnn_batch_norm_tttttbdd) _input _weight _bias _running_mean _running_var _training _exponential_average_factor _epsilon

-- cudnn_convolution :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> [Int] -> [Int] -> [Int] -> Int -> Bool -> Bool -> Tensor device dtype shape
-- cudnn_convolution _input _weight _bias _padding _stride _dilation _groups _benchmark _deterministic = unsafePerformIO $ (ATen.cast9 ATen.Managed.cudnn_convolution_tttllllbb) _input _weight _bias _padding _stride _dilation _groups _benchmark _deterministic

-- cudnn_convolution_transpose :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> [Int] -> [Int] -> [Int] -> [Int] -> Int -> Bool -> Bool -> Tensor device dtype shape
-- cudnn_convolution_transpose _input _weight _bias _padding _output_padding _stride _dilation _groups _benchmark _deterministic = unsafePerformIO $ (ATen.ATen.cast10 ATen.Managed.cudnn_convolution_transpose_tttlllllbb) _input _weight _bias _padding _output_padding _stride _dilation _groups _benchmark _deterministic

-- cudnn_grid_sampler :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape
-- cudnn_grid_sampler _input _grid = unsafePerformIO $ (ATen.cast2 ATen.Managed.cudnn_grid_sampler_tt) _input _grid

-- | Det
--
-- >>> :kind! Det '[2,2]
-- Det '[2,2] :: [Nat]
-- = '[]
-- >>> :kind! Det '[3,2,2]
-- Det '[3,2,2] :: [Nat]
-- = '[3]
type family Det (shape :: [Nat]) :: [Nat] where
  Det (n:n:'[])   = '[]
  Det (b:n:n:'[]) = '[b]
  Det _           = TypeError (Text "This shape must be square matrix or batch + squre matrix.")

-- | det
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
--
-- >>> dtype &&& shape $ det (ones :: CPUTensor 'D.Float '[2,2])
-- (Float,[])
-- >>> dtype &&& shape $ det (ones :: CPUTensor 'D.Float '[3,2,2])
-- (Float,[3])
det
  :: forall shape dtype device
   . Tensor device dtype shape -- ^ input
  -> Tensor device dtype (Det shape) -- ^ output
det input = unsafePerformIO $ ATen.cast1 ATen.Managed.det_t input

-- diag_embed :: Tensor device dtype shape -> Int -> Int -> Int -> Tensor device dtype shape
-- diag_embed _input _offset _dim1 _dim2 = unsafePerformIO $ (ATen.cast4 ATen.Managed.diag_embed_tlll) _input _offset _dim1 _dim2

-- diagflat :: Tensor device dtype shape -> Int -> Tensor device dtype shape
-- diagflat _input _offset = unsafePerformIO $ (ATen.cast2 ATen.Managed.diagflat_tl) _input _offset

-- diagonal :: Tensor device dtype shape -> Int -> Int -> Int -> Tensor device dtype shape
-- diagonal _input _offset _dim1 _dim2 = unsafePerformIO $ (ATen.cast4 ATen.Managed.diagonal_tlll) _input _offset _dim1 _dim2

type family DotDTypeIsValid (device :: (D.DeviceType, Nat)) (dtype :: D.DType) :: Constraint where
  DotDTypeIsValid '( 'D.CPU, 0)            dtype = ( DTypeIsNotBool '( 'D.CPU, 0) dtype
                                                   , DTypeIsNotHalf '( 'D.CPU, 0) dtype
                                                   )
  DotDTypeIsValid '( 'D.CUDA, deviceIndex) dtype = DTypeIsFloatingPoint '( 'D.CUDA, deviceIndex) dtype
  DotDTypeIsValid '(deviceType, _)         dtype = UnsupportedDTypeForDevice deviceType dtype

-- | dot product
-- Note that this function does not broadcast.
dot
  :: forall size dtype device
   . DotDTypeIsValid device dtype
  => Tensor device dtype '[size] -- ^ input 
  -> Tensor device dtype '[size] -- ^ other input
  -> Tensor device dtype '[] -- ^ dot product
dot input other = unsafePerformIO $ ATen.cast2 ATen.Managed.dot_tt input other

-- einsum :: String -> [Tensor device dtype shape] -> Tensor device dtype shape
-- einsum _equation _tensors = unsafePerformIO $ (ATen.cast2 ATen.Managed.einsum_sl) _equation _tensors

class KnownMaybeNat (n :: Maybe Nat) where
  maybeNatVal :: Maybe Integer

instance (KnownNat n) => KnownMaybeNat (Just n) where
  maybeNatVal = Just . natVal $ Proxy @n

instance KnownMaybeNat Nothing where
  maybeNatVal = Nothing

type family PaddingIdxCheck (idx :: Maybe Nat) (numEmbeds :: Nat) :: Constraint where
  PaddingIdxCheck (Just n) numEmbeds = n + 1 <= numEmbeds
  PaddingIdxCheck Nothing  _         = ()

-- | embedding
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
-- TODO: what about sparsity here?
-- TODO: what output dtypes are supported?
--
-- >>> weights = fromJust [[1, 1], [2, 2], [3, 3], [4, 4]] :: CPUTensor 'D.Float '[4, 2]
-- >>> indices = fromJust [[0], [2], [0], [1]] :: CPUTensor 'D.Int64 '[4, 1]
-- >>> t = embedding @('Just 0) False False weights indices
-- >>> :type t
-- t :: Tensor '( 'D.CPU, 0) 'D.Float '[4, 1, 2]
-- >>> dtype &&& shape &&& (\t' -> D.asValue (toDynamic t') :: [[[Float]]]) $ t
-- (Float,([4,1,2],[[[1.0,1.0]],[[3.0,3.0]],[[1.0,1.0]],[[2.0,2.0]]]))
-- >>> t = embedding @'Nothing False False weights indices
-- >>> :type t
-- t :: Tensor '( 'D.CPU, 0) 'D.Float '[4, 1, 2]
-- >>> dtype &&& shape &&& (\t' -> D.asValue (toDynamic t') :: [[[Float]]]) $ t
-- (Float,([4,1,2],[[[1.0,1.0]],[[3.0,3.0]],[[1.0,1.0]],[[2.0,2.0]]]))
embedding
  :: forall (paddingIdx :: Maybe Nat) numEmbeds embedDim shape dtype device
   . ( KnownMaybeNat paddingIdx
     , PaddingIdxCheck paddingIdx numEmbeds
     )
  => Bool -- ^ whether or not to scale the gradient by the frequencies
  -> Bool -- ^ whether or not the embedding is sparse
  -> Tensor device dtype    '[numEmbeds, embedDim] -- ^ weights
  -> Tensor device 'D.Int64 shape -- ^ indices
  -> Tensor device dtype    (Reverse (embedDim ': Reverse shape)) -- ^ output
embedding scaleGradByFreq sparse weights indices =
  unsafePerformIO $ ATen.cast5 ATen.Managed.embedding_ttlbb weights indices paddingIdx scaleGradByFreq sparse
 where paddingIdx :: Int
       paddingIdx = case maybeNatVal @paddingIdx of
                      Just idx -> fromIntegral idx
                      Nothing  -> -1

-- embedding_bag :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Bool -> Int -> Bool -> Tensor device dtype shape -> (Tensor device dtype shape,Tensor device dtype shape,Tensor device dtype shape,Tensor device dtype shape)
-- embedding_bag _weight _indices _offsets _scale_grad_by_freq _mode _sparse _per_sample_weights = unsafePerformIO $ (ATen.cast7 ATen.Managed.embedding_bag_tttblbt) _weight _indices _offsets _scale_grad_by_freq _mode _sparse _per_sample_weights

-- | emptyLike
-- TODO: this seems quite unsafe, the values of this tensor will be random
--
-- >>> t <- emptyLike (ones :: CPUTensor 'D.Float '[3,4,5])
-- >>> dtype &&& shape $ t
-- (Float,[3,4,5])
emptyLike
  :: forall shape dtype device
   . Tensor device dtype shape -- ^ input
  -> IO (Tensor device dtype shape) -- ^ output
emptyLike input = ATen.cast1 ATen.Managed.empty_like_t input

-- | erfc
--
-- >>> dtype &&& shape $ erfc (ones :: CPUTensor 'D.Float '[3,2])
-- (Float,[3,2])
erfc
  :: forall shape dtype device
   . (StandardFloatingPointDTypeValidation device dtype)
  => Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
erfc input = unsafePerformIO $ ATen.cast1 ATen.Managed.erfc_t input

-- | expm1
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
--
-- >>> dtype &&& shape $ expm1 (ones :: CPUTensor 'D.Float '[3,2])
-- (Float,[3,2])
expm1
  :: forall shape dtype device
   . (StandardFloatingPointDTypeValidation device dtype)
  => Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
expm1 input = unsafePerformIO $ ATen.cast1 ATen.Managed.expm1_t input

-- | expand
-- TODO: figure out what the `implicit` boolean value does
--
-- >>> t = ones :: CPUTensor 'D.Float '[2]
-- >>> t' = expand @'[3, 1, 2] False t
-- >>> dtype &&& shape $ t'
-- (Float,[3,1,2])
-- >>> t'' = expand @'[3, 1, 2] True t
-- >>> dtype &&& shape $ t''
-- (Float,[3,1,2])
-- >>> toInt (all (t' ==. t'')) == 1
-- True
expand
  :: forall shape' shape dtype device
   . ( KnownShape shape'
     , shape' ~ Broadcast shape shape'
     )
  => Bool -- ^ some boolean value with unknown function
  -> Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape' -- ^ output
expand someBool input = unsafePerformIO $ ATen.cast3 ATen.Managed.tensor_expand_lb input (shapeVal @shape') someBool

-- flatten :: Tensor device dtype shape -> Int -> Int -> Tensor device dtype shape
-- flatten _input _start_dim _end_dim = unsafePerformIO $ (ATen.cast3 ATen.Managed.flatten_tll) _input _start_dim _end_dim

-- | flattenAll
--
-- >>> t = flattenAll (ones :: CPUTensor 'D.Float '[3,2])
-- >>> dtype &&& shape $ t
-- (Float,[6])
-- >>> :t t
-- t :: Tensor '( 'D.CPU, 0) 'D.Float '[6]
flattenAll
  :: forall shape dtype device
   . KnownShape shape
  => Tensor device dtype shape -- ^ input
  -> Tensor device dtype '[Product shape] -- ^ output
flattenAll input =
  unsafePerformIO $ ATen.cast3 ATen.Managed.flatten_tll input (0 :: Int) (-1 :: Int)

-- | frac
--
-- >>> dtype &&& shape $ frac (ones :: CPUTensor 'D.Float '[3,2])
-- (Float,[3,2])
frac
  :: forall shape dtype device
   . (StandardFloatingPointDTypeValidation device dtype)
  => Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
frac input = unsafePerformIO $ ATen.cast1 ATen.Managed.frac_t input

-- | full like
--
-- >>> dtype &&& shape $ fullLike 3.0 (ones :: CPUTensor 'D.Float '[3,2])
-- (Float,[3,2])
fullLike
  :: forall shape dtype device
   . Float -- ^ fill value
  -> Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
fullLike fillValue input =
  unsafePerformIO $ ATen.cast2 ATen.Managed.full_like_ts input fillValue

-- grid_sampler :: Tensor device dtype shape -> Tensor device dtype shape -> Int -> Int -> Tensor device dtype shape
-- grid_sampler _input _grid _interpolation_mode _padding_mode = unsafePerformIO $ (ATen.cast4 ATen.Managed.grid_sampler_ttll) _input _grid _interpolation_mode _padding_mode

-- grid_sampler_2d :: Tensor device dtype shape -> Tensor device dtype shape -> Int -> Int -> Tensor device dtype shape
-- grid_sampler_2d _input _grid _interpolation_mode _padding_mode = unsafePerformIO $ (ATen.cast4 ATen.Managed.grid_sampler_2d_ttll) _input _grid _interpolation_mode _padding_mode

-- grid_sampler_3d :: Tensor device dtype shape -> Tensor device dtype shape -> Int -> Int -> Tensor device dtype shape
-- grid_sampler_3d _input _grid _interpolation_mode _padding_mode = unsafePerformIO $ (ATen.cast4 ATen.Managed.grid_sampler_3d_ttll) _input _grid _interpolation_mode _padding_mode

-- hinge_embedding_loss :: Tensor device dtype shape -> Tensor device dtype shape -> Double -> Int -> Tensor device dtype shape
-- hinge_embedding_loss _input _target _margin _reduction = unsafePerformIO $ (ATen.cast4 ATen.Managed.hinge_embedding_loss_ttdl) _input _target _margin _reduction

-- ger :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape
-- ger _input _vec2 = unsafePerformIO $ (ATen.cast2 ATen.Managed.ger_tt) _input _vec2

-- group_norm :: Tensor device dtype shape -> Int -> Tensor device dtype shape -> Tensor device dtype shape -> Double -> Bool -> Tensor device dtype shape
-- group_norm _input _num_groups _weight _bias _eps _cudnn_enabled = unsafePerformIO $ (ATen.cast6 ATen.Managed.group_norm_tlttdb) _input _num_groups _weight _bias _eps _cudnn_enabled

-- fft :: Tensor device dtype shape -> Int -> Bool -> Tensor device dtype shape
-- fft _input _signal_ndim _normalized = unsafePerformIO $ (ATen.cast3 ATen.Managed.fft_tlb) _input _signal_ndim _normalized

-- ifft :: Tensor device dtype shape -> Int -> Bool -> Tensor device dtype shape
-- ifft _input _signal_ndim _normalized = unsafePerformIO $ (ATen.cast3 ATen.Managed.ifft_tlb) _input _signal_ndim _normalized

-- rfft :: Tensor device dtype shape -> Int -> Bool -> Bool -> Tensor device dtype shape
-- rfft _input _signal_ndim _normalized _onesided = unsafePerformIO $ (ATen.cast4 ATen.Managed.rfft_tlbb) _input _signal_ndim _normalized _onesided

-- irfft :: Tensor device dtype shape -> Int -> Bool -> Bool -> [Int] -> Tensor device dtype shape
-- irfft _input _signal_ndim _normalized _onesided _signal_sizes = unsafePerformIO $ (ATen.cast5 ATen.Managed.irfft_tlbbl) _input _signal_ndim _normalized _onesided _signal_sizes

-- index :: Tensor device dtype shape -> [Tensor device dtype shape] -> Tensor device dtype shape
-- index _input _indices = unsafePerformIO $ (ATen.cast2 ATen.Managed.index_tl) _input _indices

-- index_copy :: Tensor device dtype shape -> Int -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape
-- index_copy _input _dim _index _source = unsafePerformIO $ (ATen.cast4 ATen.Managed.index_copy_tltt) _input _dim _index _source

-- index_put :: Tensor device dtype shape -> [Tensor device dtype shape] -> Tensor device dtype shape -> Bool -> Tensor device dtype shape
-- index_put _input _indices _values _accumulate = unsafePerformIO $ (ATen.cast4 ATen.Managed.index_put_tltb) _input _indices _values _accumulate

-- instance_norm :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Bool -> Double -> Double -> Bool -> Tensor device dtype shape
-- instance_norm _input _weight _bias _running_mean _running_var _use_input_stats _momentum _eps _cudnn_enabled = unsafePerformIO $ (ATen.cast9 ATen.Managed.instance_norm_tttttbddb) _input _weight _bias _running_mean _running_var _use_input_stats _momentum _eps _cudnn_enabled

-- | isclose
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
--
-- >>> dtype &&& shape $ isclose 0.1 0.1 False (ones :: CPUTensor 'D.Float '[3,2]) (ones :: CPUTensor 'D.Float '[3,2])
-- (Bool,[3,2])
isclose
  :: forall shape dtype device
   . Double -- ^ relative tolerance
  -> Double -- ^ absolute tolerance
  -> Bool -- ^ whether or not NaN equals NaN
  -> Tensor device dtype   shape -- ^ input tensor
  -> Tensor device dtype   shape -- ^ other input tensor
  -> Tensor device 'D.Bool shape -- ^ output
isclose rtol atol equalNaN input other =
  unsafePerformIO $ ATen.cast5 ATen.Managed.isclose_ttddb input other rtol atol equalNaN

-- | is NaN
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
--
-- >>> dtype &&& shape $ isNaN (ones :: CPUTensor 'D.Float '[3,2])
-- (Bool,[3,2])
isNaN
  :: forall shape dtype device
   . Tensor device dtype   shape -- ^ input
  -> Tensor device 'D.Bool shape -- ^ output
isNaN input = unsafePerformIO $ ATen.cast1 ATen.Managed.isnan_t input

-- | is distributed
isDistributed
  :: forall shape dtype device
   . Tensor device dtype shape  -- ^ input
  -> Bool -- ^ output
isDistributed input = unsafePerformIO $ ATen.cast1 ATen.Managed.is_distributed_t input

-- | is floating point
-- TODO: this can be decided statically
isFloatingPoint
  :: forall shape dtype device
   . Tensor device dtype shape  -- ^ input
  -> Bool -- ^ output
isFloatingPoint input = unsafePerformIO $ ATen.cast1 ATen.Managed.is_floating_point_t input

-- | is complex
isComplex
  :: forall shape dtype device
   . Tensor device dtype shape  -- ^ input
  -> Bool -- ^ output
isComplex input = unsafePerformIO $ ATen.cast1 ATen.Managed.is_complex_t input

-- | is non-zero
-- this operation is only defined for tensors with shape '[] or '[1]
isNonZero
  :: forall shape dtype device
   . (Numel shape ~ 1)
  => Tensor device dtype shape  -- ^ input
  -> Bool -- ^ output
isNonZero input = unsafePerformIO $ ATen.cast1 ATen.Managed.is_nonzero_t input

-- | is same size
-- TODO: this can be decided statically
isSameSize
  :: forall shape shape' dtype device
   . Tensor device dtype shape -- ^ input tensor
  -> Tensor device dtype shape' -- ^ other input tensor
  -> Bool -- ^ output
isSameSize input other =
  unsafePerformIO $ ATen.cast2 ATen.Managed.is_same_size_tt input other

isSigned
  :: forall shape dtype device
   . Tensor device dtype shape -- ^ input
  -> Bool -- ^ output
isSigned input = unsafePerformIO $ ATen.cast1 ATen.Managed.is_signed_t input

-- kl_div :: Tensor device dtype shape -> Tensor device dtype shape -> Int -> Tensor device dtype shape
-- kl_div _input _target _reduction = unsafePerformIO $ (ATen.cast3 ATen.Managed.kl_div_ttl) _input _target _reduction

-- kthvalue :: Tensor device dtype shape -> Int -> Int -> Bool -> (Tensor device dtype shape,Tensor device dtype shape)
-- kthvalue _input _k _dim _keepdim = unsafePerformIO $ (ATen.cast4 ATen.Managed.kthvalue_tllb) _input _k _dim _keepdim

-- | EndsWith
--
-- >>> :kind! EndsWith '[1] '[1]
-- EndsWith '[1] '[1] :: Constraint
-- = () :: Constraint
-- >>> :kind! EndsWith '[2, 1] '[1]
-- EndsWith '[2, 1] '[1] :: Constraint
-- = () :: Constraint
-- >>> :kind! EndsWith '[2, 1] '[2]
-- EndsWith '[2, 1] '[2] :: Constraint
-- = EndsWith '[1] '[]
-- >>> :kind! EndsWith '[2, 1] '[1, 1]
-- EndsWith '[2, 1] '[1, 1] :: Constraint
-- = EndsWith '[] '[1]
-- >>> :kind! EndsWith '[2, 1] '[2, 1]
-- EndsWith '[2, 1] '[2, 1] :: Constraint
-- = () :: Constraint
type family EndsWith (xs :: [a]) (ys :: [a]) :: Constraint where
  EndsWith '[]      '[]      = ()
  EndsWith (x : xs) (x : ys) = EndsWith xs ys
  EndsWith (x : xs) (y : ys) = EndsWith xs (y : ys)

-- | layerNorm
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
-- TODO: figure out if and when CUDNN works here, tie it also to the `device`
--
-- >>> t = layerNorm @'[1, 2] @'[2, 1, 2] @'D.Float @'( 'D.CPU, 0) ones ones 0.01 ones
-- >>> :type t
-- t :: Tensor '( 'D.CPU, 0) 'D.Float '[2, 1, 2]
-- >>> dtype &&& shape $ t
-- (Float,[2,1,2])
layerNorm
  :: forall normalizedShape shape dtype device
   . ( KnownShape normalizedShape
     , EndsWith shape normalizedShape
     )
  => Tensor device dtype normalizedShape -- ^ weight
  -> Tensor device dtype normalizedShape -- ^ bias
  -> Double -- ^ eps
  -> Tensor device dtype shape -- ^ input tensor
  -> Tensor device dtype shape -- ^ output tensor
layerNorm weight bias eps input = unsafePerformIO $ ATen.cast6
  ATen.Managed.layer_norm_tlttdb
  input
  (shapeVal @normalizedShape)
  weight
  bias
  eps
  (  cudnnIsAcceptable weight
  && cudnnIsAcceptable bias
  && cudnnIsAcceptable input
  )

-- native_layer_norm :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Int -> Int -> Double -> (Tensor device dtype shape,Tensor device dtype shape,Tensor device dtype shape)
-- native_layer_norm _input _weight _bias _M _N _eps = unsafePerformIO $ (ATen.cast6 ATen.Managed.native_layer_norm_tttlld) _input _weight _bias _M _N _eps

-- | linear
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
-- https://pytorch.org/docs/stable/_modules/torch/nn/functional.html#linear
--
-- >>> w = fromJust [[-0.5, -2,  0.5], [1.5, -0.5, 0.5]] :: CPUTensor 'D.Float '[2, 3]
-- >>> b = fromJust [0, 0.5] :: CPUTensor 'D.Float '[2]
-- >>> t = fromJust [[-2, 0.5, 1], [0.5, 0, 0], [0, 1, 0], [0, 0, 0], [1, -1, 0]] :: CPUTensor 'D.Float '[5, 3]
-- >>> t' = linear w b t
-- >>> :type t'
-- t' :: Tensor '( 'D.CPU, 0) 'D.Float '[5, 2]
-- >>> dtype &&& shape &&& (\t'' -> D.asValue (toDynamic t'') :: [[Float]]) $ t'
-- (Float,([5,2],[[0.5,-2.25],[-0.25,1.25],[-2.0,0.0],[0.0,0.5],[1.5,2.5]]))
linear
  :: forall batchSize inputFeatures outputFeatures dtype device
   . Tensor device dtype '[outputFeatures, inputFeatures]
  -> Tensor device dtype '[outputFeatures]
  -> Tensor device dtype '[batchSize, inputFeatures]
  -> Tensor device dtype '[batchSize, outputFeatures]
linear weight bias input = unsafePerformIO $ ATen.cast3 ATen.Managed.linear_ttt input weight bias

-- | linear'
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
-- TODO: can we use the ATen linear function or not here?
-- https://pytorch.org/docs/stable/_modules/torch/nn/functional.html#linear
--
-- >>> w = fromJust [[-0.5, -2,  0.5], [1.5, -0.5, 0.5]] :: CPUTensor 'D.Float '[2, 3]
-- >>> b = fromJust [0, 0.5] :: CPUTensor 'D.Float '[2]
-- >>> t = fromJust [[-2, 0.5, 1], [0.5, 0, 0], [0, 1, 0], [0, 0, 0], [1, -1, 0]] :: CPUTensor 'D.Float '[5, 3]
-- >>> t' = linear' w b t
-- >>> :type t'
-- t' :: Tensor '( 'D.CPU, 0) 'D.Float '[5, 2]
-- >>> dtype &&& shape &&& (\t'' -> D.asValue (toDynamic t'') :: [[Float]]) $ t'
-- (Float,([5,2],[[0.5,-2.25],[-0.25,1.25],[-2.0,0.0],[0.0,0.5],[1.5,2.5]]))
-- >>> t = fromJust [[[[-2, 0.5, 1], [0.5, 0, 0], [0, 1, 0], [0, 0, 0], [1, -1, 0]], [[-2, 0.5, 1], [0.5, 0, 0], [0, 1, 0], [0, 0, 0], [1, -1, 0]]]] :: CPUTensor 'D.Float '[1, 2, 5, 3]
-- >>> t' = linear' w b t
-- >>> :type t'
-- t' :: Tensor '( 'D.CPU, 0) 'D.Float '[1, 2, 5, 2]
-- >>> dtype &&& shape &&& (\t' -> D.asValue (toDynamic t') :: [[[[Float]]]]) $ t'
-- (Float,([1,2,5,2],[[[[0.5,-2.25],[-0.25,1.25],[-2.0,0.0],[0.0,0.5],[1.5,2.5]],[[0.5,-2.25],[-0.25,1.25],[-2.0,0.0],[0.0,0.5],[1.5,2.5]]]]))
linear'
  :: forall (inputFeatures :: Nat) (outputFeatures :: Nat) (shape :: [Nat]) (shape' :: [Nat]) dtype device (shape'' :: [Nat])
   . ( shape'' ~ MatMul shape '[inputFeatures, outputFeatures]
     , shape' ~ Broadcast shape'' shape''
     )
  => Tensor device dtype '[outputFeatures, inputFeatures] -- ^ weight
  -> Tensor device dtype '[outputFeatures] -- ^ bias
  -> Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape' -- ^ output
-- linear' weight bias input = Torch.Static.add (matmul input $ transpose @0 @1 weight) bias
linear' weight bias input = unsafePerformIO $ ATen.cast3 ATen.Managed.linear_ttt input weight bias

-- | mkldnnLinear
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
-- TODO: mkldnnLinear does not return a usuable tensor value and is hence broken
-- TODO: figure out `device` for this
--
-- >>> w = fromJust [[-0.5, -2,  0.5], [1.5, -0.5, 0.5]] :: CPUTensor 'D.Float '[2, 3]
-- >>> b = fromJust [0, 0.5] :: CPUTensor 'D.Float '[2]
-- >>> t = fromJust [[-2, 0.5, 1], [0.5, 0, 0], [0, 1, 0], [0, 0, 0], [1, -1, 0]] :: CPUTensor 'D.Float '[5, 3]
-- 
-- -- >>> t' = mkldnnLinear (toMKLDNN w) (toMKLDNN b) (toMKLDNN t)
-- -- >>> :type t'
-- -- t' :: Tensor '( 'D.CPU, 0) 'D.Float '[5, 2]
-- -- >>> dtype &&& shape &&& (\t'' -> D.asValue (toDynamic t'') :: [[Float]]) $ t'
-- -- (Float,([5,2],[[0.5,-2.25],[-0.25,1.25],[-2.0,0.0],[0.0,0.5],[1.5,2.5]]))
mkldnnLinear
  :: forall batchSize inputFeatures outputFeatures dtype device
   . Tensor device dtype '[outputFeatures, inputFeatures] -- ^ weight
  -> Tensor device dtype '[outputFeatures] -- ^ bias
  -> Tensor device dtype '[batchSize, inputFeatures] -- ^ input
  -> Tensor device dtype '[batchSize, outputFeatures] -- ^ output
mkldnnLinear weight bias input = unsafePerformIO $ ATen.cast3 ATen.Managed.mkldnn_linear_ttt input weight bias

-- fbgemm_linear_int8_weight :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Float -> Float -> Tensor device dtype shape -> Tensor device dtype shape
-- fbgemm_linear_int8_weight _input _weight _packed _col_offsets _weight_scale _weight_zero_point _bias = unsafePerformIO $ (ATen.cast7 ATen.Managed.fbgemm_linear_int8_weight_ttttsst) _input _weight _packed _col_offsets _weight_scale _weight_zero_point _bias

-- fbgemm_linear_quantize_weight :: Tensor device dtype shape -> (Tensor device dtype shape,Tensor device dtype shape,Double,Int)
-- fbgemm_linear_quantize_weight _input = unsafePerformIO $ (ATen.cast1 ATen.Managed.fbgemm_linear_quantize_weight_t) _input

-- fbgemm_pack_gemm_matrix_fp16 :: Tensor device dtype shape -> Tensor device dtype shape
-- fbgemm_pack_gemm_matrix_fp16 _input = unsafePerformIO $ (ATen.cast1 ATen.Managed.fbgemm_pack_gemm_matrix_fp16_t) _input

-- fbgemm_linear_fp16_weight :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape
-- fbgemm_linear_fp16_weight _input _packed_weight _bias = unsafePerformIO $ (ATen.cast3 ATen.Managed.fbgemm_linear_fp16_weight_ttt) _input _packed_weight _bias

-- fbgemm_pack_quantized_matrix :: Tensor device dtype shape -> Int -> Int -> Tensor device dtype shape
-- fbgemm_pack_quantized_matrix _input _K _N = unsafePerformIO $ (ATen.cast3 ATen.Managed.fbgemm_pack_quantized_matrix_tll) _input _K _N

-- fbgemm_is_cpu_supported :: Bool
-- fbgemm_is_cpu_supported  = unsafePerformIO $ (cast0 ATen.Managed.fbgemm_is_cpu_supported) 

-- | log
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
-- TODO: will log throw for negative numbers or just generate NaNs? should we return a Maybe?
--
-- >>> dtype &&& shape $ log (ones :: CPUTensor 'D.Float '[3,2])
-- (Float,[3,2])
log
  :: forall shape dtype device
   . Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
log input = unsafePerformIO $ ATen.cast1 ATen.Managed.log_t input

-- | logDet
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
-- TODO: will logDet throw? and if so, should we return a Maybe?
--
-- >>> dtype &&& shape $ logDet (ones :: CPUTensor 'D.Float '[2,2])
-- (Float,[])
-- >>> dtype &&& shape $ logDet (ones :: CPUTensor 'D.Float '[3,2,2])
-- (Float,[3])
logDet
  :: forall shape' shape dtype device
   . (shape' ~ Det shape)
  => Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape' -- ^ output
logDet input = unsafePerformIO $ ATen.cast1 ATen.Managed.logdet_t input

-- | logarithm of the sum of the exponentials
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
-- See https://pytorch.org/docs/stable/torch.html#torch.logsumexp.
--
-- >>> t = fromJust [[5, 1], [3, 2], [4, 1], [2, 7]] :: CPUTensor 'D.Float '[4, 2]
--
-- >>> dtype &&& shape &&& (\t' -> D.asValue (toDynamic t') :: [Float]) $ logSumExp @1 @DropDim t
-- (Float,([4],[5.01815,3.3132617,4.0485873,7.0067153]))
--
-- >>> dtype &&& shape &&& (\t' -> D.asValue (toDynamic t') :: [[Float]]) $ logSumExp @1 @KeepDim t
-- (Float,([4,1],[[5.01815],[3.3132617],[4.0485873],[7.0067153]]))
--
-- >>> dtype &&& shape &&& (\t' -> D.asValue (toDynamic t') :: [Float]) $ logSumExp @0 @DropDim t
-- (Float,([2],[5.44019,7.0116277]))
--
-- >>> dtype &&& shape &&& (\t' -> D.asValue (toDynamic t') :: [[Float]]) $ logSumExp @0 @KeepDim t
-- (Float,([1,2],[[5.44019,7.0116277]]))
logSumExp
  :: forall dim keepOrDropDim shape' shape dtype device
   . ( KnownNat dim
     , KnownKeepOrDropDim keepOrDropDim
     , Reifies dtype D.DType
     , DTypeIsFloatingPoint device dtype
     , shape' ~ ConditionalDropDimension shape dim keepOrDropDim
     )
  => Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape' -- ^ output
logSumExp input = unsafePerformIO $ ATen.cast3 ATen.Managed.logsumexp_tlb
                                          input
                                          (natValI @dim)
                                          (keepOrDropDimVal @keepOrDropDim)

-- margin_ranking_loss :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Double -> Int -> Tensor device dtype shape
-- margin_ranking_loss _input1 _input2 _target _margin _reduction = unsafePerformIO $ (ATen.cast5 ATen.Managed.margin_ranking_loss_tttdl) _input1 _input2 _target _margin _reduction

-- | matrixPower
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
-- TODO: figure out input shape restrictions, should be matrix or a batched matrix
-- TODO: figure out restrictions on the power, can it be zero or negative?
--
-- >>> dtype &&& shape $ matrixPower 2 (ones :: CPUTensor 'D.Float '[3,4,4])
-- (Float,[3,4,4])
matrixPower
  :: forall shape' shape dtype device
   . (shape' ~ Square shape)
  => Int -- ^ power
  -> Tensor device dtype shape -- ^ input matrix
  -> Tensor device dtype shape' -- ^ output
matrixPower n input = unsafePerformIO $ ATen.cast2 ATen.Managed.matrix_power_tl input n

-- | maxValues
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
--
-- >>> t = ones :: CPUTensor 'D.Float '[3,4,5]
-- >>> dtype &&& shape $ maxValues @0 @KeepDim t
-- (Float,[1,4,5])
-- >>> dtype &&& shape $ maxValues @0 @DropDim t
-- (Float,[4,5])
-- >>> dtype &&& shape $ maxValues @1 @KeepDim t
-- (Float,[3,1,5])
-- >>> dtype &&& shape $ maxValues @1 @DropDim t
-- (Float,[3,5])
maxValues
  :: forall dim keepOrDropDim shape' shape dtype device
  . ( KnownNat dim
    , KnownKeepOrDropDim keepOrDropDim
    , shape' ~ ConditionalDropDimension shape dim keepOrDropDim
    )
  => Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape' -- ^ output
maxValues input = unsafePerformIO $ ATen.cast3 ATen.Managed.max_values_tlb
                                          input
                                          (natValI @dim)
                                          (keepOrDropDimVal @keepOrDropDim)

-- | minValues
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
--
-- >>> t = ones :: CPUTensor 'D.Float '[3,4,5]
-- >>> dtype &&& shape $ minValues @0 @KeepDim t
-- (Float,[1,4,5])
-- >>> dtype &&& shape $ minValues @0 @DropDim t
-- (Float,[4,5])
-- >>> dtype &&& shape $ minValues @1 @KeepDim t
-- (Float,[3,1,5])
-- >>> dtype &&& shape $ minValues @1 @DropDim t
-- (Float,[3,5])
minValues
  :: forall dim keepOrDropDim shape' shape dtype device
   . ( KnownNat dim
     , KnownKeepOrDropDim keepOrDropDim
     , shape' ~ ConditionalDropDimension shape dim keepOrDropDim
     )
  => Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape' -- ^ output
minValues input = unsafePerformIO $ ATen.cast3 ATen.Managed.min_values_tlb
                                          input
                                          (natValI @dim)
                                          (keepOrDropDimVal @keepOrDropDim)

-- | maxPool1d
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
--
-- >>> t = maxPool1d @1 @1 @0 (ones :: CPUTensor 'D.Float '[1,3,4])
-- >>> shape t
-- [1,3,4]
-- >>> :t t
-- t :: Tensor '( 'D.CPU, 0) 'D.Float '[1, 3, 4]
maxPool1d
  :: forall kernelSize stride padding channelSize inputSize batchSize outputSize dtype device
   . ( All KnownNat '[ kernelSize
                     , stride
                     , padding
                     , channelSize
                     , inputSize
                     , batchSize
                     ]  
     , ConvSideCheck inputSize kernelSize stride padding outputSize
     )
  => Tensor device dtype '[batchSize, channelSize, inputSize] -- ^ input
  -> Tensor device dtype '[batchSize, channelSize, outputSize] -- ^ output
maxPool1d input = unsafePerformIO $ ATen.cast6 ATen.Managed.max_pool1d_tllllb
                                          input
                                          (natValI @kernelSize)
                                          (natValI @stride)
                                          (natValI @padding)
                                          (1 :: Int)
                                          False

-- max_pool1d_with_indices :: Tensor device dtype shape -> Int -> Int -> Int -> Int -> Bool -> (Tensor device dtype shape,Tensor device dtype shape)
-- max_pool1d_with_indices _input _kernel_size _stride _padding _dilation _ceil_mode = unsafePerformIO $ (ATen.cast6 ATen.Managed.max_pool1d_with_indices_tllllb) _input _kernel_size _stride _padding _dilation _ceil_mode

-- | maxPool2d
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
--
-- >>> t = maxPool2d @'(1,1) @'(1,1) @'(0,0) (ones :: CPUTensor 'D.Float '[1,3,4,5])
-- >>> shape t
-- [1,3,4,5]
-- >>> :t t
-- t :: Tensor '( 'D.CPU, 0) 'D.Float '[1, 3, 4, 5]
maxPool2d
  :: forall kernelSize stride padding channelSize inputSize0 inputSize1 batchSize outputSize0 outputSize1 dtype device
   . ( All KnownNat '[ Torch.Typed.Aux.Fst kernelSize, Torch.Typed.Aux.Snd kernelSize
                     , Torch.Typed.Aux.Fst stride, Torch.Typed.Aux.Snd stride
                     , Torch.Typed.Aux.Fst padding, Torch.Typed.Aux.Snd padding
                     , channelSize
                     , inputSize0, inputSize1
                     , batchSize
                     ]
     , ConvSideCheck inputSize0 (Torch.Typed.Aux.Fst kernelSize) (Torch.Typed.Aux.Fst stride) (Torch.Typed.Aux.Fst padding) outputSize0
     , ConvSideCheck inputSize1 (Torch.Typed.Aux.Snd kernelSize) (Torch.Typed.Aux.Snd stride) (Torch.Typed.Aux.Snd padding) outputSize1
     )
  => Tensor device dtype '[batchSize, channelSize, inputSize0, inputSize1] -- ^ input
  -> Tensor device dtype '[batchSize, channelSize, outputSize0, outputSize1] -- ^ output
maxPool2d input = unsafePerformIO $ ATen.cast6
  ATen.Managed.max_pool2d_tllllb
  input
  ([natValI @(Torch.Typed.Aux.Fst kernelSize), natValI @(Torch.Typed.Aux.Snd kernelSize)] :: [Int])
  ([natValI @(Torch.Typed.Aux.Fst stride),     natValI @(Torch.Typed.Aux.Snd stride)]     :: [Int])
  ([natValI @(Torch.Typed.Aux.Fst padding),    natValI @(Torch.Typed.Aux.Snd padding)]    :: [Int])
  ([1, 1] :: [Int])
  False

-- | mkldnnMaxPool2d
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
-- TODO: does this function work, that is, does it return values without throwing? when does it work?
-- TODO: this should probably be only callable if the device is MKLDNN?
-- -- >>> t = mkldnnMaxPool2d @'(1,1) @'(1,1) @'(0,0) (toMKLDNN (ones :: CPUTensor 'D.Float '[1,3,4,5]))
-- -- >>> shape t
-- -- [1,3,4,5]
-- -- >>> :t t
-- -- t :: Tensor '( 'D.CPU, 0) 'D.Float '[1, 3, 4, 5]
mkldnnMaxPool2d
  :: forall kernelSize stride padding channelSize inputSize0 inputSize1 batchSize outputSize0 outputSize1 dtype device
   . ( All KnownNat '[ Torch.Typed.Aux.Fst kernelSize, Torch.Typed.Aux.Snd kernelSize
                     , Torch.Typed.Aux.Fst stride, Torch.Typed.Aux.Snd stride
                     , Torch.Typed.Aux.Fst padding, Torch.Typed.Aux.Snd padding
                     , channelSize
                     , inputSize0, inputSize1
                     , batchSize
                     ]
     , ConvSideCheck inputSize0 (Torch.Typed.Aux.Fst kernelSize) (Torch.Typed.Aux.Fst stride) (Torch.Typed.Aux.Fst padding) outputSize0
     , ConvSideCheck inputSize1 (Torch.Typed.Aux.Snd kernelSize) (Torch.Typed.Aux.Snd stride) (Torch.Typed.Aux.Snd padding) outputSize1
     )
  => Tensor device dtype '[batchSize, channelSize, inputSize0, inputSize1] -- ^ input
  -> Tensor device dtype '[batchSize, channelSize, outputSize0, outputSize1] -- ^ output
mkldnnMaxPool2d input = unsafePerformIO $ ATen.cast6
  ATen.Managed.mkldnn_max_pool2d_tllllb
  input
  ([natValI @(Torch.Typed.Aux.Fst kernelSize), natValI @(Torch.Typed.Aux.Snd kernelSize)] :: [Int])
  ([natValI @(Torch.Typed.Aux.Fst stride),     natValI @(Torch.Typed.Aux.Snd stride)]     :: [Int])
  ([natValI @(Torch.Typed.Aux.Fst padding),    natValI @(Torch.Typed.Aux.Snd padding)]    :: [Int])
  ([1, 1] :: [Int])
  False

-- | quantizedMaxPool2d
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
-- TODO: what are quantized functions and when are they available?
-- -- >>> t = quantizedMaxPool2d @'(1,1) @'(1,1) @'(0,0) (ones :: CPUTensor 'D.Float '[1,3,4,5])
-- -- >>> shape t
-- -- [1,3,4,5]
-- -- >>> :t t
-- -- t :: Tensor 'D.Float '[1, 3, 4, 5]
quantizedMaxPool2d
  :: forall kernelSize stride padding channelSize inputSize0 inputSize1 batchSize outputSize0 outputSize1 dtype device
   . ( All KnownNat '[ Torch.Typed.Aux.Fst kernelSize, Torch.Typed.Aux.Snd kernelSize
                     , Torch.Typed.Aux.Fst stride, Torch.Typed.Aux.Snd stride
                     , Torch.Typed.Aux.Fst padding, Torch.Typed.Aux.Snd padding
                     , channelSize
                     , inputSize0, inputSize1
                     , batchSize
                     ]
     , ConvSideCheck inputSize0 (Torch.Typed.Aux.Fst kernelSize) (Torch.Typed.Aux.Fst stride) (Torch.Typed.Aux.Fst padding) outputSize0
     , ConvSideCheck inputSize1 (Torch.Typed.Aux.Snd kernelSize) (Torch.Typed.Aux.Snd stride) (Torch.Typed.Aux.Snd padding) outputSize1
     )
  => Tensor device dtype '[batchSize, channelSize, inputSize0, inputSize1] -- ^ input
  -> Tensor device dtype '[batchSize, channelSize, outputSize0, outputSize1] -- ^ output
quantizedMaxPool2d input = unsafePerformIO $ ATen.cast5
  ATen.Managed.quantized_max_pool2d_tllll
  input
  ([natValI @(Torch.Typed.Aux.Fst kernelSize), natValI @(Torch.Typed.Aux.Snd kernelSize)] :: [Int])
  ([natValI @(Torch.Typed.Aux.Fst stride),     natValI @(Torch.Typed.Aux.Snd stride)]     :: [Int])
  ([natValI @(Torch.Typed.Aux.Fst padding),    natValI @(Torch.Typed.Aux.Snd padding)]    :: [Int])
  ([1, 1] :: [Int])

-- | maxPool3d
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
--
-- >>> t = maxPool3d @'(1,1,1) @'(1,1,1) @'(0,0,0) (ones :: CPUTensor 'D.Float '[1,3,4,5,6])
-- >>> shape t
-- [1,3,4,5,6]
-- >>> :t t
-- t :: Tensor '( 'D.CPU, 0) 'D.Float '[1, 3, 4, 5, 6]
maxPool3d
  :: forall kernelSize stride padding channelSize
            inputSize0 inputSize1 inputSize2
            batchSize
            outputSize0 outputSize1 outputSize2
            dtype
            device
   . ( All KnownNat '[ Fst3 kernelSize, Snd3 kernelSize, Trd3 kernelSize
                     , Fst3 stride, Snd3 stride, Trd3 stride
                     , Fst3 padding, Snd3 padding, Trd3 padding
                     , channelSize
                     , inputSize0, inputSize1, inputSize2
                     , batchSize
                     ]
     , ConvSideCheck inputSize0 (Fst3 kernelSize) (Fst3 stride) (Fst3 padding) outputSize0
     , ConvSideCheck inputSize1 (Snd3 kernelSize) (Snd3 stride) (Snd3 padding) outputSize1
     , ConvSideCheck inputSize2 (Trd3 kernelSize) (Trd3 stride) (Trd3 padding) outputSize2
     )
  => Tensor device dtype '[batchSize, channelSize, inputSize0, inputSize1, inputSize2] -- ^ input
  -> Tensor device dtype '[batchSize, channelSize, outputSize0, outputSize1, outputSize2] -- ^ output
maxPool3d input = unsafePerformIO $ ATen.cast6
  ATen.Managed.max_pool3d_tllllb
  input
  ([ natValI @(Fst3 kernelSize)
   , natValI @(Snd3 kernelSize)
   , natValI @(Trd3 kernelSize)
   ] :: [Int]
  )
  ([natValI @(Fst3 stride), natValI @(Snd3 stride), natValI @(Trd3 stride)] :: [Int])
  ([natValI @(Fst3 padding), natValI @(Snd3 padding), natValI @(Trd3 padding)] :: [Int])
  ([1, 1, 1] :: [Int])
  False

-- | maskedFill
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
--
-- >>> t = ones :: CPUTensor 'D.Float '[2, 1, 3]
-- >>> m = fromJust [[False], [True], [False]] :: CPUTensor 'D.Bool '[3, 1]
-- >>> t' = maskedFill @Float m 0.5 t
-- >>> :type t'
-- t' :: Tensor '( 'D.CPU, 0) 'D.Float '[2, 3, 3]
-- >>> dtype &&& shape &&& (\u -> D.asValue (toDynamic u) :: [[[Float]]]) $ t'
-- (Float,([2,3,3],[[[1.0,1.0,1.0],[0.5,0.5,0.5],[1.0,1.0,1.0]],[[1.0,1.0,1.0],[0.5,0.5,0.5],[1.0,1.0,1.0]]]))
maskedFill
  :: forall a shape shape' shape'' dtype device
   . (D.Scalar a, shape'' ~ Broadcast shape shape')
  => Tensor device 'D.Bool shape' -- ^ mask
  -> a -- ^ fill value
  -> Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape'' -- ^ output
maskedFill mask value input =
  unsafePerformIO $ ATen.cast3 ATen.Managed.masked_fill_tts input mask value

-- mkldnn_convolution :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> [Int] -> [Int] -> [Int] -> Int -> Tensor device dtype shape
-- mkldnn_convolution _input _weight _bias _padding _stride _dilation _groups = unsafePerformIO $ (ATen.cast7 ATen.Managed.mkldnn_convolution_tttllll) _input _weight _bias _padding _stride _dilation _groups

-- miopen_batch_norm :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Bool -> Double -> Double -> (Tensor device dtype shape,Tensor device dtype shape,Tensor device dtype shape)
-- miopen_batch_norm _input _weight _bias _running_mean _running_var _training _exponential_average_factor _epsilon = unsafePerformIO $ (ATen.cast8 ATen.Managed.miopen_batch_norm_tttttbdd) _input _weight _bias _running_mean _running_var _training _exponential_average_factor _epsilon

-- miopen_convolution :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> [Int] -> [Int] -> [Int] -> Int -> Bool -> Bool -> Tensor device dtype shape
-- miopen_convolution _input _weight _bias _padding _stride _dilation _groups _benchmark _deterministic = unsafePerformIO $ (ATen.cast9 ATen.Managed.miopen_convolution_tttllllbb) _input _weight _bias _padding _stride _dilation _groups _benchmark _deterministic

-- miopen_convolution_transpose :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> [Int] -> [Int] -> [Int] -> [Int] -> Int -> Bool -> Bool -> Tensor device dtype shape
-- miopen_convolution_transpose _input _weight _bias _padding _output_padding _stride _dilation _groups _benchmark _deterministic = unsafePerformIO $ (ATen.ATen.cast10 ATen.Managed.miopen_convolution_transpose_tttlllllbb) _input _weight _bias _padding _output_padding _stride _dilation _groups _benchmark _deterministic

-- miopen_depthwise_convolution :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> [Int] -> [Int] -> [Int] -> Int -> Bool -> Bool -> Tensor device dtype shape
-- miopen_depthwise_convolution _input _weight _bias _padding _stride _dilation _groups _benchmark _deterministic = unsafePerformIO $ (ATen.cast9 ATen.Managed.miopen_depthwise_convolution_tttllllbb) _input _weight _bias _padding _stride _dilation _groups _benchmark _deterministic

-- miopen_rnn :: Tensor device dtype shape -> [Tensor device dtype shape] -> Int -> Tensor device dtype shape -> Tensor device dtype shape -> Int -> Int -> Int -> Bool -> Double -> Bool -> Bool -> [Int] -> Tensor device dtype shape -> (Tensor device dtype shape,Tensor device dtype shape,Tensor device dtype shape,Tensor device dtype shape,Tensor device dtype shape)
-- miopen_rnn _input _weight _weight_stride0 _hx _cx _mode _hidden_size _num_layers _batch_first _dropout _train _bidirectional _batch_sizes _dropout_state = unsafePerformIO $ (ATen.ATen.cast14 ATen.Managed.miopen_rnn_tllttlllbdbblt) _input _weight _weight_stride0 _hx _cx _mode _hidden_size _num_layers _batch_first _dropout _train _bidirectional _batch_sizes _dropout_state

-- | matrix-matrix multiplication
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
--
-- >>> dtype &&& shape $ mm (ones :: CPUTensor 'D.Float '[3,2]) (zeros :: CPUTensor 'D.Float '[2,4])
-- (Float,[3,4])
mm
  :: forall n k m dtype device
   . Tensor device dtype '[n, k] -- ^ first input matrix
  -> Tensor device dtype '[k, m] -- ^ second input matrix
  -> Tensor device dtype '[n, m] -- ^ output matrix
mm a b = unsafePerformIO $ ATen.cast2 ATen.Managed.mm_tt a b

-- mode :: Tensor device dtype shape -> Int -> Bool -> (Tensor device dtype shape,Tensor device dtype shape)
-- mode _input _dim _keepdim = unsafePerformIO $ (ATen.cast3 ATen.Managed.mode_tlb) _input _dim _keepdim

-- | matrix-vector multiplication
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
--
-- >>> dtype &&& shape $ mv (ones :: CPUTensor 'D.Float '[3,2]) (zeros :: CPUTensor 'D.Float '[2])
-- (Float,[3])
mv
  :: forall n m dtype device
   . Tensor device dtype '[n, m] -- ^ input matrix
  -> Tensor device dtype '[m] -- ^ input vector
  -> Tensor device dtype '[n] -- ^ output vector
mv input vec = unsafePerformIO $ ATen.cast2 ATen.Managed.mv_tt input vec

-- mvlgamma :: Tensor device dtype shape -> Int -> Tensor device dtype shape
-- mvlgamma _input _p = unsafePerformIO $ (ATen.cast2 ATen.Managed.mvlgamma_tl) _input _p

-- narrow :: Tensor device dtype shape -> Int -> Int -> Int -> Tensor device dtype shape
-- narrow _input _dim _start _length = unsafePerformIO $ (ATen.cast4 ATen.Managed.narrow_tlll) _input _dim _start _length

-- native_batch_norm :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Bool -> Double -> Double -> (Tensor device dtype shape,Tensor device dtype shape,Tensor device dtype shape)
-- native_batch_norm _input _weight _bias _running_mean _running_var _training _momentum _eps = unsafePerformIO $ (ATen.cast8 ATen.Managed.native_batch_norm_tttttbdd) _input _weight _bias _running_mean _running_var _training _momentum _eps

-- batch_norm_stats :: Tensor device dtype shape -> Double -> (Tensor device dtype shape,Tensor device dtype shape)
-- batch_norm_stats _input _eps = unsafePerformIO $ (ATen.cast2 ATen.Managed.batch_norm_stats_td) _input _eps

-- batch_norm_elemt :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Double -> Tensor device dtype shape
-- batch_norm_elemt _input _weight _bias _mean _invstd _eps = unsafePerformIO $ (ATen.cast6 ATen.Managed.batch_norm_elemt_tttttd) _input _weight _bias _mean _invstd _eps

-- batch_norm_gather_stats :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Double -> Double -> Int -> (Tensor device dtype shape,Tensor device dtype shape)
-- batch_norm_gather_stats _input _mean _invstd _running_mean _running_var _momentum _eps _count = unsafePerformIO $ (ATen.cast8 ATen.Managed.batch_norm_gather_stats_tttttddl) _input _mean _invstd _running_mean _running_var _momentum _eps _count

-- batch_norm_gather_stats_with_counts :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Double -> Double -> [Int] -> (Tensor device dtype shape,Tensor device dtype shape)
-- batch_norm_gather_stats_with_counts _input _mean _invstd _running_mean _running_var _momentum _eps _counts = unsafePerformIO $ (ATen.cast8 ATen.Managed.batch_norm_gather_stats_with_counts_tttttddl) _input _mean _invstd _running_mean _running_var _momentum _eps _counts

-- batch_norm_update_stats :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Double -> (Tensor device dtype shape,Tensor device dtype shape)
-- batch_norm_update_stats _input _running_mean _running_var _momentum = unsafePerformIO $ (ATen.cast4 ATen.Managed.batch_norm_update_stats_tttd) _input _running_mean _running_var _momentum

-- | onesLike
--
-- >>> dtype &&& shape $ onesLike (ones :: CPUTensor 'D.Float '[3,4,5])
-- (Float,[3,4,5])
onesLike
  :: forall shape dtype device
   . Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
onesLike input = unsafePerformIO $ ATen.cast1 ATen.Managed.ones_like_t input

-- pairwise_distance :: Tensor device dtype shape -> Tensor device dtype shape -> Double -> Double -> Bool -> Tensor device dtype shape
-- pairwise_distance _x1 _x2 _p _eps _keepdim = unsafePerformIO $ (ATen.cast5 ATen.Managed.pairwise_distance_ttddb) _x1 _x2 _p _eps _keepdim

-- cdist :: Tensor device dtype shape -> Tensor device dtype shape -> Double -> Tensor device dtype shape
-- cdist _x1 _x2 _p = unsafePerformIO $ (ATen.cast3 ATen.Managed.cdist_ttd) _x1 _x2 _p

-- pdist :: Tensor device dtype shape -> Double -> Tensor device dtype shape
-- pdist _input _p = unsafePerformIO $ (ATen.cast2 ATen.Managed.pdist_td) _input _p

-- cosine_similarity :: Tensor device dtype shape -> Tensor device dtype shape -> Int -> Double -> Tensor device dtype shape
-- cosine_similarity _x1 _x2 _dim _eps = unsafePerformIO $ (ATen.cast4 ATen.Managed.cosine_similarity_ttld) _x1 _x2 _dim _eps

-- pixel_shuffle :: Tensor device dtype shape -> Int -> Tensor device dtype shape
-- pixel_shuffle _input _upscale_factor = unsafePerformIO $ (ATen.cast2 ATen.Managed.pixel_shuffle_tl) _input _upscale_factor

-- pin_memory :: Tensor device dtype shape -> Tensor device dtype shape
-- pin_memory _input = unsafePerformIO $ (ATen.cast1 ATen.Managed.pin_memory_t) _input

-- pinverse :: Tensor device dtype shape -> Double -> Tensor device dtype shape
-- pinverse _input _rcond = unsafePerformIO $ (ATen.cast2 ATen.Managed.pinverse_td) _input _rcond

-- poisson_nll_loss :: Tensor device dtype shape -> Tensor device dtype shape -> Bool -> Bool -> Double -> Int -> Tensor device dtype shape
-- poisson_nll_loss _input _target _log_input _full _eps _reduction = unsafePerformIO $ (ATen.cast6 ATen.Managed.poisson_nll_loss_ttbbdl) _input _target _log_input _full _eps _reduction

-- | randLike
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
--
-- >>> t <- randLike (ones :: CPUTensor 'D.Float '[3,4,5])
-- >>> dtype &&& shape $ t
-- (Float,[3,4,5])
randLike
  :: forall shape dtype device
   . Tensor device dtype shape -- ^ input
  -> IO (Tensor device dtype shape) -- ^ output
randLike = ATen.cast1 ATen.Managed.rand_like_t

-- | randnLike
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
--
-- >>> t <- randnLike (ones :: CPUTensor 'D.Float '[3,4,5])
-- >>> dtype &&& shape $ t
-- (Float,[3,4,5])
randnLike
  :: forall shape dtype device
   . Tensor device dtype shape -- ^ input
  -> IO (Tensor device dtype shape) -- ^ output
randnLike = ATen.cast1 ATen.Managed.randn_like_t

-- | reciprocal
-- 
-- >>> dtype &&& shape $ reciprocal (ones :: CPUTensor 'D.Float '[3,2])
-- (Float,[3,2])
reciprocal 
  :: forall shape dtype device
   . Tensor device dtype shape -- ^ input 
  -> Tensor device dtype shape -- ^ output
reciprocal _input = unsafePerformIO $ (ATen.cast1 ATen.Managed.reciprocal_t) _input

-- | negate
-- TODO: probably not defined for `D.Bool` tensors
--
-- >>> dtype &&& shape $ neg (ones :: CPUTensor 'D.Float '[3,2])
-- (Float,[3,2])
neg
  :: forall shape dtype device
   . Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
neg input = unsafePerformIO $ ATen.cast1 ATen.Managed.neg_t input

-- | round
-- TODO: probably only defined for floating point tensors
--
-- >>> dtype &&& shape $ round (ones :: CPUTensor 'D.Float '[3,2])
-- (Float,[3,2])
round
  :: forall shape dtype device
   . Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
round input = unsafePerformIO $ ATen.cast1 ATen.Managed.round_t input

-- | prelu activation function
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
--
-- >>> dtype &&& shape $ prelu (ones :: CPUTensor 'D.Float '[]) (ones :: CPUTensor 'D.Float '[3,2])
-- (Float,[3,2])
prelu
  :: forall shape dtype device
   . Tensor device dtype '[] -- ^ weight
  -> Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
prelu weight input = unsafePerformIO $ ATen.cast2 ATen.Managed.prelu_tt input weight

type family GeluDTypeIsValid (device :: (D.DeviceType, Nat)) (dtype :: D.DType) :: Constraint where
  GeluDTypeIsValid '( 'D.CPU, 0)            dtype = ( DTypeIsFloatingPoint '( 'D.CPU, 0) dtype
                                                    , DTypeIsNotHalf '( 'D.CPU, 0) dtype
                                                    )
  GeluDTypeIsValid '( 'D.CUDA, deviceIndex) dtype = ( DTypeIsFloatingPoint '( 'D.CUDA, deviceIndex) dtype
                                                    , DTypeIsNotHalf '( 'D.CUDA, deviceIndex) dtype
                                                    )
  GeluDTypeIsValid '(deviceType, _)         dtype = UnsupportedDTypeForDevice deviceType dtype

-- | gelu activation function
--
-- >>> dtype &&& shape $ round (ones :: CPUTensor 'D.Float '[3,2])
-- (Float,[3,2])
gelu
  :: forall shape dtype device
   . (GeluDTypeIsValid device dtype)
  => Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
gelu input = unsafePerformIO $ ATen.cast1 ATen.Managed.gelu_t input

-- hardshrink :: Tensor device dtype shape -> Float -> Tensor device dtype shape
-- hardshrink _input _lambd = unsafePerformIO $ (ATen.cast2 ATen.Managed.hardshrink_ts) _input _lambd

-- | rsqrt
--
-- >>> dtype &&& shape $ rsqrt (ones :: CPUTensor 'D.Float '[3,2])
-- (Float,[3,2])
rsqrt
  :: forall shape dtype device
   . (StandardFloatingPointDTypeValidation device dtype)
  => Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
rsqrt input = unsafePerformIO $ ATen.cast1 ATen.Managed.rsqrt_t input

-- | celu activation function
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
--
-- >>> dtype &&& shape $ celu 3.0 (ones :: CPUTensor 'D.Float '[3,2])
-- (Float,[3,2])
celu
  :: forall shape dtype device
   . Float -- ^ alpha
  -> Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
celu alpha input = unsafePerformIO $ ATen.cast2 ATen.Managed.celu_ts input alpha

-- slice :: Tensor device dtype shape -> Int -> Int -> Int -> Int -> Tensor device dtype shape
-- slice _input _dim _start _end _step = unsafePerformIO $ (ATen.cast5 ATen.Managed.slice_tllll) _input _dim _start _end _step

-- slogdet :: Tensor device dtype shape -> (Tensor device dtype shape,Tensor device dtype shape)
-- slogdet _input = unsafePerformIO $ (ATen.cast1 ATen.Managed.slogdet_t) _input

-- smm :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape
-- smm _input _mat2 = unsafePerformIO $ (ATen.cast2 ATen.Managed.smm_tt) _input _mat2

-- split :: Tensor device dtype shape -> Int -> Int -> [Tensor device dtype shape]
-- split _input _split_size _dim = unsafePerformIO $ (ATen.cast3 ATen.Managed.split_tll) _input _split_size _dim

-- split_with_sizes :: Tensor device dtype shape -> [Int] -> Int -> [Tensor device dtype shape]
-- split_with_sizes _input _split_sizes _dim = unsafePerformIO $ (ATen.cast3 ATen.Managed.split_with_sizes_tll) _input _split_sizes _dim

-- sspaddmm :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Float -> Float -> Tensor device dtype shape
-- sspaddmm _input _mat1 _mat2 _beta _alpha = unsafePerformIO $ (ATen.cast5 ATen.Managed.sspaddmm_tttss) _input _mat1 _mat2 _beta _alpha

type family StackImpl (dim :: Nat) (tensors :: [a]) (count :: Nat) :: Maybe ([Nat], D.DType, (D.DeviceType, Nat)) where
  StackImpl dim '[]                                                                 count = Nothing
  StackImpl dim (Tensor device dtype shape ': '[])                                  count = MaybeTriple (ComputeStackShape shape dim count) (Just dtype) (Just device)
  StackImpl dim (Tensor device dtype shape ': Tensor device dtype shape ': tensors) count = StackImpl dim (Tensor device dtype shape ': tensors) (count + 1)
  StackImpl _   _                                                                   _     = Nothing

type family MaybePair (a' :: Maybe a) (b' ::  Maybe b) :: Maybe (a, b) where
  MaybePair Nothing   _         = Nothing
  MaybePair _         Nothing   = Nothing
  MaybePair (Just a') (Just b') = Just '(a', b')

type family MaybeTriple (a' :: Maybe a) (b' ::  Maybe b) (c' ::  Maybe c) :: Maybe (a, b, c) where
  MaybeTriple Nothing   _         _         = Nothing
  MaybeTriple _         Nothing   _         = Nothing
  MaybeTriple _         _         Nothing   = Nothing
  MaybeTriple (Just a') (Just b') (Just c') = Just '(a', b', c')

type family ComputeStackShape (shape :: [Nat]) (dim  :: Nat) (count :: Nat) :: Maybe [Nat] where
  ComputeStackShape _         _   0     = Nothing
  ComputeStackShape xs        0   count = Just (count ': xs)
  ComputeStackShape (x ': xs) dim count = AppendToMaybe x (ComputeStackShape xs (dim - 1) count)
  ComputeStackShape '[]       _   _     = Nothing

type family StackCheck (res :: Maybe ([Nat], D.DType, (D.DeviceType, Nat))) :: ([Nat], D.DType, (D.DeviceType, Nat)) where
  StackCheck 'Nothing                        = TypeError (Text "Stacking impossible.")
  StackCheck ('Just '(shape, dtype, device)) = '(shape, dtype, device)

-- | Stack
--
-- >>> type Ty = Stack 0 '[Tensor '( 'D.CPU, 0) 'D.Float '[]]
-- >>> :kind! Ty
-- Ty :: ([Nat], D.DType, (D.DeviceType, Nat))
-- = '( '[1], 'D.Float, '( 'D.CPU, 0))
-- >>> type Ty = Stack 0 '[Tensor '( 'D.CPU, 0) 'D.Float '[2,2]]
-- >>> :kind! Ty
-- Ty :: ([Nat], D.DType, (D.DeviceType, Nat))
-- = '( '[1, 2, 2], 'D.Float, '( 'D.CPU, 0))
-- >>> type Ty = Stack 1 '[Tensor '( 'D.CPU, 0) 'D.Float '[2,2]]
-- >>> :kind! Ty
-- Ty :: ([Nat], D.DType, (D.DeviceType, Nat))
-- = '( '[2, 1, 2], 'D.Float, '( 'D.CPU, 0))
-- >>> type Ty = Stack 2 '[Tensor '( 'D.CPU, 0) 'D.Float '[2,2]]
-- >>> :kind! Ty
-- Ty :: ([Nat], D.DType, (D.DeviceType, Nat))
-- = '( '[2, 2, 1], 'D.Float, '( 'D.CPU, 0))
-- >>> type Ty = Stack 2 '[Tensor '( 'D.CPU, 0) 'D.Float '[2,2], Tensor '( 'D.CPU, 0) 'D.Float '[2,2], Tensor '( 'D.CPU, 0) 'D.Float '[2,2]]
-- >>> :kind! Ty
-- Ty :: ([Nat], D.DType, (D.DeviceType, Nat))
-- = '( '[2, 2, 3], 'D.Float, '( 'D.CPU, 0))
type Stack dim tensors = StackCheck (StackImpl dim tensors 1)

-- | stack
--
-- >>> t = ones :: CPUTensor 'D.Float '[]
-- >>> t' = stack @0 (t :. HNil)
-- >>> :type t'
-- t' :: Tensor '( 'D.CPU, 0) 'D.Float '[1]
-- >>> dtype &&& shape &&& (\t'' -> D.asValue (toDynamic t'') :: [Float]) $ t'
-- (Float,([1],[1.0]))
-- >>> t = ones :: CPUTensor 'D.Float '[2,2]
-- >>> t' = stack @0 (t :. HNil)
-- >>> :type t'
-- t' :: Tensor '( 'D.CPU, 0) 'D.Float '[1, 2, 2]
-- >>> dtype &&& shape &&& (\t'' -> D.asValue (toDynamic t'') :: [[[Float]]]) $ t'
-- (Float,([1,2,2],[[[1.0,1.0],[1.0,1.0]]]))
-- >>> t' = stack @1 (t :. HNil)
-- >>> :type t'
-- t' :: Tensor '( 'D.CPU, 0) 'D.Float '[2, 1, 2]
-- >>> dtype &&& shape &&& (\t'' -> D.asValue (toDynamic t'') :: [[[Float]]]) $ t'
-- (Float,([2,1,2],[[[1.0,1.0]],[[1.0,1.0]]]))
-- >>> t' = stack @2 (t :. HNil)
-- >>> :type t'
-- t' :: Tensor '( 'D.CPU, 0) 'D.Float '[2, 2, 1]
-- >>> dtype &&& shape &&& (\t'' -> D.asValue (toDynamic t'') :: [[[Float]]]) $ t'
-- (Float,([2,2,1],[[[1.0],[1.0]],[[1.0],[1.0]]]))
-- >>> t' = stack @2 (t :. t :. t :. HNil)
-- >>> :type t'
-- t' :: Tensor '( 'D.CPU, 0) 'D.Float '[2, 2, 3]
-- >>> dtype &&& shape &&& (\t'' -> D.asValue (toDynamic t'') :: [[[Float]]]) $ t'
-- (Float,([2,2,3],[[[1.0,1.0,1.0],[1.0,1.0,1.0]],[[1.0,1.0,1.0],[1.0,1.0,1.0]]]))
stack
  :: forall dim shape dtype device tensors
   . ( KnownNat dim
     , '(shape, dtype, device) ~ Stack dim tensors
     , ATen.Castable (HList tensors) [D.ATenTensor]
     )
  => HList tensors -- ^ input list of tensors
  -> Tensor device dtype shape -- ^ output
stack tensors = unsafePerformIO $ ATen.cast2 ATen.Managed.stack_ll tensors (natValI @dim :: Int)

-- stft :: Tensor device dtype shape -> Int -> Int -> Int -> Tensor device dtype shape -> Bool -> Bool -> Tensor device dtype shape
-- stft _input _n_fft _hop_length _win_length _window _normalized _onesided = unsafePerformIO $ (ATen.cast7 ATen.Managed.stft_tllltbb) _input _n_fft _hop_length _win_length _window _normalized _onesided

-- stride :: Tensor device dtype shape -> Int -> Int
-- stride _input _dim = unsafePerformIO $ (ATen.cast2 ATen.Managed.stride_tl) _input _dim

-- | t
-- 
-- dtype &&& shape $ t ones :: CPUTensor 'D.Float '[3,2]
-- (Float,[3,2])
t 
  :: forall shape dtype device
   . Tensor device dtype shape -- ^ input 
  -> Tensor device dtype shape -- ^ output
t _input = unsafePerformIO $ (ATen.cast1 ATen.Managed.t_t) _input

-- | tan
--
-- >>> dtype &&& shape $ tan (ones :: CPUTensor 'D.Float '[3,2])
-- (Float,[3,2])
tan
  :: forall shape dtype device
   . (StandardFloatingPointDTypeValidation device dtype)
  => Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
tan input = unsafePerformIO $ ATen.cast1 ATen.Managed.tan_t input

-- tensordot :: Tensor device dtype shape -> Tensor device dtype shape -> [Int] -> [Int] -> Tensor device dtype shape
-- tensordot _input _other _dims_input _dims_other = unsafePerformIO $ (ATen.cast4 ATen.Managed.tensordot_ttll) _input _other _dims_input _dims_other

-- threshold :: Tensor device dtype shape -> Float -> Float -> Tensor device dtype shape
-- threshold _input _threshold _value = unsafePerformIO $ (ATen.cast3 ATen.Managed.threshold_tss) _input _threshold _value

-- one_hot :: Tensor device dtype shape -> Int -> Tensor device dtype shape
-- one_hot _input _num_classes = unsafePerformIO $ (ATen.cast2 ATen.Managed.one_hot_tl) _input _num_classes

-- flip :: Tensor device dtype shape -> [Int] -> Tensor device dtype shape
-- flip _input _dims = unsafePerformIO $ (ATen.cast2 ATen.Managed.flip_tl) _input _dims

-- roll :: Tensor device dtype shape -> Int -> Int -> Tensor device dtype shape
-- roll _input _shifts _dims = unsafePerformIO $ (ATen.cast3 ATen.Managed.roll_tll) _input _shifts _dims

-- rot90 :: Tensor device dtype shape -> Int -> [Int] -> Tensor device dtype shape
-- rot90 _input _k _dims = unsafePerformIO $ (ATen.cast3 ATen.Managed.rot90_tll) _input _k _dims

-- triplet_margin_loss :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Double -> Double -> Double -> Bool -> Int -> Tensor device dtype shape
-- triplet_margin_loss _anchor _positive _negative _margin _p _eps _swap _reduction = unsafePerformIO $ (ATen.cast8 ATen.Managed.triplet_margin_loss_tttdddbl) _anchor _positive _negative _margin _p _eps _swap _reduction

-- | trunc
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
--
-- >>> dtype &&& shape $ trunc (ones :: CPUTensor 'D.Float '[3,2])
-- (Float,[3,2])
trunc
  :: forall shape dtype device
   . (StandardFloatingPointDTypeValidation device dtype)
  => Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
trunc input = unsafePerformIO $ ATen.cast1 ATen.Managed.trunc_t input

-- unique_dim :: Tensor device dtype shape -> Int -> Bool -> Bool -> Bool -> (Tensor device dtype shape,Tensor device dtype shape,Tensor device dtype shape)
-- unique_dim _input _dim _sorted _return_inverse _return_counts = unsafePerformIO $ (ATen.cast5 ATen.Managed.unique_dim_tlbbb) _input _dim _sorted _return_inverse _return_counts

-- unique_consecutive :: Tensor device dtype shape -> Bool -> Bool -> Int -> (Tensor device dtype shape,Tensor device dtype shape,Tensor device dtype shape)
-- unique_consecutive _input _return_inverse _return_counts _dim = unsafePerformIO $ (ATen.cast4 ATen.Managed.unique_consecutive_tbbl) _input _return_inverse _return_counts _dim

-- unique_dim_consecutive :: Tensor device dtype shape -> Int -> Bool -> Bool -> (Tensor device dtype shape,Tensor device dtype shape,Tensor device dtype shape)
-- unique_dim_consecutive _input _dim _return_inverse _return_counts = unsafePerformIO $ (ATen.cast4 ATen.Managed.unique_dim_consecutive_tlbb) _input _dim _return_inverse _return_counts

-- | UnsqueezeImpl
--
-- >>> :kind! UnsqueezeImpl '[4] 0
-- UnsqueezeImpl '[4] 0 :: Maybe [Nat]
-- = 'Just '[1, 4]
-- >>> :kind! UnsqueezeImpl '[4] 1
-- UnsqueezeImpl '[4] 1 :: Maybe [Nat]
-- = 'Just '[4, 1]
-- >>> :kind! UnsqueezeImpl '[4] 2
-- UnsqueezeImpl '[4] 2 :: Maybe [Nat]
-- = 'Nothing
type family UnsqueezeImpl (shape :: [a]) (dim :: Nat) :: Maybe [a] where
  UnsqueezeImpl xs        0   = Just (1 ': xs)
  UnsqueezeImpl (x ': xs) dim = AppendToMaybe x (UnsqueezeImpl xs (dim - 1))
  UnsqueezeImpl '[]       _   = Nothing

type family UnsqueezeCheck (shape :: [a]) (dim :: Nat) (result :: Maybe [a]) :: [a] where
  UnsqueezeCheck shape dim Nothing       = TypeError (Text "Cannot unsqueeze the tensor since the specified dimension " :<>:
                                                      ShowType dim :<>:
                                                      Text " is too large (the tensor is only " :<>:
                                                      ShowType (ListLength shape) :<>:
                                                      Text "D)")
  UnsqueezeCheck _     _   (Just shape') = shape'

type Unsqueeze shape dim = UnsqueezeCheck shape dim (UnsqueezeImpl shape dim)

-- | unsqueeze
--
-- >>> t = fromJust [1, 2, 3, 4] :: CPUTensor 'D.Int64 '[4]
-- >>> t' = unsqueeze @0 t
-- >>> :type t'
-- t' :: Tensor '( 'D.CPU, 0) 'D.Int64 '[1, 4]
-- >>> dtype &&& shape &&& (\u -> D.asValue (toDynamic u) :: [[Int]]) $ t'
-- (Int64,([1,4],[[1,2,3,4]]))
-- >>> t'' = unsqueeze @1 t
-- >>> :type t''
-- t'' :: Tensor '( 'D.CPU, 0) 'D.Int64 '[4, 1]
-- >>> dtype &&& shape &&& (\u -> D.asValue (toDynamic u) :: [[Int]]) $ t''
-- (Int64,([4,1],[[1],[2],[3],[4]]))
unsqueeze
  :: forall dim shape shape' dtype device
   . (KnownNat dim, shape' ~ Unsqueeze shape dim)
  => Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape' -- ^ output
unsqueeze input = unsafePerformIO $ ATen.cast2 ATen.Managed.unsqueeze_tl input (natValI @dim)

-- where' :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape
-- where' _condition _input _other = unsafePerformIO $ (ATen.cast3 ATen.Managed.where_ttt) _condition _input _other

-- where_ :: Tensor device dtype shape -> [Tensor device dtype shape]
-- where_ _condition = unsafePerformIO $ (ATen.cast1 ATen.Managed.where_t) _condition

-- norm_except_dim :: Tensor device dtype shape -> Int -> Int -> Tensor device dtype shape
-- norm_except_dim _v _pow _dim = unsafePerformIO $ (ATen.cast3 ATen.Managed.norm_except_dim_tll) _v _pow _dim

-- | zerosLike
--
-- >>> dtype &&& shape $ zerosLike (ones :: CPUTensor 'D.Float '[3,4,5])
-- (Float,[3,4,5])
zerosLike
  :: forall shape dtype device
   . Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
zerosLike input = unsafePerformIO $ ATen.cast1 ATen.Managed.zeros_like_t input

-- native_norm :: Tensor device dtype shape -> Float -> Tensor device dtype shape
-- native_norm _input _p = unsafePerformIO $ (ATen.cast2 ATen.Managed.native_norm_ts) _input _p

-- | clone
--
-- >>> t <- clone (ones :: CPUTensor 'D.Float '[3,2])
-- >>> dtype &&& shape $ t
-- (Float,[3,2])
clone
  :: forall shape dtype device
   . Tensor device dtype shape
  -> IO (Tensor device dtype shape)
clone input = ATen.cast1 ATen.Managed.clone_t input

-- s_native_addmm :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Float -> Float -> Tensor device dtype shape
-- s_native_addmm _input _mat1 _mat2 _beta _alpha = unsafePerformIO $ (ATen.cast5 ATen.Managed.s_native_addmm_tttss) _input _mat1 _mat2 _beta _alpha

-- | addmm
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
-- TODO: can we use D.Scalar here for beta and alpha?
--
-- >>> t = addmm 1 1 (ones :: CPUTensor 'D.Float '[3,2]) (zeros :: CPUTensor 'D.Float '[2,4]) (ones :: CPUTensor 'D.Float '[])
-- >>> dtype &&& shape $ t
-- (Float,[3,4])
-- >>> :t t
-- t :: Tensor '( 'D.CPU, 0) 'D.Float '[3, 4]
addmm
  :: forall shape' shape n k m dtype device
   . ( All KnownNat '[n, k, m]
     , shape' ~ Broadcast shape '[n,m]
     )
  => Float -- ^ beta
  -> Float -- ^ alpha
  -> Tensor device dtype '[n, k] -- ^ first input matrix
  -> Tensor device dtype '[k, m] -- ^ second input matrix
  -> Tensor device dtype shape -- ^ input tensor
  -> Tensor device dtype shape' -- ^ output tensor
addmm beta alpha mat1 mat2 input = unsafePerformIO $ ATen.cast5 ATen.Managed.addmm_tttss input mat1 mat2 beta alpha

-- hspmm :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape
-- hspmm _mat1 _mat2 = unsafePerformIO $ (ATen.cast2 ATen.Managed.hspmm_tt) _mat1 _mat2

-- | numel
-- TODO: since this is decidable at compile time, this should probably be calculated from the tensor type instead
numel
  :: forall shape dtype device
   . Tensor device dtype shape -- ^ input
  -> Int -- ^ output
numel input = unsafePerformIO $ ATen.cast1 ATen.Managed.tensor_numel input

-- unbind :: Tensor device dtype shape -> Int -> [Tensor device dtype shape]
-- unbind _input _dim = unsafePerformIO $ (ATen.cast2 ATen.Managed.unbind_tl) _input _dim

-- mkldnn_reorder_conv2d_weight :: Tensor device dtype shape -> (Int,Int) -> (Int,Int) -> (Int,Int) -> Int -> Tensor device dtype shape
-- mkldnn_reorder_conv2d_weight _input _padding _stride _dilation _groups = unsafePerformIO $ (ATen.cast5 ATen.Managed.mkldnn_reorder_conv2d_weight_tllll) _input _padding _stride _dilation _groups

--quantize_linear :: Tensor device dtype shape -> Double -> Int -> DType -> Tensor device dtype shape
--quantize_linear _input _scale _zero_point _dtype = unsafePerformIO $ (ATen.cast4 ATen.Managed.quantize_linear_tdls) _input _scale _zero_point _dtype

--quantize_linear_per_channel :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> [Int] -> DType -> Tensor device dtype shape
--quantize_linear_per_channel _input _scales _zero_points _axis _dtype = unsafePerformIO $ (ATen.cast5 ATen.Managed.quantize_linear_per_channel_tttls) _input _scales _zero_points _axis _dtype

-- dequantize :: Tensor device dtype shape -> Tensor device dtype shape
-- dequantize _input = unsafePerformIO $ (ATen.cast1 ATen.Managed.dequantize_t) _input

-- | qScale
-- TODO: are there any restrictions on the dtype?
qScale
  :: forall shape dtype device
   . Tensor device dtype shape -- ^ input
  -> Double -- ^ output
qScale input = unsafePerformIO $ ATen.cast1 ATen.Managed.q_scale_t input

-- | qZeroPoint
-- TODO: are there any restrictions on the dtype?
qZeroPoint
  :: forall shape dtype device
   . Tensor device dtype shape -- ^ input
  -> Int -- ^ output
qZeroPoint input = unsafePerformIO $ ATen.cast1 ATen.Managed.q_zero_point_t input

-- int_repr :: Tensor device dtype shape -> Tensor device dtype shape
-- int_repr _input = unsafePerformIO $ (ATen.cast1 ATen.Managed.int_repr_t) _input

-- fake_quantize_per_tensor_affine :: Tensor device dtype shape -> Double -> Int -> Int -> Int -> Tensor device dtype shape
-- fake_quantize_per_tensor_affine _input _scale _zero_point _quant_min _quant_max = unsafePerformIO $ (ATen.cast5 ATen.Managed.fake_quantize_per_tensor_affine_tdlll) _input _scale _zero_point _quant_min _quant_max

-- meshgrid :: [Tensor device dtype shape] -> [Tensor device dtype shape]
-- meshgrid _tensors = unsafePerformIO $ (ATen.cast1 ATen.Managed.meshgrid_l) _tensors

-- cartesian_prod :: [Tensor device dtype shape] -> Tensor device dtype shape
-- cartesian_prod _tensors = unsafePerformIO $ (ATen.cast1 ATen.Managed.cartesian_prod_l) _tensors

-- combinations :: Tensor device dtype shape -> Int -> Bool -> Tensor device dtype shape
-- combinations _input _r _with_replacement = unsafePerformIO $ (ATen.cast3 ATen.Managed.combinations_tlb) _input _r _with_replacement

-- | The directional specification of a recurrent function
--
data RNNDirectionality =
    Bidirectional  -- ^ Forward and backward along the sequential axis using independant parameters for each.
  | Unidirectional -- ^ Forward along the sequential axis.
  deriving (Show, Generic) -- TODO:  We could also have BidirectionalTied weights.

type family NumberOfDirections (directionality :: RNNDirectionality) :: Nat where
  NumberOfDirections Bidirectional = 2
  NumberOfDirections Unidirectional = 1

class KnownRNNDirectionality (directionality :: RNNDirectionality) where
  rnnBidirectional :: Bool

instance KnownRNNDirectionality Bidirectional where
  rnnBidirectional = True

instance KnownRNNDirectionality Unidirectional where
  rnnBidirectional = False

-- | Specification for the sequential axis of a recurrent function.
data RNNShapeOrder =
    BatchFirst    -- ^ Input is of shape (Batch, Sequence, Features)
  | SequenceFirst -- ^ Input is of shape (Sequence, Batch, Features)
  deriving (Show, Generic)

class KnownRNNShapeOrder (shapeOrder :: RNNShapeOrder) where
  rnnBatchFirst :: Bool

instance KnownRNNShapeOrder BatchFirst where
  rnnBatchFirst = True

instance KnownRNNShapeOrder SequenceFirst where
  rnnBatchFirst = False

type family RNNShape (shapeOrder :: RNNShapeOrder) (seqLen :: Nat) (batchSize :: Nat) (featureSize :: Nat) :: [Nat] where
  RNNShape BatchFirst    seqLen batchSize featureSize = '[batchSize, seqLen, featureSize]
  RNNShape SequenceFirst seqLen batchSize featureSize = '[seqLen, batchSize, featureSize]

type LSTMWIShape hiddenSize inputSize = '[4 * hiddenSize, inputSize]
type LSTMWHShape hiddenSize inputSize = '[4 * hiddenSize, hiddenSize]
type LSTMBIShape hiddenSize inputSize = '[4 * hiddenSize]
type LSTMBHShape hiddenSize inputSize = '[4 * hiddenSize]

type family LSTMRImpl (inputSize :: Nat) (hiddenSize :: Nat) (numLayers :: Nat) (directionality :: RNNDirectionality) :: [[Nat]] where
  LSTMRImpl inputSize hiddenSize 1         'Unidirectional = '[ LSTMWIShape hiddenSize inputSize
                                                              , LSTMWHShape hiddenSize inputSize
                                                              , LSTMBIShape hiddenSize inputSize
                                                              , LSTMBHShape hiddenSize inputSize
                                                              ]
  LSTMRImpl inputSize hiddenSize numLayers 'Unidirectional = LSTMRImpl inputSize hiddenSize (numLayers - 1) 'Unidirectional ++
                                                             '[ LSTMWIShape hiddenSize (hiddenSize * NumberOfDirections 'Unidirectional)
                                                              , LSTMWHShape hiddenSize (hiddenSize * NumberOfDirections 'Unidirectional)
                                                              , LSTMBIShape hiddenSize (hiddenSize * NumberOfDirections 'Unidirectional)
                                                              , LSTMBHShape hiddenSize (hiddenSize * NumberOfDirections 'Unidirectional)
                                                              ]
  LSTMRImpl inputSize hiddenSize 1         'Bidirectional  = '[ LSTMWIShape hiddenSize inputSize
                                                              , LSTMWHShape hiddenSize inputSize
                                                              , LSTMBIShape hiddenSize inputSize
                                                              , LSTMBHShape hiddenSize inputSize
                                                              , LSTMWIShape hiddenSize inputSize
                                                              , LSTMWHShape hiddenSize inputSize
                                                              , LSTMBIShape hiddenSize inputSize
                                                              , LSTMBHShape hiddenSize inputSize
                                                              ]
  LSTMRImpl inputSize hiddenSize numLayers 'Bidirectional  = LSTMRImpl inputSize hiddenSize (numLayers - 1) 'Bidirectional ++
                                                             '[ LSTMWIShape hiddenSize (hiddenSize * NumberOfDirections 'Bidirectional)
                                                              , LSTMWHShape hiddenSize (hiddenSize * NumberOfDirections 'Bidirectional)
                                                              , LSTMBIShape hiddenSize (hiddenSize * NumberOfDirections 'Bidirectional)
                                                              , LSTMBHShape hiddenSize (hiddenSize * NumberOfDirections 'Bidirectional)
                                                              , LSTMWIShape hiddenSize (hiddenSize * NumberOfDirections 'Bidirectional)
                                                              , LSTMWHShape hiddenSize (hiddenSize * NumberOfDirections 'Bidirectional)
                                                              , LSTMBIShape hiddenSize (hiddenSize * NumberOfDirections 'Bidirectional)
                                                              , LSTMBHShape hiddenSize (hiddenSize * NumberOfDirections 'Bidirectional)
                                                              ]

type family LSTMR' (shapes :: [[Nat]]) (dtype :: D.DType) (device :: (D.DeviceType, Nat)) :: [a] where
  LSTMR' '[]               dtype device = '[]
  LSTMR' (shape ': shapes) dtype device = Tensor device dtype shape ': LSTMR' shapes dtype device

type LSTMR inputSize hiddenSize numLayers directionality dtype device = LSTMR' (LSTMRImpl inputSize hiddenSize numLayers directionality) dtype device

-- | lstm
-- Parameters for this ATen function are non-trivially provided.  See the
-- `Typed.NN.LSTM` module for doctests.
--
lstm
  :: forall
       shapeOrder
       directionality
       numLayers
       seqLen
       batchSize
       inputSize
       outputSize
       hiddenSize
       inputShape
       outputShape
       hxShape
       tensorParameters
       dtype
       device
   . ( KnownNat numLayers
     , KnownRNNShapeOrder shapeOrder
     , KnownRNNDirectionality directionality
     , outputSize ~ (hiddenSize * NumberOfDirections directionality)
     , inputShape ~ RNNShape shapeOrder seqLen batchSize inputSize
     , outputShape ~ RNNShape shapeOrder seqLen batchSize outputSize
     , hxShape ~ '[numLayers * NumberOfDirections directionality, batchSize, hiddenSize]
     , tensorParameters ~ LSTMR inputSize hiddenSize numLayers directionality dtype device
     , ATen.Castable (HList tensorParameters) [D.ATenTensor]
     )
  => HList tensorParameters
  -> Double
  -> Bool
  -> (Tensor device dtype hxShape, Tensor device dtype hxShape)
  -> Tensor device dtype inputShape
  -> ( Tensor device dtype outputShape
     , Tensor device dtype hxShape
     , Tensor device dtype hxShape
     )
lstm tensorParameters dropoutProb dropoutOn (cc, hc) input = unsafePerformIO $ ATen.cast9
  ATen.Managed.lstm_tllbldbbb
  input
  hx
  tensorParameters
  hasBiases
  numLayers
  dropoutProb
  dropoutOn
  (rnnBidirectional @directionality)
  (rnnBatchFirst @shapeOrder)
 where
  hasBiases = True
  hx :: [Tensor device dtype hxShape]
  hx = [cc, hc]
  numLayers :: I.Int64
  numLayers  = fromIntegral $ natValI @numLayers

-- | lstmCell
--
-- >>> dtype &&& shape $ fst $ lstmCell (ones :: CPUTensor 'D.Float '[12,2]) (ones :: CPUTensor 'D.Float '[12,3]) (ones :: CPUTensor 'D.Float '[12]) (ones :: CPUTensor 'D.Float '[12]) ((ones :: CPUTensor 'D.Float '[2,3]), (ones :: CPUTensor 'D.Float '[2,3])) (ones :: CPUTensor 'D.Float '[2,2])
-- (Float,[2,3])
lstmCell
  :: forall inputSize hiddenSize batchSize dtype device
   . Tensor device dtype '[4 * hiddenSize, inputSize]
  -> Tensor device dtype '[4 * hiddenSize, hiddenSize]
  -> Tensor device dtype '[4 * hiddenSize]
  -> Tensor device dtype '[4 * hiddenSize]
  -> ( Tensor device dtype '[batchSize, hiddenSize]
     , Tensor device dtype '[batchSize, hiddenSize]
     )
  -> Tensor device dtype '[batchSize, inputSize]
  -> ( Tensor device dtype '[batchSize, hiddenSize]
     , Tensor device dtype '[batchSize, hiddenSize]
     )
lstmCell wi wh bi bh (cc, hc) input =
  unsafePerformIO
    $ ATen.cast6 ATen.Managed.lstm_cell_tltttt input hx wi wh bi bh
  where hx = [cc, hc] :: [Tensor device dtype '[batchSize, hiddenSize]]

type GRUWIShape hiddenSize inputSize = '[3 * hiddenSize, inputSize]
type GRUWHShape hiddenSize inputSize = '[3 * hiddenSize, hiddenSize]
type GRUBIShape hiddenSize inputSize = '[3 * hiddenSize]
type GRUBHShape hiddenSize inputSize = '[3 * hiddenSize]

type family GRURImpl (inputSize :: Nat) (hiddenSize :: Nat) (numLayers :: Nat) (directionality :: RNNDirectionality) :: [[Nat]] where
  GRURImpl inputSize hiddenSize 1         'Unidirectional = '[ GRUWIShape hiddenSize inputSize
                                                             , GRUWHShape hiddenSize inputSize
                                                             , GRUBIShape hiddenSize inputSize
                                                             , GRUBHShape hiddenSize inputSize
                                                             ]
  GRURImpl inputSize hiddenSize numLayers 'Unidirectional = GRURImpl inputSize hiddenSize (numLayers - 1) 'Unidirectional ++
                                                            '[ GRUWIShape hiddenSize (hiddenSize * NumberOfDirections 'Unidirectional)
                                                             , GRUWHShape hiddenSize (hiddenSize * NumberOfDirections 'Unidirectional)
                                                             , GRUBIShape hiddenSize (hiddenSize * NumberOfDirections 'Unidirectional)
                                                             , GRUBHShape hiddenSize (hiddenSize * NumberOfDirections 'Unidirectional)
                                                             ]
  GRURImpl inputSize hiddenSize 1         'Bidirectional  = '[ GRUWIShape hiddenSize inputSize
                                                             , GRUWHShape hiddenSize inputSize
                                                             , GRUBIShape hiddenSize inputSize
                                                             , GRUBHShape hiddenSize inputSize
                                                             , GRUWIShape hiddenSize inputSize
                                                             , GRUWHShape hiddenSize inputSize
                                                             , GRUBIShape hiddenSize inputSize
                                                             , GRUBHShape hiddenSize inputSize
                                                             ]
  GRURImpl inputSize hiddenSize numLayers 'Bidirectional  = GRURImpl inputSize hiddenSize (numLayers - 1) 'Bidirectional ++
                                                            '[ GRUWIShape hiddenSize (hiddenSize * NumberOfDirections 'Bidirectional)
                                                             , GRUWHShape hiddenSize (hiddenSize * NumberOfDirections 'Bidirectional)
                                                             , GRUBIShape hiddenSize (hiddenSize * NumberOfDirections 'Bidirectional)
                                                             , GRUBHShape hiddenSize (hiddenSize * NumberOfDirections 'Bidirectional)
                                                             , GRUWIShape hiddenSize (hiddenSize * NumberOfDirections 'Bidirectional)
                                                             , GRUWHShape hiddenSize (hiddenSize * NumberOfDirections 'Bidirectional)
                                                             , GRUBIShape hiddenSize (hiddenSize * NumberOfDirections 'Bidirectional)
                                                             , GRUBHShape hiddenSize (hiddenSize * NumberOfDirections 'Bidirectional)
                                                             ]

type family GRUR' (shapes :: [[Nat]]) (dtype :: D.DType) (device :: (D.DeviceType, Nat)) :: [a] where
  GRUR' '[]               dtype device = '[]
  GRUR' (shape ': shapes) dtype device = Tensor device dtype shape ': GRUR' shapes dtype device

type GRUR inputSize hiddenSize numLayers directionality dtype device = GRUR' (GRURImpl inputSize hiddenSize numLayers directionality) dtype device

-- | gru
-- Parameters for this ATen function are non-trivially provided.  See the
-- `Typed.NN.GRU` module for doctests.
--
gru
  :: forall
       shapeOrder
       directionality
       numLayers
       seqLen
       batchSize
       inputSize
       outputSize
       hiddenSize
       inputShape
       outputShape
       hcShape
       tensorParameters
       dtype
       device
   . ( KnownNat numLayers
     , KnownRNNShapeOrder shapeOrder
     , KnownRNNDirectionality directionality
     , outputSize ~ (hiddenSize * NumberOfDirections directionality)
     , inputShape ~ RNNShape shapeOrder seqLen batchSize inputSize
     , outputShape ~ RNNShape shapeOrder seqLen batchSize outputSize
     , hcShape ~ '[numLayers * NumberOfDirections directionality, batchSize, hiddenSize]
     , tensorParameters ~ GRUR inputSize hiddenSize numLayers directionality dtype device
     , ATen.Castable (HList tensorParameters) [D.ATenTensor]
     )
  => HList tensorParameters
  -> Double
  -> Bool
  -> Tensor device dtype hcShape
  -> Tensor device dtype inputShape
  -> ( Tensor device dtype outputShape
     , Tensor device dtype hcShape
     )
gru tensorParameters dropoutProb dropoutOn hc input = unsafePerformIO $ ATen.cast9
  ATen.Managed.gru_ttlbldbbb
  input
  hc
  tensorParameters
  hasBiases
  numLayers
  dropoutProb
  dropoutOn
  (rnnBidirectional @directionality)
  (rnnBatchFirst @shapeOrder)
 where
  hasBiases = True
  numLayers :: I.Int64
  numLayers  = fromIntegral $ natValI @numLayers

-- | gruCell
--
-- >>> dtype &&& shape $ gruCell (ones :: CPUTensor 'D.Float '[9,2]) (ones :: CPUTensor 'D.Float '[9,3]) (ones :: CPUTensor 'D.Float '[9]) (ones :: CPUTensor 'D.Float '[9]) (ones :: CPUTensor 'D.Float '[2,3]) (ones :: CPUTensor 'D.Float '[2,2])
-- (Float,[2,3])
gruCell
  :: forall inputSize hiddenSize batchSize dtype device
   . Tensor device dtype '[3 * hiddenSize, inputSize]
  -> Tensor device dtype '[3 * hiddenSize, hiddenSize]
  -> Tensor device dtype '[3 * hiddenSize]
  -> Tensor device dtype '[3 * hiddenSize]
  -> Tensor device dtype '[batchSize, hiddenSize]
  -> Tensor device dtype '[batchSize, inputSize]
  -> Tensor device dtype '[batchSize, hiddenSize]
gruCell wi wh bi bh hx input =
  unsafePerformIO
    $ ATen.cast6 ATen.Managed.gru_cell_tttttt input hx wi wh bi bh

-- rnn_tanh_cell :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape
-- rnn_tanh_cell _input _hx _w_ih _w_hh _b_ih _b_hh = unsafePerformIO $ (ATen.cast6 ATen.Managed.rnn_tanh_cell_tttttt) _input _hx _w_ih _w_hh _b_ih _b_hh

-- rnn_relu_cell :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape
-- rnn_relu_cell _input _hx _w_ih _w_hh _b_ih _b_hh = unsafePerformIO $ (ATen.cast6 ATen.Managed.rnn_relu_cell_tttttt) _input _hx _w_ih _w_hh _b_ih _b_hh

-- quantized_lstm :: Tensor device dtype shape -> [Tensor device dtype shape] -> [Tensor device dtype shape] -> Bool -> Int -> Double -> Bool -> Bool -> Bool -> DType -> (Tensor device dtype shape,Tensor device dtype shape,Tensor device dtype shape)
-- quantized_lstm _input _hx _params _has_biases _num_layers _dropout _train _bidirectional _batch_first _dtype = unsafePerformIO $ (ATen.ATen.cast10 ATen.Managed.quantized_lstm_tllbldbbbs) _input _hx _params _has_biases _num_layers _dropout _train _bidirectional _batch_first _dtype

-- quantized_lstm_cell :: Tensor device dtype shape -> [Tensor device dtype shape] -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Float -> Float -> Float -> Float -> (Tensor device dtype shape,Tensor device dtype shape)
-- quantized_lstm_cell _input _hx _w_ih _w_hh _b_ih _b_hh _packed_ih _packed_hh _col_offsets_ih _col_offsets_hh _scale_ih _scale_hh _zero_point_ih _zero_point_hh = unsafePerformIO $ (ATen.ATen.cast14 ATen.Managed.quantized_lstm_cell_tlttttttttssss) _input _hx _w_ih _w_hh _b_ih _b_hh _packed_ih _packed_hh _col_offsets_ih _col_offsets_hh _scale_ih _scale_hh _zero_point_ih _zero_point_hh

-- quantized_gru_cell :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Float -> Float -> Float -> Float -> Tensor device dtype shape
-- quantized_gru_cell _input _hx _w_ih _w_hh _b_ih _b_hh _packed_ih _packed_hh _col_offsets_ih _col_offsets_hh _scale_ih _scale_hh _zero_point_ih _zero_point_hh = unsafePerformIO $ (ATen.ATen.cast14 ATen.Managed.quantized_gru_cell_ttttttttttssss) _input _hx _w_ih _w_hh _b_ih _b_hh _packed_ih _packed_hh _col_offsets_ih _col_offsets_hh _scale_ih _scale_hh _zero_point_ih _zero_point_hh

-- quantized_rnn_relu_cell :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Float -> Float -> Float -> Float -> Tensor device dtype shape
-- quantized_rnn_relu_cell _input _hx _w_ih _w_hh _b_ih _b_hh _packed_ih _packed_hh _col_offsets_ih _col_offsets_hh _scale_ih _scale_hh _zero_point_ih _zero_point_hh = unsafePerformIO $ (ATen.ATen.cast14 ATen.Managed.quantized_rnn_relu_cell_ttttttttttssss) _input _hx _w_ih _w_hh _b_ih _b_hh _packed_ih _packed_hh _col_offsets_ih _col_offsets_hh _scale_ih _scale_hh _zero_point_ih _zero_point_hh

-- quantized_rnn_tanh_cell :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Float -> Float -> Float -> Float -> Tensor device dtype shape
-- quantized_rnn_tanh_cell _input _hx _w_ih _w_hh _b_ih _b_hh _packed_ih _packed_hh _col_offsets_ih _col_offsets_hh _scale_ih _scale_hh _zero_point_ih _zero_point_hh = unsafePerformIO $ (ATen.ATen.cast14 ATen.Managed.quantized_rnn_tanh_cell_ttttttttttssss) _input _hx _w_ih _w_hh _b_ih _b_hh _packed_ih _packed_hh _col_offsets_ih _col_offsets_hh _scale_ih _scale_hh _zero_point_ih _zero_point_hh

-- masked_scatter :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape
-- masked_scatter _input _mask _source = unsafePerformIO $ (ATen.cast3 ATen.Managed.masked_scatter_ttt) _input _mask _source

-- index_add :: Tensor device dtype shape -> Int -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape
-- index_add _input _dim _index _source = unsafePerformIO $ (ATen.cast4 ATen.Managed.index_add_tltt) _input _dim _index _source

-- scatter_add :: Tensor device dtype shape -> Int -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape
-- scatter_add _input _dim _index _src = unsafePerformIO $ (ATen.cast4 ATen.Managed.scatter_add_tltt) _input _dim _index _src

-- addbmm :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Float -> Float -> Tensor device dtype shape
-- addbmm _input _batch1 _batch2 _beta _alpha = unsafePerformIO $ (ATen.cast5 ATen.Managed.addbmm_tttss) _input _batch1 _batch2 _beta _alpha

-- cross :: Tensor device dtype shape -> Tensor device dtype shape -> Int -> Tensor device dtype shape
-- cross _input _other _dim = unsafePerformIO $ (ATen.cast3 ATen.Managed.cross_ttl) _input _other _dim

type family MatrixOrMatrixBatch (shape :: [Nat]) :: [Nat] where
  MatrixOrMatrixBatch (n : m : '[])     = '[n, m]
  MatrixOrMatrixBatch (b : n : m : '[]) = '[b, n, m]
  MatrixOrMatrixBatch _                 = TypeError (Text "The input must be matrix or a batch of matrices.")

-- | triu
-- TODO: triu is not implemented for D.Bool, or maybe numeric type is lifted?
--
-- >>> t = ones :: CPUTensor 'D.Float '[3, 4]
-- >>> dtype &&& shape &&& (\t' -> D.asValue (toDynamic t') :: [[Float]]) $ triu 0 t
-- (Float,([3,4],[[1.0,1.0,1.0,1.0],[0.0,1.0,1.0,1.0],[0.0,0.0,1.0,1.0]]))
-- >>> dtype &&& shape &&& (\t' -> D.asValue (toDynamic t') :: [[Float]]) $ triu 1 t
-- (Float,([3,4],[[0.0,1.0,1.0,1.0],[0.0,0.0,1.0,1.0],[0.0,0.0,0.0,1.0]]))
-- >>> dtype &&& shape &&& (\t' -> D.asValue (toDynamic t') :: [[Float]]) $ triu (-1) t
-- (Float,([3,4],[[1.0,1.0,1.0,1.0],[1.0,1.0,1.0,1.0],[0.0,1.0,1.0,1.0]]))
triu
  :: forall shape dtype device
   . (shape ~ MatrixOrMatrixBatch shape)
  => Int -- ^ diagonal
  -> Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
triu diagonal input = unsafePerformIO $ ATen.cast2 ATen.Managed.triu_tl input diagonal

-- | tril
-- TODO: tril is not implemented for D.Bool, or maybe numeric type is lifted?
--
-- >>> t = ones :: CPUTensor 'D.Float '[3, 4]
-- >>> dtype &&& shape &&& (\t' -> D.asValue (toDynamic t') :: [[Float]]) $ tril 0 t
-- (Float,([3,4],[[1.0,0.0,0.0,0.0],[1.0,1.0,0.0,0.0],[1.0,1.0,1.0,0.0]]))
-- >>> dtype &&& shape &&& (\t' -> D.asValue (toDynamic t') :: [[Float]]) $ tril 1 t
-- (Float,([3,4],[[1.0,1.0,0.0,0.0],[1.0,1.0,1.0,0.0],[1.0,1.0,1.0,1.0]]))
-- >>> dtype &&& shape &&& (\t' -> D.asValue (toDynamic t') :: [[Float]]) $ tril (-1) t
-- (Float,([3,4],[[0.0,0.0,0.0,0.0],[1.0,0.0,0.0,0.0],[1.0,1.0,0.0,0.0]]))
tril
  :: forall shape dtype device
   . (shape ~ MatrixOrMatrixBatch shape)
  => Int -- ^ diagonal
  -> Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
tril diagonal input = unsafePerformIO $ ATen.cast2 ATen.Managed.tril_tl input diagonal

-- | trace
--
-- >>> dtype &&& shape $ trace (ones :: CPUTensor 'D.Float '[3,2])
-- (Float,[3,2])
trace 
  :: forall shape dtype device
   . Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
trace _input = unsafePerformIO $ (ATen.cast1 ATen.Managed.trace_t) _input

-- take :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape
-- take _input _index = unsafePerformIO $ (ATen.cast2 ATen.Managed.take_tt) _input _index

-- index_select :: Tensor device dtype shape -> Int -> Tensor device dtype shape -> Tensor device dtype shape
-- index_select _input _dim _index = unsafePerformIO $ (ATen.cast3 ATen.Managed.index_select_tlt) _input _dim _index

-- masked_select :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape
-- masked_select _input _mask = unsafePerformIO $ (ATen.cast2 ATen.Managed.masked_select_tt) _input _mask

-- | nonzero
-- 
-- >>> dtype &&& shape $ nonzero (zeros :: CPUTensor 'D.Float '[3,2])
-- (Float,[3,2])
nonzero 
  :: forall shape dtype device
   . Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
nonzero _input = unsafePerformIO $ (ATen.cast1 ATen.Managed.nonzero_t) _input

-- nonzero_numpy :: Tensor device dtype shape -> [Tensor device dtype shape]
-- nonzero_numpy _input = unsafePerformIO $ (ATen.cast1 ATen.Managed.nonzero_numpy_t) _input

-- gather :: Tensor device dtype shape -> Int -> Tensor device dtype shape -> Bool -> Tensor device dtype shape
-- gather _input _dim _index _sparse_grad = unsafePerformIO $ (ATen.cast4 ATen.Managed.gather_tltb) _input _dim _index _sparse_grad

-- addcmul :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Float -> Tensor device dtype shape
-- addcmul _input _tensor1 _tensor2 _value = unsafePerformIO $ (ATen.cast4 ATen.Managed.addcmul_ttts) _input _tensor1 _tensor2 _value

-- addcdiv :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Float -> Tensor device dtype shape
-- addcdiv _input _tensor1 _tensor2 _value = unsafePerformIO $ (ATen.cast4 ATen.Managed.addcdiv_ttts) _input _tensor1 _tensor2 _value

-- lstsq :: Tensor device dtype shape -> Tensor device dtype shape -> (Tensor device dtype shape,Tensor device dtype shape)
-- lstsq _input _A = unsafePerformIO $ (ATen.cast2 ATen.Managed.lstsq_tt) _input _A

-- triangular_solve :: Tensor device dtype shape -> Tensor device dtype shape -> Bool -> Bool -> Bool -> (Tensor device dtype shape,Tensor device dtype shape)
-- triangular_solve _input _A _upper _transpose _unitriangular = unsafePerformIO $ (ATen.cast5 ATen.Managed.triangular_solve_ttbbb) _input _A _upper _transpose _unitriangular

-- qr :: Tensor device dtype shape -> Bool -> (Tensor device dtype shape,Tensor device dtype shape)
-- qr _input _some = unsafePerformIO $ (ATen.cast2 ATen.Managed.qr_tb) _input _some

-- ormqr :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Bool -> Bool -> Tensor device dtype shape
-- ormqr _input _input2 _input3 _left _transpose = unsafePerformIO $ (ATen.cast5 ATen.Managed.ormqr_tttbb) _input _input2 _input3 _left _transpose

-- lu_solve :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape
-- lu_solve _input _LU_data _LU_pivots = unsafePerformIO $ (ATen.cast3 ATen.Managed.lu_solve_ttt) _input _LU_data _LU_pivots

-- | lgamma function
--
-- >>> dtype &&& shape $ lgamma (ones :: CPUTensor 'D.Float '[3,2])
-- (Float,[3,2])
lgamma
  :: forall shape dtype device
   . (StandardFloatingPointDTypeValidation device dtype)
  => Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
lgamma input = unsafePerformIO $ ATen.cast1 ATen.Managed.lgamma_t input

-- | digamma function
--
-- >>> dtype &&& shape $ digamma (ones :: CPUTensor 'D.Float '[3,2])
-- (Float,[3,2])
digamma
  :: forall shape dtype device
   . (StandardFloatingPointDTypeValidation device dtype)
  => Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
digamma input = unsafePerformIO $ ATen.cast1 ATen.Managed.digamma_t input

-- | polygamma function
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
polygamma
  :: forall shape dtype device
   . Int -- ^ order
  -> Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
polygamma n input = unsafePerformIO $ ATen.cast2 ATen.Managed.polygamma_lt n input

-- | inverse of the error function
--
-- >>> dtype &&& shape $ erfinv (ones :: CPUTensor 'D.Float '[3,2])
-- (Float,[3,2])
erfinv
  :: forall shape dtype device
   . (StandardFloatingPointDTypeValidation device dtype)
  => Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
erfinv input = unsafePerformIO $ ATen.cast1 ATen.Managed.erfinv_t input

-- dist :: Tensor device dtype shape -> Tensor device dtype shape -> Float -> Tensor device dtype shape
-- dist _input _other _p = unsafePerformIO $ (ATen.cast3 ATen.Managed.dist_tts) _input _other _p

-- atan2 :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape
-- atan2 _input _other = unsafePerformIO $ (ATen.cast2 ATen.Managed.atan2_tt) _input _other

-- histc :: Tensor device dtype shape -> Int -> Float -> Float -> Tensor device dtype shape
-- histc _input _bins _min _max = unsafePerformIO $ (ATen.cast4 ATen.Managed.histc_tlss) _input _bins _min _max

-- | minAll
--
-- >>> dtype &&& shape $ minAll (ones :: CPUTensor 'D.Float '[2,2])
-- (Float,[])
minAll
  :: forall shape dtype device
   . Tensor device dtype shape -- ^ input
  -> Tensor device dtype '[] -- ^ output
minAll input = unsafePerformIO $ ATen.cast1 ATen.Managed.min_t input

type family DropValue (shape :: [Nat]) (i :: Nat) :: [Nat] where
  DropValue '[]     _ = TypeError (Text "Can not find a element in the list.")
  DropValue (x: xs) 0 = xs
  DropValue (x: xs) i = x ': DropValue xs (i-1)

-- | minDim
--
-- >>> t = ones :: CPUTensor 'D.Float '[3,4,5]
-- >>> dtype &&& shape $ fst $ minDim @0 t
-- (Float,[4,5])
-- >>> dtype &&& shape $ fst $ minDim @1 t
-- (Float,[3,5])
-- >>> dtype &&& shape $ fst $ minDim @2 t
-- (Float,[3,4])
minDim
  :: forall d shape dtype device
   . (KnownNat d)
  => Tensor device dtype shape -- ^ input
  -> ( Tensor device dtype    (DropValue shape d)
     , Tensor device 'D.Int64 (DropValue shape d)
     ) -- ^ output
minDim input = unsafePerformIO $ ATen.cast2 ATen.Managed.min_tl input (natValI @d)

-- | maxAll
--
-- >>> dtype &&& shape $ maxAll (ones :: CPUTensor 'D.Float '[2,2])
-- (Float,[])
maxAll
  :: forall shape dtype device
   . Tensor device dtype shape -- ^ input
  -> Tensor device dtype '[] -- ^ output
maxAll input = unsafePerformIO $ ATen.cast1 ATen.Managed.max_t input

-- | maxDim
--
-- >>> t = ones :: CPUTensor 'D.Float '[3,4,5]
-- >>> dtype &&& shape $ fst $ maxDim @0 t
-- (Float,[4,5])
-- >>> dtype &&& shape $ fst $ maxDim @1 t
-- (Float,[3,5])
-- >>> dtype &&& shape $ fst $ maxDim @2 t
-- (Float,[3,4])
maxDim
  :: forall d shape dtype device
   . (KnownNat d)
  => Tensor device dtype shape -- ^ input
  -> ( Tensor device dtype    (DropValue shape d)
     , Tensor device 'D.Int64 (DropValue shape d)
     ) -- ^ output
maxDim input = unsafePerformIO $ ATen.cast2 ATen.Managed.max_tl input (natValI @d)

-- | medianAll
--
-- >>> dtype &&& shape $ medianAll (ones :: CPUTensor 'D.Float '[2,2])
-- (Float,[])
medianAll
  :: forall shape dtype device
   . Tensor device dtype shape -- ^ input
  -> Tensor device dtype '[] -- ^ output
medianAll input = unsafePerformIO $ ATen.cast1 ATen.Managed.median_t input

-- | medianDim
--
-- >>> t = ones :: CPUTensor 'D.Float '[3,4,5]
-- >>> dtype &&& shape $ fst $ medianDim @0 t
-- (Float,[4,5])
-- >>> dtype &&& shape $ fst $ medianDim @1 t
-- (Float,[3,5])
-- >>> dtype &&& shape $ fst $ medianDim @2 t
-- (Float,[3,4])
medianDim
  :: forall d shape dtype device
   . (KnownNat d)
  => Tensor device dtype shape -- ^ input
  -> ( Tensor device dtype    (DropValue shape d)
     , Tensor device 'D.Int64 (DropValue shape d)
     ) -- ^ output
medianDim input = unsafePerformIO $ ATen.cast2 ATen.Managed.median_tl input (natValI @d)

-- | median
-- See https://pytorch.org/docs/stable/torch.html#torch.median.
--
-- >>> t = fromJust [[5, 1], [3, 2], [4, 1], [2, 7]] :: CPUTensor 'D.Float '[4, 2]
-- >>> median' @0 @KeepDim t
-- (Tensor Float [1,2] [[ 3.0000   ,  1.0000   ]],Tensor Int64 [1,2] [[ 1,  0]])
median'
  :: forall dim keepOrDropDim shape dtype device
   . (KnownNat dim, KnownKeepOrDropDim keepOrDropDim)
  => Tensor device dtype shape
  -> ( Tensor device dtype    (ConditionalDropDimension shape dim keepOrDropDim)
     , Tensor device 'D.Int64 (ConditionalDropDimension shape dim keepOrDropDim)
     )
median' input = unsafePerformIO $ ATen.cast3 ATen.Managed.median_tlb
                                        input
                                        (natValI @dim)
                                        (keepOrDropDimVal @keepOrDropDim)

-- | meanAll
--
-- >>> dtype &&& shape $ meanAll (ones :: CPUTensor 'D.Float '[2,2])
-- (Float,[])
meanAll
  :: forall shape dtype device
   . Tensor device dtype shape -- ^ input
  -> Tensor device dtype '[] -- ^ output
meanAll input = unsafePerformIO $ ATen.cast1 ATen.Managed.mean_t input

-- | meanDim
--
-- >>> t = ones :: CPUTensor 'D.Float '[3,4,5]
-- >>> dtype &&& shape $ meanDim @0 t
-- (Float,[4,5])
-- >>> dtype &&& shape $ meanDim @1 t
-- (Float,[3,5])
-- >>> dtype &&& shape $ meanDim @2 t
-- (Float,[3,4])
meanDim
  :: forall d shape dtype device
   . (KnownNat d)
  => Tensor device dtype shape -- ^ input
  -> Tensor device dtype    (DropValue shape d) -- ^ output
meanDim input = unsafePerformIO $ ATen.cast2 ATen.Managed.mean_tl input (natValI @d)

-- | mean
-- See https://pytorch.org/docs/stable/torch.html#torch.mean.
--
-- >>> t = fromJust [[5, 1], [3, 2], [4, 1], [2, 7]] :: CPUTensor 'D.Float '[4, 2]
-- >>> mean' @0 @KeepDim t
-- Tensor Float [1,2] [[ 3.5000   ,  2.7500   ]]
mean'
  :: forall dim keepOrDropDim shape dtype device
   . (KnownNat dim, KnownKeepOrDropDim keepOrDropDim)
  => Tensor device dtype shape
  -> Tensor device dtype (ConditionalDropDimension shape dim keepOrDropDim)
mean' input = unsafePerformIO $ ATen.cast3 ATen.Managed.mean_tlb
                                        input
                                        (natValI @dim)
                                        (keepOrDropDimVal @keepOrDropDim)

-- sort :: Tensor device dtype shape -> Int -> Bool -> (Tensor device dtype shape,Tensor device dtype shape)
-- sort _input _dim _descending = unsafePerformIO $ (ATen.cast3 ATen.Managed.sort_tlb) _input _dim _descending

-- argsort :: Tensor device dtype shape -> Int -> Bool -> Tensor device dtype shape
-- argsort _input _dim _descending = unsafePerformIO $ (ATen.cast3 ATen.Managed.argsort_tlb) _input _dim _descending

type family TopKCheck (k :: Nat) (shape :: [Nat]) (dim :: Nat) (satd :: Maybe Nat) (result :: Maybe a) :: a where
  TopKCheck _ shape dim _ Nothing       = DimOutOfBound shape dim
  TopKCheck _ shape dim Nothing _       = DimOutOfBound shape dim
  TopKCheck k shape dim (Just v) (Just result) = If ( k <=? v ) result (TypeError (Text "k must be less than or equal to the number of elements in the requested dimension."))


type TopK k shape dim = TopKCheck k shape dim (ExtractDim dim shape) (ReplaceDim dim shape k)

type family TopKDeviceAndDTypeCheck dtype (device :: (D.DeviceType, Nat)) :: Constraint where 
  TopKDeviceAndDTypeCheck D.Bool _           = (TypeError (Text "topk is not defined for Bool tensors."))
  TopKDeviceAndDTypeCheck D.Half '(D.CPU, _) = (TypeError (Text "topk is not defined for Half types on CPU."))
  TopKDeviceAndDTypeCheck _ _ = ()


-- | Returns the k largest (if largest is `True`) elements of the given input tensor along a given dimension.
--
-- >>> topk @3 @1 True True (ones :: CPUTensor 'D.Float '[2,3])
-- (Tensor Float [2,3] [[ 1.0000   ,  1.0000   ,  1.0000   ],
--                     [ 1.0000   ,  1.0000   ,  1.0000   ]],Tensor Int64 [2,3] [[ 0,  1,  2],
--                     [ 0,  1,  2]])
-- >>> topk @0 @1 True True (ones :: CPUTensor 'D.Float '[2,3])
-- (Tensor Float [2,0] [[],
--                     []],Tensor Int64 [2,0] [[],
--                     []])
--
topk 
  :: forall k dim shape dtype device 
   . (KnownNat k, KnownNat dim, All KnownNat shape, TopKDeviceAndDTypeCheck dtype device) 
   => Bool -- ^ if we're returning the top k largest (or, if False, the top k smallest)
   -> Bool -- ^ if the resulting k elements are themselves sorted
   -> Tensor device dtype shape 
   -> (Tensor device dtype (TopK k shape dim), Tensor device 'D.Int64 (TopK k shape dim))
topk _largest _sorted _input = unsafePerformIO $ (ATen.cast5 ATen.Managed.topk_tllbb) _input _k _dim _largest _sorted
  where 
  _k = natValI @k
  _dim = natValI @dim

-- renorm :: Tensor device dtype shape -> Float -> Int -> Float -> Tensor device dtype shape
-- renorm _input _p _dim _maxnorm = unsafePerformIO $ (ATen.cast4 ATen.Managed.renorm_tsls) _input _p _dim _maxnorm

-- equal :: Tensor device dtype shape -> Tensor device dtype shape -> Bool
-- equal _input _other = unsafePerformIO $ (ATen.cast2 ATen.Managed.equal_tt) _input _other

-- | alias
-- 
-- >>> dtype &&& shape $ alias (ones :: CPUTensor 'D.Float '[3,2])
-- (Float,[3,2])

alias 
  :: forall shape dtype device
   . Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
alias _input = unsafePerformIO $ (ATen.cast1 ATen.Managed.alias_t) _input

-- | L1 loss
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
--
-- >>> dtype &&& shape $ l1Loss @ReduceNone (ones :: CPUTensor 'D.Float '[2,2]) (ones :: CPUTensor 'D.Float '[2,2])
-- (Float,[2,2])
-- >>> dtype &&& shape $ l1Loss @ReduceSum (ones :: CPUTensor 'D.Float '[2,2]) (ones :: CPUTensor 'D.Float '[2,2])
-- (Float,[])
l1Loss
  :: forall reduction shape dtype device
   . (KnownReduction reduction)
  => Tensor device dtype shape -- ^ prediciton
  -> Tensor device dtype shape -- ^ target
  -> Tensor device dtype (ConditionalReduction shape reduction) -- ^ loss
l1Loss prediction target = unsafePerformIO
  $ ATen.cast3 ATen.Managed.l1_loss_ttl prediction target (reductionVal @reduction)

-- multi_margin_loss :: Tensor device dtype shape -> Tensor device dtype shape -> Float -> Float -> Tensor device dtype shape -> Int -> Tensor device dtype shape
-- multi_margin_loss _input _target _p _margin _weight _reduction = unsafePerformIO $ (ATen.cast6 ATen.Managed.multi_margin_loss_ttsstl) _input _target _p _margin _weight _reduction

-- multilabel_margin_loss :: Tensor device dtype shape -> Tensor device dtype shape -> Int -> Tensor device dtype shape
-- multilabel_margin_loss _input _target _reduction = unsafePerformIO $ (ATen.cast3 ATen.Managed.multilabel_margin_loss_ttl) _input _target _reduction

-- | negative log likelihood loss
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
-- See https://pytorch.org/docs/stable/nn.functional.html?highlight=nll_loss#torch.nn.functional.nll_loss.
--
-- >>> input <- randn @'[3, 5] @'D.Float @'( 'D.CPU, 0)
-- >>> target = fromJust [1, 0, 4] :: CPUTensor 'D.Int64 '[3]
-- >>> weight = ones @'[5] @'D.Float @'( 'D.CPU, 0)
-- >>> dtype &&& shape $ nllLoss @ReduceNone @3 @5 @'[] weight (-100) (logSoftmax @1 input) target
-- (Float,[3])
-- >>> dtype &&& shape $ nllLoss @ReduceMean @3 @5 @'[] weight (-100) (logSoftmax @1 input) target
-- (Float,[])
-- >>> input <- randn @'[3, 5, 2] @'D.Float @'( 'D.CPU, 0)
-- >>> target = fromJust [[1, 1], [0, 1], [4, 0]] :: CPUTensor 'D.Int64 '[3, 2]
-- >>> weight = ones @'[5] @'D.Float @'( 'D.CPU, 0)
-- >>> dtype &&& shape $ nllLoss @ReduceNone @3 @5 @'[2] weight (-100) (logSoftmax @1 input) target
-- (Float,[3,2])
-- >>> dtype &&& shape $ nllLoss @ReduceMean @3 @5 @'[2] weight (-100) (logSoftmax @1 input) target
-- (Float,[])
-- >>> input <- randn @'[3, 5, 1, 2] @'D.Float @'( 'D.CPU, 0)
-- >>> target = fromJust [[[1, 1]], [[0, 1]], [[4, 0]]] :: CPUTensor 'D.Int64 '[3, 1, 2]
-- >>> weight = ones @'[5] @'D.Float @'( 'D.CPU, 0)
-- >>> dtype &&& shape $ nllLoss @ReduceNone @3 @5 @'[1, 2] weight (-100) (logSoftmax @1 input) target
-- (Float,[3,1,2])
-- >>> dtype &&& shape $ nllLoss @ReduceMean @3 @5 @'[1, 2] weight (-100) (logSoftmax @1 input) target
-- (Float,[])
-- >>> input <- randn @'[3, 5, 2, 1, 2] @'D.Float @'( 'D.CPU, 0)
-- >>> target = fromJust [[[[1, 1]], [[0, 2]]], [[[0, 1]], [[1, 0]]], [[[4, 0]], [[1, 2]]]] :: CPUTensor 'D.Int64 '[3, 2, 1, 2]
-- >>> weight = ones @'[5] @'D.Float @'( 'D.CPU, 0)
-- >>> dtype &&& shape $ nllLoss @ReduceNone @3 @5 @'[2, 1, 2] weight (-100) (logSoftmax @1 input) target
-- (Float,[3,2,1,2])
-- >>> dtype &&& shape $ nllLoss @ReduceMean @3 @5 @'[2, 1, 2] weight (-100) (logSoftmax @1 input) target
-- (Float,[])
nllLoss
  :: forall reduction n c ds dtype device
   . (KnownReduction reduction, KnownNat n, KnownNat c, KnownShape ds)
  => Tensor device dtype '[c] -- ^ weight
  -> Int -- ^ ignore which index
  -> Tensor device dtype    (n ': c ': ds) -- ^ prediction
  -> Tensor device 'D.Int64 (n ': ds) -- ^ target
  -> Tensor device dtype    (ConditionalReduction (n ': ds) reduction) -- ^ loss
nllLoss weight ignoreIndex prediction target = case shapeVal @ds of
  [] -> unsafePerformIO $ ATen.cast5 ATen.Managed.nll_loss_tttll
                                prediction
                                target
                                weight
                                (reductionVal @reduction)
                                ignoreIndex
  [_h, _w] -> unsafePerformIO $ ATen.cast5 ATen.Managed.nll_loss2d_tttll
                                      prediction
                                      target
                                      weight
                                      (reductionVal @reduction)
                                      ignoreIndex
  h : t -> case reductionVal @reduction of
    0 -> UnsafeMkTensor . (D.reshape ((natValI @n) : h : t)) $ out
    _ -> UnsafeMkTensor out
   where
    t'      = [1, foldl (*) h t]
    input'  = D.reshape (natValI @n : natValI @c : t') (toDynamic prediction)
    target' = D.reshape (natValI @n : t') (toDynamic target)
    out     = unsafePerformIO $ ATen.cast5 ATen.Managed.nll_loss2d_tttll
                                      input'
                                      target'
                                      weight
                                      (reductionVal @reduction)
                                      ignoreIndex

-- | smooth L1 loss
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
--
-- >>> dtype &&& shape $ smoothL1Loss @ReduceNone (ones :: CPUTensor 'D.Float '[2,2]) (ones :: CPUTensor 'D.Float '[2,2])
-- (Float,[2,2])
-- >>> dtype &&& shape $ smoothL1Loss @ReduceSum (ones :: CPUTensor 'D.Float '[2,2]) (ones :: CPUTensor 'D.Float '[2,2])
-- (Float,[])
smoothL1Loss
  :: forall reduction shape dtype device
   . (KnownReduction reduction)
  => Tensor device dtype shape -- ^ prediction
  -> Tensor device dtype shape -- ^ target
  -> Tensor device dtype (ConditionalReduction shape reduction) -- ^ loss
smoothL1Loss prediction target = unsafePerformIO
  $ ATen.cast3 ATen.Managed.smooth_l1_loss_ttl prediction target (reductionVal @reduction)

-- | soft margin loss
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
--
-- >>> dtype &&& shape $ softMarginLoss @ReduceNone (ones :: CPUTensor 'D.Float '[2,2]) (ones :: CPUTensor 'D.Float '[2,2])
-- (Float,[2,2])
-- >>> dtype &&& shape $ softMarginLoss @ReduceSum (ones :: CPUTensor 'D.Float '[2,2]) (ones :: CPUTensor 'D.Float '[2,2])
-- (Float,[])
softMarginLoss
  :: forall reduction shape dtype device
   . (KnownReduction reduction)
  => Tensor device dtype shape -- ^ prediction
  -> Tensor device dtype shape -- ^ target
  -> Tensor device dtype (ConditionalReduction shape reduction) -- ^ loss
softMarginLoss prediciton target = unsafePerformIO $ ATen.cast3
  ATen.Managed.soft_margin_loss_ttl
  prediciton
  target
  (reductionVal @reduction)

-- | elu
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
--
-- >>> dtype &&& shape $ elu 0.1 0.1 0.3 (ones :: CPUTensor 'D.Float '[3,2])
-- (Float,[3,2])
elu
  :: forall shape dtype a device
   . (D.Scalar a, StandardFloatingPointDTypeValidation device dtype)
  => a -- ^ alpha
  -> a -- ^ scale
  -> a -- ^ input scale
  -> Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
elu alpha scale inputScale input =
  unsafePerformIO $ ATen.cast4 ATen.Managed.elu_tsss input alpha scale inputScale

-- | glu
-- -- >>> dtype &&& shape $ glu (ones :: CPUTensor 'D.Float '[3,2]) 1
-- -- (Float,[3,1])
-- -- >>> dtype &&& shape $ glu (ones :: CPUTensor 'D.Float '[3,2]) 3
-- -- (Float,[3,2])
-- glu :: Tensor device dtype shape -> Int -> Tensor device dtype shape
-- glu _input _dim = unsafePerformIO $ (ATen.cast2 ATen.Managed.glu_tl) _input _dim


-- | hard tanh
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
--
-- >>> dtype &&& shape $ hardTanh 0 1 (ones :: CPUTensor 'D.Float '[3,2])
-- (Float,[3,2])
hardTanh
  :: forall shape dtype device
   . Float -- ^ minimum value
  -> Float -- ^ maximum value
  -> Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
hardTanh min_val max_val input =
  unsafePerformIO $ ATen.cast3 ATen.Managed.hardtanh_tss input min_val max_val

-- | leaky relu
--
-- >>> dtype &&& shape $ leakyRelu 0.01 (ones :: CPUTensor 'D.Float '[3,2])
-- (Float,[3,2])
leakyRelu
  :: forall a shape dtype device
   . (D.Scalar a, StandardFloatingPointDTypeValidation device dtype)
  => a -- ^ negative slope
  -> Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
leakyRelu negativeSlope input =
  unsafePerformIO $ ATen.cast2 ATen.Managed.leaky_relu_ts input negativeSlope

-- | logarithm of the sigmoid
--
-- >>> dtype &&& shape $ logSigmoid (ones :: CPUTensor 'D.Float '[3,2])
-- (Float,[3,2])
logSigmoid
  :: forall shape dtype device
   . (StandardFloatingPointDTypeValidation device dtype)
  => Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
logSigmoid input = unsafePerformIO $ ATen.cast1 ATen.Managed.log_sigmoid_t input

-- | softplus
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
-- See https://pytorch.org/docs/stable/nn.functional.html?highlight=softplus#torch.nn.functional.softplus.
--
-- >>> dtype &&& shape &&& (\t -> D.asValue (toDynamic t) :: [[Float]]) $ softplus 1 20 (ones :: CPUTensor 'D.Float '[3,2])
-- (Float,([3,2],[[1.3132616,1.3132616],[1.3132616,1.3132616],[1.3132616,1.3132616]]))
softplus
  :: forall a shape dtype device
   . D.Scalar a
  => a
  -> a
  -> Tensor device dtype shape
  -> Tensor device dtype shape
softplus beta threshold input = unsafePerformIO $ ATen.cast3 ATen.Managed.softplus_tss input beta threshold

-- | soft shrink
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
--
-- >>> dtype &&& shape $ softShrink 0.2 (ones :: CPUTensor 'D.Float '[3,2])
-- (Float,[3,2])
softShrink
  :: forall shape dtype device
   . Float -- ^ lambda
  -> Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
softShrink lambda input =
  unsafePerformIO $ ATen.cast2 ATen.Managed.softshrink_ts input lambda

-- | adaptive averaged 2-D pooling
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
--
-- >>> t = adaptiveAvgPool2d @'(8,16) (ones :: CPUTensor 'D.Float '[1,3,16,32])
-- >>> shape t
-- [1,3,8,16]
-- >>> :t t
-- t :: Tensor '( 'D.CPU, 0) 'D.Float '[1, 3, 8, 16]
adaptiveAvgPool2d
  :: forall outputSize channelSize inputSize0 inputSize1 batchSize dtype device
   . ( All KnownNat '[ channelSize
                     , inputSize0, inputSize1
                     , batchSize
                     , Torch.Typed.Aux.Fst outputSize, Torch.Typed.Aux.Snd outputSize
                     ]
     )
  => Tensor device dtype '[batchSize, channelSize, inputSize0, inputSize1] -- ^ input
  -> Tensor device dtype '[batchSize, channelSize, Torch.Typed.Aux.Fst outputSize, Torch.Typed.Aux.Snd outputSize] -- ^ output
adaptiveAvgPool2d input = unsafePerformIO $ ATen.cast2
  ATen.Managed.adaptive_avg_pool2d_tl
  input
  ([natValI @(Torch.Typed.Aux.Fst outputSize), natValI @(Torch.Typed.Aux.Snd outputSize)] :: [Int])

-- | MKLDNN adaptive averaged 2-D pooling
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
-- TODO: broken?
-- TODO: only defined for MKLDNN device?
-- TODO: test for availability of MKLDNN device?
-- TODO: merge with adaptiveAvgPool2d and dispatch based on (availability of MKLDNN) device in the function body?
--
-- -- >>> t = mkldnnAdaptiveAvgPool2d @'(8,16) (toMKLDNN (ones :: CPUTensor 'D.Float '[1,3,16,32]))
-- -- >>> shape t
-- -- [1,3,8,16]
-- -- >>> :t t
-- -- t :: Tensor '( 'D.CPU, 0) 'D.Float '[1, 3, 8, 16]
mkldnnAdaptiveAvgPool2d
  :: forall outputSize channelSize inputSize0 inputSize1 batchSize dtype device
   . ( All KnownNat '[ channelSize
                     , inputSize0, inputSize1
                     , batchSize
                     , Torch.Typed.Aux.Fst outputSize, Torch.Typed.Aux.Snd outputSize
                     ]
     )
  => Tensor device dtype '[batchSize, channelSize, inputSize0, inputSize1] -- ^ input
  -> Tensor device dtype '[batchSize, channelSize, Torch.Typed.Aux.Fst outputSize, Torch.Typed.Aux.Snd outputSize] -- ^ output
mkldnnAdaptiveAvgPool2d input = unsafePerformIO $ ATen.cast2
  ATen.Managed.adaptive_avg_pool2d_tl
  input
  ([natValI @(Torch.Typed.Aux.Fst outputSize), natValI @(Torch.Typed.Aux.Snd outputSize)] :: [Int])

-- | adaptive averaged 3-D pooling
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
--
-- >>> t = adaptiveAvgPool3d @'(8,16,2) (ones :: CPUTensor 'D.Float '[1,3,16,32,4])
-- >>> shape t
-- [1,3,8,16,2]
-- >>> :t t
-- t :: Tensor '( 'D.CPU, 0) 'D.Float '[1, 3, 8, 16, 2]
adaptiveAvgPool3d
  :: forall outputSize
            channelSize
            inputSize0 inputSize1 inputSize2
            batchSize
            dtype
            device
   . ( All KnownNat '[ channelSize
                     , inputSize0, inputSize1, inputSize2
                     , batchSize
                     , Fst3 outputSize, Snd3 outputSize, Trd3 outputSize
                     ]
     )
  => Tensor device dtype '[batchSize, channelSize, inputSize0, inputSize1, inputSize2] -- ^ input
  -> Tensor device dtype '[batchSize, channelSize, Fst3 outputSize, Snd3 outputSize, Trd3 outputSize] -- ^ output
adaptiveAvgPool3d input = unsafePerformIO $ ATen.cast2
  ATen.Managed.adaptive_avg_pool3d_tl
  input
  ([ natValI @(Fst3 outputSize)
   , natValI @(Snd3 outputSize)
   , natValI @(Trd3 outputSize)
   ] :: [Int]
  )

-- | adaptive 2-D max-pool
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
--
-- >>> (t, t') = adaptiveMaxPool2d @'(8,16) (ones :: CPUTensor 'D.Float '[1,3,16,32])
-- >>> shape t
-- [1,3,8,16]
-- >>> :t t
-- t :: Tensor '( 'D.CPU, 0) 'D.Float '[1, 3, 8, 16]
adaptiveMaxPool2d
  :: forall outputSize channelSize inputSize0 inputSize1 batchSize dtype device
   . ( All KnownNat '[ channelSize
                     , inputSize0, inputSize1
                     , batchSize
                     , Torch.Typed.Aux.Fst outputSize, Torch.Typed.Aux.Snd outputSize
                     ]
     )
  => Tensor device dtype '[batchSize, channelSize, inputSize0, inputSize1] -- ^ input
  -> ( Tensor device dtype    '[batchSize, channelSize, Torch.Typed.Aux.Fst outputSize, Torch.Typed.Aux.Snd outputSize]
     , Tensor device 'D.Int64 '[batchSize, channelSize, Torch.Typed.Aux.Fst outputSize, Torch.Typed.Aux.Snd outputSize]
     ) -- ^ output
adaptiveMaxPool2d input = unsafePerformIO $ ATen.cast2
  ATen.Managed.adaptive_max_pool2d_tl
  input
  ([natValI @(Torch.Typed.Aux.Fst outputSize), natValI @(Torch.Typed.Aux.Snd outputSize)] :: [Int])

-- | adaptive 3-D max-pool
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
--
-- >>> (t, t') = adaptiveMaxPool3d @'(8,16,2) (ones :: CPUTensor 'D.Float '[1,3,16,32,4])
-- >>> shape t
-- [1,3,8,16,2]
-- >>> :t t
-- t :: Tensor '( 'D.CPU, 0) 'D.Float '[1, 3, 8, 16, 2]
adaptiveMaxPool3d
  :: forall outputSize
            channelSize
            inputSize0 inputSize1 inputSize2
            batchSize
            dtype
            device
   . ( All KnownNat '[ channelSize
                     , inputSize0, inputSize1, inputSize2
                     , batchSize
                     , Fst3 outputSize, Snd3 outputSize, Trd3 outputSize
                     ]
     )
  => Tensor device dtype '[batchSize, channelSize, inputSize0, inputSize1, inputSize2] -- ^ input
  -> ( Tensor device dtype    '[batchSize, channelSize, Fst3 outputSize, Snd3 outputSize, Trd3 outputSize]
     , Tensor device 'D.Int64 '[batchSize, channelSize, Fst3 outputSize, Snd3 outputSize, Trd3 outputSize]
     ) -- ^ output
adaptiveMaxPool3d input = unsafePerformIO $ (ATen.cast2 ATen.Managed.adaptive_max_pool3d_tl)
  input
  ([ natValI @(Fst3 outputSize)
   , natValI @(Snd3 outputSize)
   , natValI @(Trd3 outputSize)
   ] :: [Int]
  )

-- | averaged 2-D pooling
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
--
-- >>> t = avgPool2d @'(1,1) @'(1,1) @'(0,0) (ones :: CPUTensor 'D.Float '[1,3,4,5])
-- >>> shape t
-- [1,3,4,5]
-- >>> :t t
-- t :: Tensor '( 'D.CPU, 0) 'D.Float '[1, 3, 4, 5]
avgPool2d
  :: forall kernelSize
            stride
            padding
            channelSize
            inputSize0 inputSize1
            batchSize
            outputSize0 outputSize1
            dtype
            device
   . ( All KnownNat '[ Torch.Typed.Aux.Fst kernelSize, Torch.Typed.Aux.Snd kernelSize
                     , Torch.Typed.Aux.Fst stride, Torch.Typed.Aux.Snd stride
                     , Torch.Typed.Aux.Fst padding, Torch.Typed.Aux.Snd padding
                     , channelSize
                     , inputSize0, inputSize1
                     , batchSize
                     ]
     , ConvSideCheck inputSize0 (Torch.Typed.Aux.Fst kernelSize) (Torch.Typed.Aux.Fst stride) (Torch.Typed.Aux.Fst padding) outputSize0
     , ConvSideCheck inputSize1 (Torch.Typed.Aux.Snd kernelSize) (Torch.Typed.Aux.Snd stride) (Torch.Typed.Aux.Snd padding) outputSize1
     )
  => Tensor device dtype '[batchSize, channelSize, inputSize0, inputSize1] -- ^ input
  -> Tensor device dtype '[batchSize, channelSize, outputSize0, outputSize1] -- ^ output
avgPool2d input = unsafePerformIO $ ATen.cast7
  ATen.Managed.avg_pool2d_tlllbbl
  input
  ([natValI @(Torch.Typed.Aux.Fst kernelSize), natValI @(Torch.Typed.Aux.Snd kernelSize)] :: [Int])
  ([natValI @(Torch.Typed.Aux.Fst stride),     natValI @(Torch.Typed.Aux.Snd stride)]     :: [Int])
  ([natValI @(Torch.Typed.Aux.Fst padding),    natValI @(Torch.Typed.Aux.Snd padding)]    :: [Int])
  False
  True
  (1 :: Int)

-- | averaged 3-D pooling
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
--
-- >>> t = avgPool3d @'(1,1,1) @'(1,1,1) @'(0,0,0) (ones :: CPUTensor 'D.Float '[1,3,4,5,6])
-- >>> shape t
-- [1,3,4,5,6]
-- >>> :t t
-- t :: Tensor '( 'D.CPU, 0) 'D.Float '[1, 3, 4, 5, 6]
avgPool3d
  :: forall kernelSize
            stride
            padding
            channelSize
            inputSize0 inputSize1 inputSize2
            batchSize
            outputSize0 outputSize1 outputSize2
            dtype
            device
   . ( All KnownNat '[ Fst3 kernelSize, Snd3 kernelSize, Trd3 kernelSize
                     , Fst3 stride, Snd3 stride, Trd3 stride
                     , Fst3 padding, Snd3 padding, Trd3 padding
                     , channelSize
                     , inputSize0, inputSize1, inputSize2
                     , batchSize
                     ]
     , ConvSideCheck inputSize0 (Fst3 kernelSize) (Fst3 stride) (Fst3 padding) outputSize0
     , ConvSideCheck inputSize1 (Snd3 kernelSize) (Snd3 stride) (Snd3 padding) outputSize1
     , ConvSideCheck inputSize2 (Trd3 kernelSize) (Trd3 stride) (Trd3 padding) outputSize2
     )
  => Tensor device dtype '[batchSize, channelSize, inputSize0, inputSize1, inputSize2] -- ^ input
  -> Tensor device dtype '[batchSize, channelSize, outputSize0, outputSize1, outputSize2] -- ^ output
avgPool3d input = unsafePerformIO $ ATen.cast7
  ATen.Managed.avg_pool3d_tlllbbl
  input
  ([ natValI @(Fst3 kernelSize)
   , natValI @(Snd3 kernelSize)
   , natValI @(Trd3 kernelSize)
   ] :: [Int]
  )
  ([natValI @(Fst3 stride), natValI @(Snd3 stride), natValI @(Trd3 stride)] :: [Int])
  ([natValI @(Fst3 padding), natValI @(Snd3 padding), natValI @(Trd3 padding)] :: [Int])
  False
  True
  (1 :: Int)

-- fractional_max_pool2d :: Tensor device dtype shape -> (Int,Int) -> (Int,Int) -> Tensor device dtype shape -> (Tensor device dtype shape,Tensor device dtype shape)
-- fractional_max_pool2d _input _kernel_size _output_size _random_samples = unsafePerformIO $ (ATen.cast4 ATen.Managed.fractional_max_pool2d_tllt) _input _kernel_size _output_size _random_samples

-- fractional_max_pool3d :: Tensor device dtype shape -> (Int,Int,Int) -> (Int,Int,Int) -> Tensor device dtype shape -> (Tensor device dtype shape,Tensor device dtype shape)
-- fractional_max_pool3d _input _kernel_size _output_size _random_samples = unsafePerformIO $ (ATen.cast4 ATen.Managed.fractional_max_pool3d_tllt) _input _kernel_size _output_size _random_samples

-- max_pool2d_with_indices :: Tensor device dtype shape -> (Int,Int) -> (Int,Int) -> (Int,Int) -> (Int,Int) -> Bool -> (Tensor device dtype shape,Tensor device dtype shape)
-- max_pool2d_with_indices _input _kernel_size _stride _padding _dilation _ceil_mode = unsafePerformIO $ (ATen.cast6 ATen.Managed.max_pool2d_with_indices_tllllb) _input _kernel_size _stride _padding _dilation _ceil_mode

-- max_pool3d_with_indices :: Tensor device dtype shape -> (Int,Int,Int) -> (Int,Int,Int) -> (Int,Int,Int) -> (Int,Int,Int) -> Bool -> (Tensor device dtype shape,Tensor device dtype shape)
-- max_pool3d_with_indices _input _kernel_size _stride _padding _dilation _ceil_mode = unsafePerformIO $ (ATen.cast6 ATen.Managed.max_pool3d_with_indices_tllllb) _input _kernel_size _stride _padding _dilation _ceil_mode

-- max_unpool2d :: Tensor device dtype shape -> Tensor device dtype shape -> (Int,Int) -> Tensor device dtype shape
-- max_unpool2d _input _indices _output_size = unsafePerformIO $ (ATen.cast3 ATen.Managed.max_unpool2d_ttl) _input _indices _output_size

-- max_unpool3d :: Tensor device dtype shape -> Tensor device dtype shape -> (Int,Int,Int) -> (Int,Int,Int) -> (Int,Int,Int) -> Tensor device dtype shape
-- max_unpool3d _input _indices _output_size _stride _padding = unsafePerformIO $ (ATen.cast5 ATen.Managed.max_unpool3d_ttlll) _input _indices _output_size _stride _padding

-- reflection_pad1d :: Tensor device dtype shape -> (Int,Int) -> Tensor device dtype shape
-- reflection_pad1d _input _padding = unsafePerformIO $ (ATen.cast2 ATen.Managed.reflection_pad1d_tl) _input _padding

-- reflection_pad2d :: Tensor device dtype shape -> (Int,Int,Int,Int) -> Tensor device dtype shape
-- reflection_pad2d _input _padding = unsafePerformIO $ (ATen.cast2 ATen.Managed.reflection_pad2d_tl) _input _padding

-- replication_pad1d :: Tensor device dtype shape -> (Int,Int) -> Tensor device dtype shape
-- replication_pad1d _input _padding = unsafePerformIO $ (ATen.cast2 ATen.Managed.replication_pad1d_tl) _input _padding

-- replication_pad2d :: Tensor device dtype shape -> (Int,Int,Int,Int) -> Tensor device dtype shape
-- replication_pad2d _input _padding = unsafePerformIO $ (ATen.cast2 ATen.Managed.replication_pad2d_tl) _input _padding

-- replication_pad3d :: Tensor device dtype shape -> (Int,Int,Int,Int,Int,Int) -> Tensor device dtype shape
-- replication_pad3d _input _padding = unsafePerformIO $ (ATen.cast2 ATen.Managed.replication_pad3d_tl) _input _padding

-- upsample_linear1d :: Tensor device dtype shape -> Int -> Bool -> Tensor device dtype shape
-- upsample_linear1d _input _output_size _align_corners = unsafePerformIO $ (ATen.cast3 ATen.Managed.upsample_linear1d_tlb) _input _output_size _align_corners


type family Upsample2dCheck shape h w where 
  Upsample2dCheck (b : c : w : h : '[]) h' w' = 
    If ( h <=? h') 
      (If (w <=? w') (b : c : w' : h' : '[]) 
        (TypeError (Text "Target width must be greater than current width!"))
      ) 
      (TypeError (Text "Target height must be greater than current height!"))
  Upsample2dCheck _ _ _ = TypeError (Text "Shape must be 4 dimensional!")

type Upsample2d shape h w = Upsample2dCheck shape h w

-- | Applies a 2D bilinear upsampling to an input signal composed of several input channels.
--
-- >>> upsample_bilinear2d @3 @5 False (ones :: CPUTensor 'D.Float '[2,3,2,2])
-- Tensor Float [2,3,3,5]
upsample_bilinear2d :: forall w h shape dtype device . 
  (KnownNat h, KnownNat w, All KnownNat shape) 
  => Bool -- ^ if True, the corner pixels of the input and output tensors are aligned, and thus preserving the values at those pixels. 
  -> Tensor device dtype shape 
  -> Tensor device dtype (Upsample2d shape h w)
upsample_bilinear2d _align_corners _input
  = unsafePerformIO $ (ATen.cast3 ATen.Managed.upsample_bilinear2d_tlb) _input ([w,h] :: [Int]) _align_corners
  where  
    w = natValI @w :: Int
    h = natValI @h :: Int

-- | Applies a 2D bicubic upsampling to an input signal composed of several input channels.
--
-- >>> upsample_bicubic2d @3 @5 False (ones :: CPUTensor 'D.Float '[2,3,2,2])
-- Tensor Float [2,3,3,5]
upsample_bicubic2d :: forall w h shape dtype device . 
  (KnownNat h, KnownNat w, All KnownNat shape) 
  => Bool 
  -> Tensor device dtype shape 
  -> Tensor device dtype (Upsample2d shape h w)
upsample_bicubic2d _align_corners _input = unsafePerformIO $ (ATen.cast3 ATen.Managed.upsample_bicubic2d_tlb) _input ([w,h] :: [Int]) _align_corners
  where 
    w = natValI @w :: Int
    h = natValI @h :: Int

-- upsample_trilinear3d :: Tensor device dtype shape -> (Int,Int,Int) -> Bool -> Tensor device dtype shape
-- upsample_trilinear3d _input _output_size _align_corners = unsafePerformIO $ (ATen.cast3 ATen.Managed.upsample_trilinear3d_tlb) _input _output_size _align_corners

-- upsample_nearest1d :: Tensor device dtype shape -> Int -> Tensor device dtype shape
-- upsample_nearest1d _input _output_size = unsafePerformIO $ (ATen.cast2 ATen.Managed.upsample_nearest1d_tl) _input _output_size

-- | Applies a 2D bicubic upsampling to an input signal composed of several input channels.
--
-- >>> upsample_nearest2d @3 @5 (ones :: CPUTensor 'D.Float '[2,3,2,2])
-- Tensor Float [2,3,3,5]
upsample_nearest2d :: forall w h shape dtype device . 
  (KnownNat h, KnownNat w, All KnownNat shape) 
  => Tensor device dtype shape 
  -> Tensor device dtype (Upsample2d shape h w)
upsample_nearest2d _input = unsafePerformIO $ (ATen.cast2 ATen.Managed.upsample_nearest2d_tl) _input ([w,h] :: [Int])
  where
    w = natValI @w :: Int
    h = natValI @h :: Int

-- upsample_nearest3d :: Tensor device dtype shape -> (Int,Int,Int) -> Tensor device dtype shape
-- upsample_nearest3d _input _output_size = unsafePerformIO $ (ATen.cast2 ATen.Managed.upsample_nearest3d_tl) _input _output_size

-- conv_dilated2d :: Tensor device dtype shape -> Tensor device dtype shape -> (Int,Int) -> Tensor device dtype shape -> (Int,Int) -> (Int,Int) -> (Int,Int) -> Tensor device dtype shape
-- conv_dilated2d _input _weight _kernel_size _bias _stride _padding _dilation = unsafePerformIO $ (ATen.cast7 ATen.Managed.conv_dilated2d_ttltlll) _input _weight _kernel_size _bias _stride _padding _dilation

-- conv_dilated3d :: Tensor device dtype shape -> Tensor device dtype shape -> (Int,Int,Int) -> Tensor device dtype shape -> (Int,Int,Int) -> (Int,Int,Int) -> (Int,Int,Int) -> Tensor device dtype shape
-- conv_dilated3d _input _weight _kernel_size _bias _stride _padding _dilation = unsafePerformIO $ (ATen.cast7 ATen.Managed.conv_dilated3d_ttltlll) _input _weight _kernel_size _bias _stride _padding _dilation

-- col2im :: Tensor device dtype shape -> (Int,Int) -> (Int,Int) -> (Int,Int) -> (Int,Int) -> (Int,Int) -> Tensor device dtype shape
-- col2im _input _output_size _kernel_size _dilation _padding _stride = unsafePerformIO $ (ATen.cast6 ATen.Managed.col2im_tlllll) _input _output_size _kernel_size _dilation _padding _stride

-- im2col :: Tensor device dtype shape -> (Int,Int) -> (Int,Int) -> (Int,Int) -> (Int,Int) -> Tensor device dtype shape
-- im2col _input _kernel_size _dilation _padding _stride = unsafePerformIO $ (ATen.cast5 ATen.Managed.im2col_tllll) _input _kernel_size _dilation _padding _stride