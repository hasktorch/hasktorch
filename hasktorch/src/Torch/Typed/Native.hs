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

module Torch.Typed.Native where

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
                                                )
import           Data.Finite
import qualified Data.Int                      as I
import           Data.HList
import           Data.Kind                      ( Constraint
                                                , Type
                                                )
import           Data.Maybe
import           Data.Proxy
import           Data.Reflection
import           Control.Arrow                  ( (&&&) )
import           GHC.Natural                    ( Natural )
import           GHC.TypeLits
import           GHC.TypeLits.Extra
import           System.IO.Unsafe
import           Data.Singletons.Prelude.List   ( Product )

import           Foreign.ForeignPtr
import qualified ATen.Managed.Native           as ATen
import qualified ATen.Managed.Type.Tensor      as ATen
import qualified ATen.Managed.Type.Scalar      as ATen
import qualified ATen.Managed.Type.Tuple       as ATen
import qualified ATen.Const                    as ATen
import qualified ATen.Type                     as ATen
import qualified ATen.Managed.Cast
import           ATen.Cast

import qualified Torch.Tensor                  as D
import qualified Torch.TensorFactories         as D
import qualified Torch.TensorOptions           as D
import qualified Torch.DType                   as D
import qualified Torch.Device                  as D
import qualified Torch.Scalar                  as D
import           Torch.Functions                ( Reduction(..)
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
sumAll input = unsafePerformIO $ cast1 ATen.sum_t input

-- | sumDim
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
sumDim input = unsafePerformIO $ cast2 ATen.sum_tl input (natValI @d)

-- | abs
-- >>> dtype &&& shape $ abs (ones :: CPUTensor 'D.Float '[2,2])
-- (Float,[2,2])
abs
  :: forall shape dtype device
   . (DTypeIsNotHalf device dtype, DTypeIsNotBool device dtype)
  => Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
abs input = unsafePerformIO $ cast1 ATen.abs_t input

-- | ceil
-- >>> dtype &&& shape $ ceil (ones :: CPUTensor 'D.Float '[2,2])
-- (Float,[2,2])
ceil
  :: forall shape dtype device
   . (StandardFloatingPointDTypeValidation device dtype)
  => Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
ceil input = unsafePerformIO $ cast1 ATen.ceil_t input

-- | floor
-- >>> dtype &&& shape $ floor (ones :: CPUTensor 'D.Float '[2,2])
-- (Float,[2,2])
floor
  :: forall shape dtype device
   . (StandardFloatingPointDTypeValidation device dtype)
  => Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
floor input = unsafePerformIO $ cast1 ATen.floor_t input

-- TODO: better error messages, "Couldn't match type ‘'False’ with ‘'True’" isn't great
type family AllDimsPositive (shape :: [Nat]) :: Constraint where
  AllDimsPositive '[] = ()
  AllDimsPositive (x ': xs) = (1 <= x, AllDimsPositive xs)

type family AggregationDTypeIsValid (device :: (D.DeviceType, Nat)) (dtype :: D.DType) :: Constraint where
  AggregationDTypeIsValid '( 'D.CPU, 0)    dtype = DTypeIsNotHalf '( 'D.CPU, 0) dtype
  AggregationDTypeIsValid '( 'D.CUDA, _)   dtype = ()
  AggregationDTypeIsValid '(deviceType, _) dtype = UnsupportedDTypeForDevice deviceType dtype

-- | min
-- >>> dtype &&& shape $ min (ones :: CPUTensor 'D.Float '[2,2])
-- (Float,[])
min
  :: forall shape dtype device
   . ( AggregationDTypeIsValid device dtype
     , AllDimsPositive shape
     )
  => Tensor device dtype shape -- ^ input
  -> Tensor device dtype '[] -- ^ output
min input = unsafePerformIO $ cast1 ATen.min_t input

-- | max
-- >>> dtype &&& shape $ max (ones :: CPUTensor 'D.Float '[2,2])
-- (Float,[])
max
  :: forall shape dtype device
  . ( AggregationDTypeIsValid device dtype
    , AllDimsPositive shape
    )
  => Tensor device dtype shape -- ^ input
  -> Tensor device dtype '[] -- ^ output
max input = unsafePerformIO $ cast1 ATen.max_t input

-- | median
-- >>> dtype &&& shape $ median (ones :: CPUTensor 'D.Float '[2,2])
-- (Float,[])
median
  :: forall shape dtype device
  . ( AggregationDTypeIsValid device dtype
    , AllDimsPositive shape
    )
  => Tensor device dtype shape -- ^ input
  -> Tensor device dtype '[] -- ^ output
median input = unsafePerformIO $ cast1 ATen.median_t input

-- | cmul
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
cmul
  :: forall a shape dtype device
   . D.Scalar a
  => a -- ^ scalar input
  -> Tensor device dtype shape -- ^ tensor input
  -> Tensor device dtype shape -- ^ output
cmul a input = unsafePerformIO $ cast2 ATen.mul_ts input a

-- | erf
-- >>> dtype &&& shape $ erf (ones :: CPUTensor 'D.Float '[3,2])
-- (Float,[3,2])
erf
  :: forall shape dtype device
   . (StandardFloatingPointDTypeValidation device dtype)
  => Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
erf input = unsafePerformIO $ cast1 ATen.erf_t input

-- | exp
-- >>> dtype &&& shape $ exp (ones :: CPUTensor 'D.Float '[3,2])
-- (Float,[3,2])
exp
  :: forall shape dtype device
   . (StandardFloatingPointDTypeValidation device dtype)
  => Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
exp input = unsafePerformIO $ cast1 ATen.exp_t input

-- | log1p
-- >>> dtype &&& shape $ log1p (ones :: CPUTensor 'D.Float '[3,2])
-- (Float,[3,2])
log1p
  :: forall shape dtype device
   . (StandardFloatingPointDTypeValidation device dtype)
  => Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
log1p input = unsafePerformIO $ cast1 ATen.log1p_t input

-- | log2
-- >>> dtype &&& shape $ log2 (ones :: CPUTensor 'D.Float '[3,2])
-- (Float,[3,2])
log2
  :: forall shape dtype device
   . (StandardFloatingPointDTypeValidation device dtype)
  => Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
log2 input = unsafePerformIO $ cast1 ATen.log2_t input

-- | log10
-- >>> dtype &&& shape $ log10 (ones :: CPUTensor 'D.Float '[3,2])
-- (Float,[3,2])
log10
  :: forall shape dtype device
   . (StandardFloatingPointDTypeValidation device dtype)
  => Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
log10 input = unsafePerformIO $ cast1 ATen.log10_t input

-- | pow
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
-- >>> dtype &&& shape $ pow 2 (ones :: CPUTensor 'D.Float '[3,2])
-- (Float,[3,2])
pow
  :: forall a shape dtype device
   . D.Scalar a
  => a -- ^ power
  -> Tensor device dtype shape -- ^ input tensor
  -> Tensor device dtype shape -- ^ output tensor
pow a input = unsafePerformIO $ cast2 ATen.pow_ts input a

-- | relu activation function
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
-- >>> dtype &&& shape $ relu (ones :: CPUTensor 'D.Float '[3,2])
-- (Float,[3,2])
relu
  :: forall shape dtype device
   . (StandardFloatingPointDTypeValidation device dtype)
  => Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
relu input = unsafePerformIO $ cast1 ATen.relu_t input

-- | selu
-- >>> dtype &&& shape $ selu (ones :: CPUTensor 'D.Float '[3,2])
-- (Float,[3,2])
selu
  :: forall shape dtype device
   . (StandardFloatingPointDTypeValidation device dtype)
  => Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
selu input = unsafePerformIO $ cast1 ATen.selu_t input

-- | sigmoid
-- >>> dtype &&& shape $ sigmoid (ones :: CPUTensor 'D.Float '[3,2])
-- (Float,[3,2])
sigmoid
  :: forall shape dtype device
   . (StandardFloatingPointDTypeValidation device dtype)
  => Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
sigmoid input = unsafePerformIO $ cast1 ATen.sigmoid_t input

-- | sin
-- >>> dtype &&& shape $ sin (ones :: CPUTensor 'D.Float '[3,2])
-- (Float,[3,2])
sin
  :: forall shape dtype device
   . (StandardFloatingPointDTypeValidation device dtype)
  => Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
sin input = unsafePerformIO $ cast1 ATen.sin_t input

-- | sinh
-- >>> dtype &&& shape $ sinh (ones :: CPUTensor 'D.Float '[3,2])
-- (Float,[3,2])
sinh
  :: forall shape dtype device
   . (StandardFloatingPointDTypeValidation device dtype)
  => Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
sinh input = unsafePerformIO $ cast1 ATen.sinh_t input

-- | cos
-- >>> dtype &&& shape $ cos (ones :: CPUTensor 'D.Float '[3,2])
-- (Float,[3,2])
cos
  :: forall shape dtype device
   . (StandardFloatingPointDTypeValidation device dtype)
  => Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
cos input = unsafePerformIO $ cast1 ATen.cos_t input

-- | sqrt
sqrt
  :: forall shape dtype device
   . (StandardFloatingPointDTypeValidation device dtype)
  => Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
sqrt input = unsafePerformIO $ cast1 ATen.sqrt_t input

-- | tanh
tanh 
  :: forall shape dtype device
   . (StandardFloatingPointDTypeValidation device dtype)
  => Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
tanh input = unsafePerformIO $ cast1 ATen.tanh_t input

-- | toDType
-- TODO: since we have Torch.Typed.Tensor.toType, do we need this one?
-- >>> dtype &&& shape $ toDType @'D.Double (ones :: CPUTensor 'D.Float '[2,2])
-- (Double,[2,2])
toDType
  :: forall dtype' dtype shape device
   . (KnownDType dtype')
  => Tensor device dtype  shape -- ^ input
  -> Tensor device dtype' shape -- ^ output
toDType input = unsafePerformIO $ cast4 ATen.tensor_to_sbb input (dtypeVal @dtype') False False

type family SqueezeAll (shape :: [Nat]) :: [Nat] where
  SqueezeAll '[] = '[]
  SqueezeAll (1: xs) = SqueezeAll xs
  SqueezeAll (x: xs) = x ': SqueezeAll xs

-- | squeezeAll
-- >>> dtype &&& shape $ squeezeAll (ones :: CPUTensor 'D.Float '[2,1,2,1,2])
-- (Float,[2,2,2])
squeezeAll
  :: forall shape shape' dtype device
   . (shape' ~ SqueezeAll shape)
  => Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape' -- ^ output
squeezeAll input = unsafePerformIO $ cast1 ATen.squeeze_t input

-- | ConditionalReduction
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
binaryCrossEntropy weight prediction target = unsafePerformIO $ cast4
  ATen.binary_cross_entropy_tttl
  prediction
  target
  weight
  (reductionVal @reduction)

-- | mseLoss
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
mseLoss prediction target = unsafePerformIO $ cast3
  ATen.mse_loss_ttl
  prediction
  target
  (reductionVal @reduction)

-- | softmax
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
  $ cast3 ATen.softmax_tls input (natValI @dim) (dtypeVal @dtype)

-- | logSoftmax
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
  $ cast3 ATen.log_softmax_tls input (natValI @dim) (dtypeVal @dtype)

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

-- | inverse
-- TODO: if rank < n for any tensors in the batch, then this will not work. we can't decide this statically, but we should prevent runtime errors. therefore, return Maybe?
-- >>> t <- randn :: IO (CPUTensor 'D.Float '[3,2,2])
-- >>> dtype &&& shape $ inverse t
-- (Float,[3,2,2])
-- >>> t <- randn :: IO (CPUTensor 'D.Float '[2,2])
-- >>> dtype &&& shape $ inverse t
-- (Float,[2,2])
inverse
  :: forall shape shape' dtype device
   . ( shape' ~ Square shape
     , StandardFloatingPointDTypeValidation device dtype
     )
  => Tensor device dtype shape -- ^ inverse
  -> Tensor device dtype shape' -- ^ output
inverse input = unsafePerformIO $ cast1 ATen.inverse_t input

type family SymeigDTypeIsValid (device :: (D.DeviceType, Nat)) (dtype :: D.DType) :: Constraint where
  SymeigDTypeIsValid '( 'D.CPU, 0)            dtype = ( DTypeIsFloatingPoint '( 'D.CPU, 0) dtype
                                                      , DTypeIsNotHalf '( 'D.CPU, 0) dtype
                                                      )
  SymeigDTypeIsValid '( 'D.CUDA, deviceIndex) dtype = ( DTypeIsFloatingPoint '( 'D.CUDA, deviceIndex) dtype
                                                      , DTypeIsNotHalf '( 'D.CUDA, deviceIndex) dtype
                                                      )
  SymeigDTypeIsValid '(deviceType, _)         dtype = UnsupportedDTypeForDevice deviceType dtype

-- | symeig
-- TODO: split this function into two, one that calculates the eigenvectors and another that does not?
-- >>> t <- rand :: IO (CPUTensor 'D.Float '[3,2,2])
-- >>> (eigenVals,eigenVecs) = symeig True Upper t
-- >>> dtype &&& shape $ eigenVals
-- (Float,[3,2])
-- >>> :t eigenVals
-- eigenVals :: Tensor '( 'D.CPU, 0) 'D.Float '[3, 2]
-- >>> dtype &&& shape $ eigenVecs
-- (Float,[3,2,2])
-- >>> :t eigenVecs
-- eigenVecs :: Tensor '( 'D.CPU, 0) 'D.Float '[3, 2, 2]
-- >>> (eigenVals,eigenVecs) = symeig False Upper t
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
  => Bool -- ^ whether or not to calculate eigenvectors
  -> Tri -- ^ upper or lower triagonal
  -> Tensor device dtype shape -- ^ input
  -> ( Tensor device dtype shape'
     , Tensor device dtype shape''
     ) -- ^ eigenvalues and eigenvectors
symeig eigenvectors upper input = unsafePerformIO
  $ cast3 ATen.symeig_tbb input eigenvectors boolUpper
  where boolUpper = isUpper upper

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
  unsafePerformIO $ cast2 ATen.eig_tb input (enableEigenVectors @eigenvectors)

-- svd :: Tensor device dtype shape -> Bool -> Bool -> (Tensor device dtype shape, Tensor device dtype shape, Tensor device dtype shape)
-- svd t some compute_uv = unsafePerformIO $ (cast3 ATen.svd_tbb) t some compute_uv

type family CholeskyDTypeIsValid (device :: (D.DeviceType, Nat)) (dtype :: D.DType) :: Constraint where
  CholeskyDTypeIsValid '( 'D.CPU, 0)            dtype = ( DTypeIsFloatingPoint '( 'D.CPU, 0) dtype
                                                        , DTypeIsNotHalf '( 'D.CPU, 0) dtype
                                                        )
  CholeskyDTypeIsValid '( 'D.CUDA, deviceIndex) dtype = ( DTypeIsFloatingPoint '( 'D.CUDA, deviceIndex) dtype
                                                        , DTypeIsNotHalf '( 'D.CUDA, deviceIndex) dtype
                                                        )
  CholeskyDTypeIsValid '(deviceType, _)         dtype = UnsupportedDTypeForDevice deviceType dtype

-- | cholesky
-- >>> t <- rand :: IO (CPUTensor 'D.Float '[2,2])
-- >>> c = cholesky Upper (t `matmul` transpose2D t)
-- >>> dtype &&& shape $ c
-- (Float,[2,2])
-- >>> :t c
-- c :: Tensor '( 'D.CPU, 0) 'D.Float '[2, 2]
cholesky
  :: forall shape shape' dtype device
   . ( shape' ~ Square shape
     , CholeskyDTypeIsValid device dtype
     )
  => Tri -- ^ upper or lower triangular part of the matrix
  -> Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape' -- ^ output
cholesky upper input = unsafePerformIO $ cast2 ATen.cholesky_tb input boolUpper
  where boolUpper = isUpper upper

-- cholesky_solve :: Tensor device dtype shape -> Tensor device dtype shape -> Tri -> Tensor device dtype shape
-- cholesky_solve t1 t2 upper = unsafePerformIO $ (cast3 ATen.cholesky_solve_ttb) t1 t2 boolUpper
--   where boolUpper = isUpper upper

type family SolveDTypeIsValid (device :: (D.DeviceType, Nat)) (dtype :: D.DType) :: Constraint where
  SolveDTypeIsValid '( 'D.CPU, 0)            dtype = ( DTypeIsFloatingPoint '( 'D.CPU, 0) dtype
                                                     , DTypeIsNotHalf '( 'D.CPU, 0) dtype
                                                     )
  SolveDTypeIsValid '( 'D.CUDA, deviceIndex) dtype = ( DTypeIsFloatingPoint '( 'D.CUDA, deviceIndex) dtype
                                                     , DTypeIsNotHalf '( 'D.CUDA, deviceIndex) dtype
                                                     )
  SolveDTypeIsValid '(deviceType, _)         dtype = UnsupportedDTypeForDevice deviceType dtype

-- | solve the system of linear equations represented by `a c = b` and return the LU decomposition of `a`
-- >>> a <- rand :: IO (CPUTensor 'D.Float '[10,10])
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
  => Tensor device dtype m_k -- ^ b
  -> Tensor device dtype m_m -- ^ a
  -> ( Tensor device dtype m_k
     , Tensor device dtype m_m
     ) -- ^ c and lu
solve b a = unsafePerformIO $ cast2 ATen.solve_tt b a

-- | choleskyInverse
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
-- >>> dtype &&& shape $ choleskyInverse Upper (ones :: CPUTensor 'D.Float '[3,3])
-- (Float,[3,3])
choleskyInverse
  :: forall shape shape' dtype device
   . (shape' ~ Square shape)
  => Tri
  -> Tensor device dtype shape
  -> Tensor device dtype shape'
choleskyInverse upper input =
  unsafePerformIO $ cast2 ATen.cholesky_inverse_tb input boolUpper
 where boolUpper = isUpper upper

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
geqrf input = unsafePerformIO $ cast1 ATen.geqrf_t input

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
orgqr a tau = unsafePerformIO $ cast2 ATen.orgqr_tt a tau

-- | sign
-- works for all dtypes
-- >>> dtype &&& shape $ sign (ones :: CPUTensor 'D.Float '[3,2])
-- (Float,[3,2])
sign
  :: forall shape dtype device
   . Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
sign input = unsafePerformIO $ cast1 ATen.sign_t input

type family SetValue (shape :: [Nat]) (i :: Nat) (j :: Nat)  :: [Nat] where
  SetValue '[]     _ _ = '[]
  SetValue (x: xs) 0 j = j: xs
  SetValue (x: xs) i j = x: SetValue xs (i-1) j

type family GetValue (shape :: [Nat]) (i :: Nat) :: Nat where
  GetValue '[]     _ = TypeError (Text "Can not find a element in the list.")
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
-- See ../../../../deps/pytorch/aten/src/ATen/native/TensorShape.cpp.
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
  unsafePerformIO $ cast3 ATen.transpose_tll input (natValI @n) (natValI @m)

-- | transpose2d, special case for a 2D tensor
-- >>> dtype &&& shape $ transpose2D (ones :: CPUTensor 'D.Float '[3,2])
-- (Float,[2,3])
transpose2D
  :: forall (i :: Nat) (j :: Nat) dtype device
   . Tensor device dtype '[i, j] -- ^ input
  -> Tensor device dtype '[j, i] -- ^ output
transpose2D = transpose @0 @1

-- diag :: Tensor device dtype shape -> Int -> Tensor device dtype shape
-- diag t index = unsafePerformIO $ (cast2 ATen.tensor_diag_l) t index

-- | all
-- See https://pytorch.org/docs/stable/tensors.html#torch.BoolTensor.all.
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
all input = unsafePerformIO $ cast1 ATen.all_t input

-- | any
-- See https://pytorch.org/docs/stable/tensors.html#torch.BoolTensor.any.
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
any input = unsafePerformIO $ cast1 ATen.any_t input

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

-- | all'
-- See https://pytorch.org/docs/stable/tensors.html#torch.BoolTensor.all.
-- >>> t = fromJust [[True, True], [True, False], [True, True], [True, True]] :: CPUTensor 'D.Bool '[4, 2]
--
-- >>> dtype &&& shape &&& (\t' -> D.asValue (toDynamic t') :: [Bool]) $ all' @1 @DropDim t
-- (Bool,([4],[True,False,True,True]))
--
-- >>> dtype &&& shape &&& (\t' -> D.asValue (toDynamic t') :: [[Bool]]) $ all' @1 @KeepDim t
-- (Bool,([4,1],[[True],[False],[True],[True]]))
--
-- >>> dtype &&& shape &&& (\t' -> D.asValue (toDynamic t') :: [Bool]) $ all' @0 @DropDim t
-- (Bool,([2],[True,False]))
--
-- >>> dtype &&& shape &&& (\t' -> D.asValue (toDynamic t') :: [[Bool]]) $ all' @0 @KeepDim t
-- (Bool,([1,2],[[True,False]]))
all'
  :: forall dim keepOrDropDim shape' shape device
   . ( KnownNat dim
     , KnownKeepOrDropDim keepOrDropDim
     , shape' ~ ConditionalDropDimension shape dim keepOrDropDim
     )
  => Tensor device 'D.Bool shape -- ^ input
  -> Tensor device 'D.Bool shape' -- ^ output
all' input = unsafePerformIO
  $ cast3 ATen.all_tlb input (natValI @dim) (keepOrDropDimVal @keepOrDropDim)

-- | any'
-- See https://pytorch.org/docs/stable/tensors.html#torch.BoolTensor.any.
-- >>> t = fromJust [[True, True], [True, False], [True, True], [True, True]] :: CPUTensor 'D.Bool '[4, 2]
--
-- >>> dtype &&& shape &&& (\t' -> D.asValue (toDynamic t') :: [Bool]) $ any' @1 @DropDim t
-- (Bool,([4],[True,True,True,True]))
--
-- >>> dtype &&& shape &&& (\t' -> D.asValue (toDynamic t') :: [[Bool]]) $ any' @1 @KeepDim t
-- (Bool,([4,1],[[True],[True],[True],[True]]))
--
-- >>> dtype &&& shape &&& (\t' -> D.asValue (toDynamic t') :: [Bool]) $ any' @0 @DropDim t
-- (Bool,([2],[True,True]))
--
-- >>> dtype &&& shape &&& (\t' -> D.asValue (toDynamic t') :: [[Bool]]) $ any' @0 @KeepDim t
-- (Bool,([1,2],[[True,True]]))
any'
  :: forall dim keepOrDropDim shape' shape device
   . ( KnownNat dim
     , KnownKeepOrDropDim keepOrDropDim
     , shape' ~ ConditionalDropDimension shape dim keepOrDropDim
     )
  => Tensor device 'D.Bool shape -- ^ input
  -> Tensor device 'D.Bool shape' -- ^ output
any' input = unsafePerformIO $ cast3 ATen.any_tlb input (natValI @dim) (keepOrDropDimVal @keepOrDropDim)

-- | dropout
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
-- TODO: get rid of IO by exposing the RNG state
-- TODO: can we use D.Scalar for the dropout probability?
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
dropout p train input = cast3 ATen.dropout_tdb input p train

-- | featureDropout
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
-- TODO: why not IO?
-- TODO: can we use D.Scalar for the dropout probability?
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
  unsafePerformIO $ cast3 ATen.feature_dropout_tdb input p train

-- | alphaDropout
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
-- TODO: why not IO?
-- TODO: can we use D.Scalar for the dropout probability?
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
  unsafePerformIO $ cast3 ATen.alpha_dropout_tdb input p train

-- | featureAlphaDropout
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
-- TODO: why not IO?
-- TODO: can we use D.Scalar for the dropout probability?
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
  unsafePerformIO $ cast3 ATen.feature_alpha_dropout_tdb input p train

-- | acos
-- >>> dtype &&& shape $ acos (ones :: CPUTensor 'D.Float '[3,2])
-- (Float,[3,2])
acos
  :: forall shape dtype device
   . (StandardFloatingPointDTypeValidation device dtype)
  => Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
acos input = unsafePerformIO $ cast1 ATen.acos_t input

-- | avgPool1d
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
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
avgPool1d input = unsafePerformIO $ cast6 ATen.avg_pool1d_tlllbb
  input
  (natValI @kernelSize)
  (natValI @stride)
  (natValI @padding)
  False
  True

-- | adaptiveAvgPool1d
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
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
  $ cast2 ATen.adaptive_avg_pool1d_tl input (natValI @outputSize)

-- | adaptiveMaxPool1d
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
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
  $ cast2 ATen.adaptive_max_pool1d_tl input (natValI @outputSize)

-- | addmv
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
-- TODO: can we use D.Scalar for beta and alpha?
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
addmv beta alpha mat vec input = unsafePerformIO $ cast5 ATen.addmv_tttss input mat vec beta alpha

-- | addr
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
-- TODO: can we use D.Scalar for beta and alpha?
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
addr beta alpha vec1 vec2 input = unsafePerformIO $ cast5 ATen.addr_tttss input vec1 vec2 beta alpha

-- affine_grid_generator :: Tensor device dtype shape -> [Int] -> Tensor device dtype shape
-- affine_grid_generator _theta _size = unsafePerformIO $ (cast2 ATen.affine_grid_generator_tl) _theta _size

-- | allclose
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
  unsafePerformIO $ cast5 ATen.allclose_ttddb input other rtol atol equalNaN

-- | argmax
-- See https://pytorch.org/docs/stable/torch.html#torch.argmax.
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
argmax input = unsafePerformIO $ cast3 ATen.argmax_tlb
                                       input
                                       (natValI @dim)
                                       (keepOrDropDimVal @keepOrDropDim)

-- | argmin
-- See https://pytorch.org/docs/stable/torch.html#torch.argmin.
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
argmin input = unsafePerformIO $ cast3 ATen.argmin_tlb
                                       input
                                       (natValI @dim)
                                       (keepOrDropDimVal @keepOrDropDim)

-- as_strided :: Tensor device dtype shape -> [Int] -> [Int] -> Int -> Tensor device dtype shape
-- as_strided _input _size _stride _storage_offset = unsafePerformIO $ (cast4 ATen.as_strided_tlll) _input _size _stride _storage_offset

-- | asin
-- >>> dtype &&& shape $ asin (ones :: CPUTensor 'D.Float '[3,2])
-- (Float,[3,2])
asin
  :: forall shape dtype device
   . (StandardFloatingPointDTypeValidation device dtype)
  => Tensor device dtype shape
  -> Tensor device dtype shape
asin input = unsafePerformIO $ cast1 ATen.asin_t input

-- | atan
-- >>> dtype &&& shape $ atan (ones :: CPUTensor 'D.Float '[3,2])
-- (Float,[3,2])
atan
  :: forall shape dtype device
   . (StandardFloatingPointDTypeValidation device dtype)
  => Tensor device dtype shape
  -> Tensor device dtype shape
atan input = unsafePerformIO $ cast1 ATen.atan_t input

-- | baddbmm
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
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
baddbmm beta alpha batch1 batch2 input = unsafePerformIO $ cast5 ATen.baddbmm_tttss input batch1 batch2 beta alpha

-- batch_norm :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Bool -> Double -> Double -> Bool -> Tensor device dtype shape
-- batch_norm _input _weight _bias _running_mean _running_var _training _momentum _eps _cudnn_enabled = unsafePerformIO $ (cast9 ATen.batch_norm_tttttbddb) _input _weight _bias _running_mean _running_var _training _momentum _eps _cudnn_enabled

-- bilinear :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape
-- bilinear _input1 _input2 _weight _bias = unsafePerformIO $ (cast4 ATen.bilinear_tttt) _input1 _input2 _weight _bias

-- binary_cross_entropy_with_logits :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Int -> Tensor device dtype shape
-- binary_cross_entropy_with_logits _input _target _weight _pos_weight _reduction = unsafePerformIO $ (cast5 ATen.binary_cross_entropy_with_logits_ttttl) _input _target _weight _pos_weight _reduction

-- bincount :: Tensor device dtype shape -> Tensor device dtype shape -> Int -> Tensor device dtype shape
-- bincount _input _weights _minlength = unsafePerformIO $ (cast3 ATen.bincount_ttl) _input _weights _minlength

-- | bitwise_not
-- >>> dtype &&& shape $ bitwiseNot (ones :: CPUTensor 'D.Bool [3,3])
-- (Bool,[3,3])
bitwiseNot
  :: forall shape device
   . Tensor device 'D.Bool shape
  -> Tensor device 'D.Bool shape
bitwiseNot input = unsafePerformIO $ cast1 ATen.bitwise_not_t input

-- | batched matrix multiplication
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
-- >>> dtype &&& shape $ bmm (ones :: CPUTensor 'D.Float '[5,3,2]) (zeros :: CPUTensor 'D.Float '[5,2,4])
-- (Float,[5,3,4])
bmm
  :: forall batchSize n m k dtype device
   . Tensor device dtype '[batchSize, n, k] -- ^ input
  -> Tensor device dtype '[batchSize, k, m] -- ^ other input
  -> Tensor device dtype '[batchSize, n, m] -- ^ output
bmm input other = unsafePerformIO $ cast2 ATen.bmm_tt input other

-- | BroadcastTensorsImpl
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
     , HFoldrM IO TensorListFold [D.ATenTensor] tensors
     , Apply TensorListUnfold [D.ATenTensor] (HUnfoldMRes IO [D.ATenTensor] tensors)
     , HUnfoldM IO TensorListUnfold (HUnfoldMRes IO [D.ATenTensor] tensors) tensors
     , HFoldrM IO TensorListFold [D.ATenTensor] tensors'
     , Apply TensorListUnfold [D.ATenTensor] (HUnfoldMRes IO [D.ATenTensor] tensors')
     , HUnfoldM IO TensorListUnfold (HUnfoldMRes IO [D.ATenTensor] tensors') tensors'
     )
  => HList tensors -- ^ input list of tensors
  -> HList tensors' -- ^ output list of tensors
broadcastTensors tensors = unsafePerformIO $ cast1 ATen.broadcast_tensors_l tensors

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
     , HFoldrM IO TensorListFold [D.ATenTensor] tensors
     , Apply TensorListUnfold [D.ATenTensor] (HUnfoldMRes IO [D.ATenTensor] tensors)
     , HUnfoldM IO TensorListUnfold (HUnfoldMRes IO [D.ATenTensor] tensors) tensors
     )
  => HList tensors -- ^ input list of tensors
  -> Tensor device dtype shape -- ^ output tensor
cat tensors = unsafePerformIO $ cast2 ATen.cat_ll tensors (natValI @dim :: Int)

-- chain_matmul :: [Tensor device dtype shape] -> Tensor device dtype shape
-- chain_matmul _matrices = unsafePerformIO $ (cast1 ATen.chain_matmul_l) _matrices

type family ChunkImpl (chunkShapes :: Maybe [[Nat]]) (dtype :: D.DType) (device :: (D.DeviceType, Nat)) :: Maybe a where
  ChunkImpl (Just '[])               _     _     = Just '[]
  ChunkImpl (Just (shape ': shapes)) dtype device = AppendToMaybe (Tensor device dtype shape) (ChunkImpl (Just shapes) dtype device)
  ChunkImpl Nothing                  _     _     = Nothing

type family ChunkCheck (shape :: [Nat]) (dim :: Nat) (result :: Maybe a) :: a where
  ChunkCheck shape dim Nothing = DimOutOfBound shape dim
  ChunkCheck _ _ (Just result) = result

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
-- >>> :type chunk @3 @1 (ones :: CPUTensor 'D.Float '[2, 2])
-- chunk @3 @1 (ones :: CPUTensor 'D.Float '[2, 2])
--   :: HList
--        '[Tensor '( 'D.CPU, 0) 'D.Float '[2, 1],
--          Tensor '( 'D.CPU, 0) 'D.Float '[2, 1]]
-- >>> t0 :. t1 :. HNil = chunk @3 @1 (ones :: CPUTensor 'D.Float '[2, 2])
-- >>> dtype &&& shape $ t0
-- (Float,[2,1])
-- >>> dtype &&& shape $ t1
-- (Float,[2,1])
-- >>> :type chunk @3 @1 (ones :: CPUTensor 'D.Float '[1, 0, 3])
-- chunk @3 @1 (ones :: CPUTensor 'D.Float '[1, 0, 3])
--   :: HList
--        '[Tensor '( 'D.CPU, 0) 'D.Float '[1, 0, 3],
--          Tensor '( 'D.CPU, 0) 'D.Float '[1, 0, 3],
--          Tensor '( 'D.CPU, 0) 'D.Float '[1, 0, 3]]
-- >>> t0 :. t1 :. t2 :. HNil = chunk @3 @1 (ones :: CPUTensor 'D.Float '[1, 0, 3])
-- >>> dtype &&& shape $ t0
-- (Float,[1,0,3])
-- >>> dtype &&& shape $ t1
-- (Float,[1,0,3])
-- >>> dtype &&& shape $ t2
-- (Float,[1,0,3])
-- >>> :type chunk @6 @0 (ones :: CPUTensor 'D.Float '[19, 4])
-- chunk @6 @0 (ones :: CPUTensor 'D.Float '[19, 4])
--   :: HList
--        '[Tensor '( 'D.CPU, 0) 'D.Float '[4, 4],
--          Tensor '( 'D.CPU, 0) 'D.Float '[4, 4],
--          Tensor '( 'D.CPU, 0) 'D.Float '[4, 4],
--          Tensor '( 'D.CPU, 0) 'D.Float '[4, 4],
--          Tensor '( 'D.CPU, 0) 'D.Float '[3, 4]]
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
     , HFoldrM IO TensorListFold [D.ATenTensor] tensorChunks
     , Apply TensorListUnfold [D.ATenTensor] (HUnfoldMRes IO [D.ATenTensor] tensorChunks)
     , HUnfoldM IO TensorListUnfold (HUnfoldMRes IO [D.ATenTensor] tensorChunks) tensorChunks
     )
  => Tensor device dtype shape -- ^ input tensor
  -> HList tensorChunks -- ^ output list of tensors
chunk input = unsafePerformIO
  $ cast3 ATen.chunk_tll input (natValI @chunks :: Int) (natValI @dim :: Int)

-- | clamp
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
-- TODO: can we use D.Scalar for the minimum and maximum values?
-- >>> dtype &&& shape $ clamp 0 1 (ones :: CPUTensor 'D.Float '[3,2])
-- (Float,[3,2])
clamp
  :: forall shape dtype device
   . Float -- ^ minimum value
  -> Float -- ^ maximum value
  -> Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
clamp min max input = unsafePerformIO $ cast3 ATen.clamp_tss input min max

-- | clampMax
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
-- TODO: can we use D.Scalar for the maximum value?
-- >>> dtype &&& shape $ clampMax 1 (ones :: CPUTensor 'D.Float '[3,2])
-- (Float,[3,2])
clampMax
  :: forall shape dtype device
   . Float -- ^ maximum value
  -> Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
clampMax max input = unsafePerformIO $ cast2 ATen.clamp_max_ts input max

-- | clampMin
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
-- TODO: can we use D.Scalar for the minimum value?
-- >>> dtype &&& shape $ clampMin 0 (ones :: CPUTensor 'D.Float '[3,2])
-- (Float,[3,2])
clampMin
  :: forall shape dtype device
   . Float -- ^ minimum value
  -> Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
clampMin min input = unsafePerformIO $ cast2 ATen.clamp_min_ts input min

-- | cudnnIsAcceptable
-- TODO: calling this probably makes only sense when the device is CUDA
cudnnIsAcceptable
  :: forall shape dtype device
   . Tensor device dtype shape -- ^ input
  -> Bool -- ^ output
cudnnIsAcceptable input =
  unsafePerformIO $ cast1 ATen.cudnn_is_acceptable_t input

-- constant_pad_nd :: Tensor device dtype shape -> [Int] -> Float -> Tensor device dtype shape
-- constant_pad_nd _input _pad _value = unsafePerformIO $ (cast3 ATen.constant_pad_nd_tls) _input _pad _value

constantPadNd1d
  :: forall (pad :: (Nat, Nat)) n dtype device
   . (All KnownNat '[Fst pad, Snd pad, n])
  => Float
  -> Tensor device dtype '[n]
  -> Tensor device dtype '[n + Fst pad + Snd pad]
constantPadNd1d value input = unsafePerformIO $ cast3
  ATen.constant_pad_nd_tls
  input
  ([natValI @(Fst pad), natValI @(Snd pad)] :: [Int])
  value

-- convolution :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> [Int] -> [Int] -> [Int] -> Bool -> [Int] -> Int -> Tensor device dtype shape
-- convolution _input _weight _bias _stride _padding _dilation _transposed _output_padding _groups = unsafePerformIO $ (cast9 ATen.convolution_tttlllbll) _input _weight _bias _stride _padding _dilation _transposed _output_padding _groups

type ConvSideCheck h k d (p :: Nat) o =
  (
  -- kernel and step size must be > 0
    k >= 1, d >= 1
  -- kernel size can't be greater than actual input size
  , ((h + (2 * p)) + 1) >= k
  -- output size must be greater than 0
  , o >= 1
  -- output forumlation:
  , o ~ ((Div ((h + (2 * p)) - k) d) + 1)
  )

-- | ConvOutputSize
-- TODO: this doesn't seem to be used, remove? use it above in ConvSideCheck?
-- >>> :kind! ConvOutputSize 1 0 1 4
-- ConvOutputSize 1 0 1 4 :: Nat
-- = 4
type family ConvOutputSize (stride :: Nat) (padding :: Nat) (kernel_size :: Nat)  (input_size :: Nat) :: Nat where
    ConvOutputSize s p k i = (Div (i + 2 * p - k) s) + 1

-- | conv1d
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
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
conv1d weight bias input = unsafePerformIO $ cast7 ATen.conv1d_tttllll
                                                   input
                                                   weight
                                                   bias
                                                   (natValI @stride :: Int)
                                                   (natValI @padding :: Int)
                                                   (1 :: Int)
                                                   (1 :: Int)

-- | conv2d
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
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
   . ( All KnownNat '[ Fst stride, Snd stride
                     , Fst padding, Snd padding
                     , inputChannelSize, outputChannelSize
                     , kernelSize0, kernelSize1
                     , inputSize0, inputSize1
                     , batchSize
                     , outputSize0, outputSize1
                     ]
     , ConvSideCheck inputSize0 kernelSize0 (Fst stride) (Fst padding) outputSize0
     , ConvSideCheck inputSize1 kernelSize1 (Snd stride) (Snd padding) outputSize1
     )
  => Tensor device dtype '[outputChannelSize, inputChannelSize, kernelSize0, kernelSize1] -- ^ weight
  -> Tensor device dtype '[outputChannelSize] -- ^ bias
  -> Tensor device dtype '[batchSize, inputChannelSize, inputSize0, inputSize1] -- ^ input
  -> Tensor device dtype '[batchSize, outputChannelSize, outputSize0, outputSize1] -- ^ output
conv2d weight bias input = unsafePerformIO $ cast7
  ATen.conv2d_tttllll
  input
  weight
  bias
  ([natValI @(Fst stride), natValI @(Snd stride)] :: [Int])
  ([natValI @(Fst padding), natValI @(Snd padding)] :: [Int])
  ([1, 1] :: [Int])
  (1 :: Int)

-- | conv3d
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
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
conv3d weight bias input = unsafePerformIO $ cast7
  ATen.conv3d_tttllll
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
  unsafePerformIO $ cast4 ATen.conv_tbc_tttl input weight bias (natValI @padding)

-- conv_transpose1d :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Int -> Int -> Int -> Int -> Int -> Tensor device dtype shape
-- conv_transpose1d _input _weight _bias _stride _padding _output_padding _groups _dilation = unsafePerformIO $ (cast8 ATen.conv_transpose1d_tttlllll) _input _weight _bias _stride _padding _output_padding _groups _dilation

-- | cosh
-- >>> dtype &&& shape $ cosh (ones :: CPUTensor 'D.Float '[3,2])
-- (Float,[3,2])
cosh
  :: forall shape dtype device
   . (StandardFloatingPointDTypeValidation device dtype)
  => Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
cosh input = unsafePerformIO $ cast1 ATen.cosh_t input

-- cosine_embedding_loss :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Double -> Int -> Tensor device dtype shape
-- cosine_embedding_loss _input1 _input2 _target _margin _reduction = unsafePerformIO $ (cast5 ATen.cosine_embedding_loss_tttdl) _input1 _input2 _target _margin _reduction

-- cudnn_affine_grid_generator :: Tensor device dtype shape -> Int -> Int -> Int -> Int -> Tensor device dtype shape
-- cudnn_affine_grid_generator _theta _N _C _H _W = unsafePerformIO $ (cast5 ATen.cudnn_affine_grid_generator_tllll) _theta _N _C _H _W

-- cudnn_batch_norm :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Bool -> Double -> Double -> (Tensor device dtype shape,Tensor device dtype shape,Tensor device dtype shape)
-- cudnn_batch_norm _input _weight _bias _running_mean _running_var _training _exponential_average_factor _epsilon = unsafePerformIO $ (cast8 ATen.cudnn_batch_norm_tttttbdd) _input _weight _bias _running_mean _running_var _training _exponential_average_factor _epsilon

-- cudnn_convolution :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> [Int] -> [Int] -> [Int] -> Int -> Bool -> Bool -> Tensor device dtype shape
-- cudnn_convolution _input _weight _bias _padding _stride _dilation _groups _benchmark _deterministic = unsafePerformIO $ (cast9 ATen.cudnn_convolution_tttllllbb) _input _weight _bias _padding _stride _dilation _groups _benchmark _deterministic

-- cudnn_convolution_transpose :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> [Int] -> [Int] -> [Int] -> [Int] -> Int -> Bool -> Bool -> Tensor device dtype shape
-- cudnn_convolution_transpose _input _weight _bias _padding _output_padding _stride _dilation _groups _benchmark _deterministic = unsafePerformIO $ (cast10 ATen.cudnn_convolution_transpose_tttlllllbb) _input _weight _bias _padding _output_padding _stride _dilation _groups _benchmark _deterministic

-- cudnn_grid_sampler :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape
-- cudnn_grid_sampler _input _grid = unsafePerformIO $ (cast2 ATen.cudnn_grid_sampler_tt) _input _grid

-- | Det
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
-- >>> dtype &&& shape $ det (ones :: CPUTensor 'D.Float '[2,2])
-- (Float,[])
-- >>> dtype &&& shape $ det (ones :: CPUTensor 'D.Float '[3,2,2])
-- (Float,[3])
det
  :: forall shape dtype device
   . Tensor device dtype shape -- ^ input
  -> Tensor device dtype (Det shape) -- ^ output
det input = unsafePerformIO $ cast1 ATen.det_t input

-- diag_embed :: Tensor device dtype shape -> Int -> Int -> Int -> Tensor device dtype shape
-- diag_embed _input _offset _dim1 _dim2 = unsafePerformIO $ (cast4 ATen.diag_embed_tlll) _input _offset _dim1 _dim2

-- diagflat :: Tensor device dtype shape -> Int -> Tensor device dtype shape
-- diagflat _input _offset = unsafePerformIO $ (cast2 ATen.diagflat_tl) _input _offset

-- diagonal :: Tensor device dtype shape -> Int -> Int -> Int -> Tensor device dtype shape
-- diagonal _input _offset _dim1 _dim2 = unsafePerformIO $ (cast4 ATen.diagonal_tlll) _input _offset _dim1 _dim2

-- dot :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape
-- dot _input _tensor = unsafePerformIO $ (cast2 ATen.dot_tt) _input _tensor

-- einsum :: String -> [Tensor device dtype shape] -> Tensor device dtype shape
-- einsum _equation _tensors = unsafePerformIO $ (cast2 ATen.einsum_sl) _equation _tensors

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
  unsafePerformIO $ cast5 ATen.embedding_ttlbb weights indices paddingIdx scaleGradByFreq sparse
 where paddingIdx :: Int
       paddingIdx = case maybeNatVal @paddingIdx of
                      Just idx -> fromIntegral idx
                      Nothing  -> -1

-- embedding_bag :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Bool -> Int -> Bool -> Tensor device dtype shape -> (Tensor device dtype shape,Tensor device dtype shape,Tensor device dtype shape,Tensor device dtype shape)
-- embedding_bag _weight _indices _offsets _scale_grad_by_freq _mode _sparse _per_sample_weights = unsafePerformIO $ (cast7 ATen.embedding_bag_tttblbt) _weight _indices _offsets _scale_grad_by_freq _mode _sparse _per_sample_weights

-- | emptyLike
-- TODO: this seems quite unsafe, the values of this tensor will be random
-- >>> t <- emptyLike (ones :: CPUTensor 'D.Float '[3,4,5])
-- >>> dtype &&& shape $ t
-- (Float,[3,4,5])
emptyLike
  :: forall shape dtype device
   . Tensor device dtype shape -- ^ input
  -> IO (Tensor device dtype shape) -- ^ output
emptyLike input = cast1 ATen.empty_like_t input

-- | erfc
-- >>> dtype &&& shape $ erfc (ones :: CPUTensor 'D.Float '[3,2])
-- (Float,[3,2])
erfc
  :: forall shape dtype device
   . (StandardFloatingPointDTypeValidation device dtype)
  => Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
erfc input = unsafePerformIO $ cast1 ATen.erfc_t input

-- | expm1
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
-- >>> dtype &&& shape $ expm1 (ones :: CPUTensor 'D.Float '[3,2])
-- (Float,[3,2])
expm1
  :: forall shape dtype device
   . (StandardFloatingPointDTypeValidation device dtype)
  => Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
expm1 input = unsafePerformIO $ cast1 ATen.expm1_t input

-- | expand
-- TODO: figure out what the boolean value does
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
expand someBool input = unsafePerformIO $ cast3 ATen.tensor_expand_lb input (shapeVal @shape') someBool

-- flatten :: Tensor device dtype shape -> Int -> Int -> Tensor device dtype shape
-- flatten _input _start_dim _end_dim = unsafePerformIO $ (cast3 ATen.flatten_tll) _input _start_dim _end_dim

-- | flattenAll
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
  unsafePerformIO $ cast3 ATen.flatten_tll input (0 :: Int) (-1 :: Int)

-- | frac
-- >>> dtype &&& shape $ frac (ones :: CPUTensor 'D.Float '[3,2])
-- (Float,[3,2])
frac
  :: forall shape dtype device
   . (StandardFloatingPointDTypeValidation device dtype)
  => Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
frac input = unsafePerformIO $ cast1 ATen.frac_t input

-- | full like
-- >>> dtype &&& shape $ fullLike 3.0 (ones :: CPUTensor 'D.Float '[3,2])
-- (Float,[3,2])
fullLike
  :: forall shape dtype device
   . Float -- ^ fill value
  -> Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
fullLike fillValue input =
  unsafePerformIO $ cast2 ATen.full_like_ts input fillValue

-- grid_sampler :: Tensor device dtype shape -> Tensor device dtype shape -> Int -> Int -> Tensor device dtype shape
-- grid_sampler _input _grid _interpolation_mode _padding_mode = unsafePerformIO $ (cast4 ATen.grid_sampler_ttll) _input _grid _interpolation_mode _padding_mode

-- grid_sampler_2d :: Tensor device dtype shape -> Tensor device dtype shape -> Int -> Int -> Tensor device dtype shape
-- grid_sampler_2d _input _grid _interpolation_mode _padding_mode = unsafePerformIO $ (cast4 ATen.grid_sampler_2d_ttll) _input _grid _interpolation_mode _padding_mode

-- grid_sampler_3d :: Tensor device dtype shape -> Tensor device dtype shape -> Int -> Int -> Tensor device dtype shape
-- grid_sampler_3d _input _grid _interpolation_mode _padding_mode = unsafePerformIO $ (cast4 ATen.grid_sampler_3d_ttll) _input _grid _interpolation_mode _padding_mode

-- hinge_embedding_loss :: Tensor device dtype shape -> Tensor device dtype shape -> Double -> Int -> Tensor device dtype shape
-- hinge_embedding_loss _input _target _margin _reduction = unsafePerformIO $ (cast4 ATen.hinge_embedding_loss_ttdl) _input _target _margin _reduction

-- ger :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape
-- ger _input _vec2 = unsafePerformIO $ (cast2 ATen.ger_tt) _input _vec2

-- group_norm :: Tensor device dtype shape -> Int -> Tensor device dtype shape -> Tensor device dtype shape -> Double -> Bool -> Tensor device dtype shape
-- group_norm _input _num_groups _weight _bias _eps _cudnn_enabled = unsafePerformIO $ (cast6 ATen.group_norm_tlttdb) _input _num_groups _weight _bias _eps _cudnn_enabled

-- fft :: Tensor device dtype shape -> Int -> Bool -> Tensor device dtype shape
-- fft _input _signal_ndim _normalized = unsafePerformIO $ (cast3 ATen.fft_tlb) _input _signal_ndim _normalized

-- ifft :: Tensor device dtype shape -> Int -> Bool -> Tensor device dtype shape
-- ifft _input _signal_ndim _normalized = unsafePerformIO $ (cast3 ATen.ifft_tlb) _input _signal_ndim _normalized

-- rfft :: Tensor device dtype shape -> Int -> Bool -> Bool -> Tensor device dtype shape
-- rfft _input _signal_ndim _normalized _onesided = unsafePerformIO $ (cast4 ATen.rfft_tlbb) _input _signal_ndim _normalized _onesided

-- irfft :: Tensor device dtype shape -> Int -> Bool -> Bool -> [Int] -> Tensor device dtype shape
-- irfft _input _signal_ndim _normalized _onesided _signal_sizes = unsafePerformIO $ (cast5 ATen.irfft_tlbbl) _input _signal_ndim _normalized _onesided _signal_sizes

-- index :: Tensor device dtype shape -> [Tensor device dtype shape] -> Tensor device dtype shape
-- index _input _indices = unsafePerformIO $ (cast2 ATen.index_tl) _input _indices

-- index_copy :: Tensor device dtype shape -> Int -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape
-- index_copy _input _dim _index _source = unsafePerformIO $ (cast4 ATen.index_copy_tltt) _input _dim _index _source

-- index_put :: Tensor device dtype shape -> [Tensor device dtype shape] -> Tensor device dtype shape -> Bool -> Tensor device dtype shape
-- index_put _input _indices _values _accumulate = unsafePerformIO $ (cast4 ATen.index_put_tltb) _input _indices _values _accumulate

-- instance_norm :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Bool -> Double -> Double -> Bool -> Tensor device dtype shape
-- instance_norm _input _weight _bias _running_mean _running_var _use_input_stats _momentum _eps _cudnn_enabled = unsafePerformIO $ (cast9 ATen.instance_norm_tttttbddb) _input _weight _bias _running_mean _running_var _use_input_stats _momentum _eps _cudnn_enabled

-- | isclose
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
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
  unsafePerformIO $ cast5 ATen.isclose_ttddb input other rtol atol equalNaN

-- | is NaN
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
-- >>> dtype &&& shape $ isNaN (ones :: CPUTensor 'D.Float '[3,2])
-- (Bool,[3,2])
isNaN
  :: forall shape dtype device
   . Tensor device dtype   shape -- ^ input
  -> Tensor device 'D.Bool shape -- ^ output
isNaN input = unsafePerformIO $ cast1 ATen.isnan_t input

-- | is distributed
isDistributed
  :: forall shape dtype device
   . Tensor device dtype shape  -- ^ input
  -> Bool -- ^ output
isDistributed input = unsafePerformIO $ cast1 ATen.is_distributed_t input

-- | is floating point
-- TODO: this can be decided statically
isFloatingPoint
  :: forall shape dtype device
   . Tensor device dtype shape  -- ^ input
  -> Bool -- ^ output
isFloatingPoint input = unsafePerformIO $ cast1 ATen.is_floating_point_t input

-- | is complex
isComplex
  :: forall shape dtype device
   . Tensor device dtype shape  -- ^ input
  -> Bool -- ^ output
isComplex input = unsafePerformIO $ cast1 ATen.is_complex_t input

-- | is non-zero
isNonZero
  :: forall shape dtype device
   . Tensor device dtype shape  -- ^ input
  -> Bool -- ^ output
isNonZero input = unsafePerformIO $ cast1 ATen.is_nonzero_t input

-- | is same size
-- TODO: this can be decided statically
isSameSize
  :: forall shape shape' dtype device
   . Tensor device dtype shape -- ^ input tensor
  -> Tensor device dtype shape' -- ^ other input tensor
  -> Bool -- ^ output
isSameSize input other =
  unsafePerformIO $ cast2 ATen.is_same_size_tt input other

isSigned
  :: forall shape dtype device
   . Tensor device dtype shape -- ^ input
  -> Bool -- ^ output
isSigned input = unsafePerformIO $ cast1 ATen.is_signed_t input

-- kl_div :: Tensor device dtype shape -> Tensor device dtype shape -> Int -> Tensor device dtype shape
-- kl_div _input _target _reduction = unsafePerformIO $ (cast3 ATen.kl_div_ttl) _input _target _reduction

-- kthvalue :: Tensor device dtype shape -> Int -> Int -> Bool -> (Tensor device dtype shape,Tensor device dtype shape)
-- kthvalue _input _k _dim _keepdim = unsafePerformIO $ (cast4 ATen.kthvalue_tllb) _input _k _dim _keepdim

-- | EndsWith
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
layerNorm weight bias eps input = unsafePerformIO $ cast6
  ATen.layer_norm_tlttdb
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
-- native_layer_norm _input _weight _bias _M _N _eps = unsafePerformIO $ (cast6 ATen.native_layer_norm_tttlld) _input _weight _bias _M _N _eps

-- | linear
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
-- https://pytorch.org/docs/stable/_modules/torch/nn/functional.html#linear
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
linear weight bias input = unsafePerformIO $ cast3 ATen.linear_ttt input weight bias

-- | linear'
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
-- TODO: can we use the ATen linear function or not here?
-- https://pytorch.org/docs/stable/_modules/torch/nn/functional.html#linear
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
  :: forall (inputFeatures :: Nat) (outputFeatures :: Nat) (shape :: [Nat]) (shape' :: [Nat]) dtype device
   . (shape' ~ CheckBroadcast
                 (CheckMatMul
                     shape
                     '[inputFeatures, outputFeatures]
                     (ComputeMatMul
                       (ReverseImpl shape '[]) '[outputFeatures, inputFeatures]))
                 '[outputFeatures]
                 (ComputeBroadcast
                     (ReverseImpl
                       (CheckMatMul
                           shape
                           '[inputFeatures, outputFeatures]
                           (ComputeMatMul
                             (ReverseImpl shape '[]) '[outputFeatures, inputFeatures]))
                       '[])
                     '[outputFeatures])
     )
  => Tensor device dtype '[outputFeatures, inputFeatures] -- ^ weight
  -> Tensor device dtype '[outputFeatures] -- ^ bias
  -> Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape' -- ^ output
-- linear' weight bias input = Torch.Static.add (matmul input $ transpose @0 @1 weight) bias
linear' weight bias input = unsafePerformIO $ cast3 ATen.linear_ttt input weight bias

-- | mkldnnLinear
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
-- TODO: mkldnnLinear does not return a usuable tensor value and is hence broken
-- TODO: figure out `device` for this
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
mkldnnLinear weight bias input = unsafePerformIO $ cast3 ATen.mkldnn_linear_ttt input weight bias

-- fbgemm_linear_int8_weight :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Float -> Float -> Tensor device dtype shape -> Tensor device dtype shape
-- fbgemm_linear_int8_weight _input _weight _packed _col_offsets _weight_scale _weight_zero_point _bias = unsafePerformIO $ (cast7 ATen.fbgemm_linear_int8_weight_ttttsst) _input _weight _packed _col_offsets _weight_scale _weight_zero_point _bias

-- fbgemm_linear_quantize_weight :: Tensor device dtype shape -> (Tensor device dtype shape,Tensor device dtype shape,Double,Int)
-- fbgemm_linear_quantize_weight _input = unsafePerformIO $ (cast1 ATen.fbgemm_linear_quantize_weight_t) _input

-- fbgemm_pack_gemm_matrix_fp16 :: Tensor device dtype shape -> Tensor device dtype shape
-- fbgemm_pack_gemm_matrix_fp16 _input = unsafePerformIO $ (cast1 ATen.fbgemm_pack_gemm_matrix_fp16_t) _input

-- fbgemm_linear_fp16_weight :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape
-- fbgemm_linear_fp16_weight _input _packed_weight _bias = unsafePerformIO $ (cast3 ATen.fbgemm_linear_fp16_weight_ttt) _input _packed_weight _bias

-- fbgemm_pack_quantized_matrix :: Tensor device dtype shape -> Int -> Int -> Tensor device dtype shape
-- fbgemm_pack_quantized_matrix _input _K _N = unsafePerformIO $ (cast3 ATen.fbgemm_pack_quantized_matrix_tll) _input _K _N

-- fbgemm_is_cpu_supported :: Bool
-- fbgemm_is_cpu_supported  = unsafePerformIO $ (cast0 ATen.fbgemm_is_cpu_supported) 

-- | log
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
-- TODO: will log throw for negative numbers or just generate NaNs? should we return a Maybe?
-- >>> dtype &&& shape $ log (ones :: CPUTensor 'D.Float '[3,2])
-- (Float,[3,2])
log
  :: forall shape dtype device
   . Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
log input = unsafePerformIO $ cast1 ATen.log_t input

-- | logDet
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
-- TODO: will logDet throw? and if so, should we return a Maybe?
-- >>> dtype &&& shape $ logDet (ones :: CPUTensor 'D.Float '[2,2])
-- (Float,[])
-- >>> dtype &&& shape $ logDet (ones :: CPUTensor 'D.Float '[3,2,2])
-- (Float,[3])
logDet
  :: forall shape' shape dtype device
   . (shape' ~ Det shape)
  => Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape' -- ^ output
logDet input = unsafePerformIO $ cast1 ATen.logdet_t input                       

-- | logarithm of the sum of the exponentials
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
-- See https://pytorch.org/docs/stable/torch.html#torch.logsumexp.
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
logSumExp input = unsafePerformIO $ cast3 ATen.logsumexp_tlb
                                          input
                                          (natValI @dim)
                                          (keepOrDropDimVal @keepOrDropDim)

-- margin_ranking_loss :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Double -> Int -> Tensor device dtype shape
-- margin_ranking_loss _input1 _input2 _target _margin _reduction = unsafePerformIO $ (cast5 ATen.margin_ranking_loss_tttdl) _input1 _input2 _target _margin _reduction

-- | matrixPower
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
-- TODO: figure out input shape restrictions, should be matrix or a batched matrix
-- TODO: figure out restrictions on the power, can it be zero or negative?
-- >>> dtype &&& shape $ matrixPower 2 (ones :: CPUTensor 'D.Float '[3,4,4])
-- (Float,[3,4,4])
matrixPower
  :: forall shape' shape dtype device
   . (shape' ~ Square shape)
  => Int -- ^ power
  -> Tensor device dtype shape -- ^ input matrix
  -> Tensor device dtype shape' -- ^ output
matrixPower n input = unsafePerformIO $ cast2 ATen.matrix_power_tl input n

-- | maxValues
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
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
maxValues input = unsafePerformIO $ cast3 ATen.max_values_tlb
                                          input
                                          (natValI @dim)
                                          (keepOrDropDimVal @keepOrDropDim)

-- | minValues
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
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
minValues input = unsafePerformIO $ cast3 ATen.min_values_tlb
                                          input
                                          (natValI @dim)
                                          (keepOrDropDimVal @keepOrDropDim)

-- | maxPool1d
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
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
maxPool1d input = unsafePerformIO $ cast6 ATen.max_pool1d_tllllb
                                          input
                                          (natValI @kernelSize)
                                          (natValI @stride)
                                          (natValI @padding)
                                          (1 :: Int)
                                          False

-- max_pool1d_with_indices :: Tensor device dtype shape -> Int -> Int -> Int -> Int -> Bool -> (Tensor device dtype shape,Tensor device dtype shape)
-- max_pool1d_with_indices _input _kernel_size _stride _padding _dilation _ceil_mode = unsafePerformIO $ (cast6 ATen.max_pool1d_with_indices_tllllb) _input _kernel_size _stride _padding _dilation _ceil_mode

-- | maxPool2d
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
-- >>> t = maxPool2d @'(1,1) @'(1,1) @'(0,0) (ones :: CPUTensor 'D.Float '[1,3,4,5])
-- >>> shape t
-- [1,3,4,5]
-- >>> :t t
-- t :: Tensor '( 'D.CPU, 0) 'D.Float '[1, 3, 4, 5]
maxPool2d
  :: forall kernelSize stride padding channelSize inputSize0 inputSize1 batchSize outputSize0 outputSize1 dtype device
   . ( All KnownNat '[ Fst kernelSize, Snd kernelSize
                     , Fst stride, Snd stride
                     , Fst padding, Snd padding
                     , channelSize
                     , inputSize0, inputSize1
                     , batchSize
                     ]
     , ConvSideCheck inputSize0 (Fst kernelSize) (Fst stride) (Fst padding) outputSize0
     , ConvSideCheck inputSize1 (Snd kernelSize) (Snd stride) (Snd padding) outputSize1
     )
  => Tensor device dtype '[batchSize, channelSize, inputSize0, inputSize1] -- ^ input
  -> Tensor device dtype '[batchSize, channelSize, outputSize0, outputSize1] -- ^ output
maxPool2d input = unsafePerformIO $ cast6
  ATen.max_pool2d_tllllb
  input
  ([natValI @(Fst kernelSize), natValI @(Snd kernelSize)] :: [Int])
  ([natValI @(Fst stride), natValI @(Snd stride)] :: [Int])
  ([natValI @(Fst padding), natValI @(Snd padding)] :: [Int])
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
   . ( All KnownNat '[ Fst kernelSize, Snd kernelSize
                     , Fst stride, Snd stride
                     , Fst padding, Snd padding
                     , channelSize
                     , inputSize0, inputSize1
                     , batchSize
                     ]
     , ConvSideCheck inputSize0 (Fst kernelSize) (Fst stride) (Fst padding) outputSize0
     , ConvSideCheck inputSize1 (Snd kernelSize) (Snd stride) (Snd padding) outputSize1
     )
  => Tensor device dtype '[batchSize, channelSize, inputSize0, inputSize1] -- ^ input
  -> Tensor device dtype '[batchSize, channelSize, outputSize0, outputSize1] -- ^ output
mkldnnMaxPool2d input = unsafePerformIO $ cast6
  ATen.mkldnn_max_pool2d_tllllb
  input
  ([natValI @(Fst kernelSize), natValI @(Snd kernelSize)] :: [Int])
  ([natValI @(Fst stride), natValI @(Snd stride)] :: [Int])
  ([natValI @(Fst padding), natValI @(Snd padding)] :: [Int])
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
   . ( All KnownNat '[ Fst kernelSize, Snd kernelSize
                     , Fst stride, Snd stride
                     , Fst padding, Snd padding
                     , channelSize
                     , inputSize0, inputSize1
                     , batchSize
                     ]
     , ConvSideCheck inputSize0 (Fst kernelSize) (Fst stride) (Fst padding) outputSize0
     , ConvSideCheck inputSize1 (Snd kernelSize) (Snd stride) (Snd padding) outputSize1
     )
  => Tensor device dtype '[batchSize, channelSize, inputSize0, inputSize1] -- ^ input
  -> Tensor device dtype '[batchSize, channelSize, outputSize0, outputSize1] -- ^ output
quantizedMaxPool2d input = unsafePerformIO $ cast5
  ATen.quantized_max_pool2d_tllll
  input
  ([natValI @(Fst kernelSize), natValI @(Snd kernelSize)] :: [Int])
  ([natValI @(Fst stride), natValI @(Snd stride)] :: [Int])
  ([natValI @(Fst padding), natValI @(Snd padding)] :: [Int])
  ([1, 1] :: [Int])

-- | maxPool3d
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
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
maxPool3d input = unsafePerformIO $ cast6
  ATen.max_pool3d_tllllb
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
  unsafePerformIO $ cast3 ATen.masked_fill_tts input mask value

-- mkldnn_convolution :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> [Int] -> [Int] -> [Int] -> Int -> Tensor device dtype shape
-- mkldnn_convolution _input _weight _bias _padding _stride _dilation _groups = unsafePerformIO $ (cast7 ATen.mkldnn_convolution_tttllll) _input _weight _bias _padding _stride _dilation _groups

-- miopen_batch_norm :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Bool -> Double -> Double -> (Tensor device dtype shape,Tensor device dtype shape,Tensor device dtype shape)
-- miopen_batch_norm _input _weight _bias _running_mean _running_var _training _exponential_average_factor _epsilon = unsafePerformIO $ (cast8 ATen.miopen_batch_norm_tttttbdd) _input _weight _bias _running_mean _running_var _training _exponential_average_factor _epsilon

-- miopen_convolution :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> [Int] -> [Int] -> [Int] -> Int -> Bool -> Bool -> Tensor device dtype shape
-- miopen_convolution _input _weight _bias _padding _stride _dilation _groups _benchmark _deterministic = unsafePerformIO $ (cast9 ATen.miopen_convolution_tttllllbb) _input _weight _bias _padding _stride _dilation _groups _benchmark _deterministic

-- miopen_convolution_transpose :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> [Int] -> [Int] -> [Int] -> [Int] -> Int -> Bool -> Bool -> Tensor device dtype shape
-- miopen_convolution_transpose _input _weight _bias _padding _output_padding _stride _dilation _groups _benchmark _deterministic = unsafePerformIO $ (cast10 ATen.miopen_convolution_transpose_tttlllllbb) _input _weight _bias _padding _output_padding _stride _dilation _groups _benchmark _deterministic

-- miopen_depthwise_convolution :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> [Int] -> [Int] -> [Int] -> Int -> Bool -> Bool -> Tensor device dtype shape
-- miopen_depthwise_convolution _input _weight _bias _padding _stride _dilation _groups _benchmark _deterministic = unsafePerformIO $ (cast9 ATen.miopen_depthwise_convolution_tttllllbb) _input _weight _bias _padding _stride _dilation _groups _benchmark _deterministic

-- miopen_rnn :: Tensor device dtype shape -> [Tensor device dtype shape] -> Int -> Tensor device dtype shape -> Tensor device dtype shape -> Int -> Int -> Int -> Bool -> Double -> Bool -> Bool -> [Int] -> Tensor device dtype shape -> (Tensor device dtype shape,Tensor device dtype shape,Tensor device dtype shape,Tensor device dtype shape,Tensor device dtype shape)
-- miopen_rnn _input _weight _weight_stride0 _hx _cx _mode _hidden_size _num_layers _batch_first _dropout _train _bidirectional _batch_sizes _dropout_state = unsafePerformIO $ (cast14 ATen.miopen_rnn_tllttlllbdbblt) _input _weight _weight_stride0 _hx _cx _mode _hidden_size _num_layers _batch_first _dropout _train _bidirectional _batch_sizes _dropout_state

-- | matrix-matrix multiplication
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
-- >>> dtype &&& shape $ mm (ones :: CPUTensor 'D.Float '[3,2]) (zeros :: CPUTensor 'D.Float '[2,4])
-- (Float,[3,4])
mm
  :: forall n k m dtype device
   . Tensor device dtype '[n, k] -- ^ first input matrix
  -> Tensor device dtype '[k, m] -- ^ second input matrix
  -> Tensor device dtype '[n, m] -- ^ output matrix
mm a b = unsafePerformIO $ cast2 ATen.mm_tt a b

-- mode :: Tensor device dtype shape -> Int -> Bool -> (Tensor device dtype shape,Tensor device dtype shape)
-- mode _input _dim _keepdim = unsafePerformIO $ (cast3 ATen.mode_tlb) _input _dim _keepdim

-- | matrix-vector multiplication
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
-- >>> dtype &&& shape $ mv (ones :: CPUTensor 'D.Float '[3,2]) (zeros :: CPUTensor 'D.Float '[2])
-- (Float,[3])
mv
  :: forall n m dtype device
   . Tensor device dtype '[n, m] -- ^ input matrix
  -> Tensor device dtype '[m] -- ^ input vector
  -> Tensor device dtype '[n] -- ^ output vector
mv input vec = unsafePerformIO $ cast2 ATen.mv_tt input vec

-- mvlgamma :: Tensor device dtype shape -> Int -> Tensor device dtype shape
-- mvlgamma _input _p = unsafePerformIO $ (cast2 ATen.mvlgamma_tl) _input _p

-- narrow :: Tensor device dtype shape -> Int -> Int -> Int -> Tensor device dtype shape
-- narrow _input _dim _start _length = unsafePerformIO $ (cast4 ATen.narrow_tlll) _input _dim _start _length

-- native_batch_norm :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Bool -> Double -> Double -> (Tensor device dtype shape,Tensor device dtype shape,Tensor device dtype shape)
-- native_batch_norm _input _weight _bias _running_mean _running_var _training _momentum _eps = unsafePerformIO $ (cast8 ATen.native_batch_norm_tttttbdd) _input _weight _bias _running_mean _running_var _training _momentum _eps

-- batch_norm_stats :: Tensor device dtype shape -> Double -> (Tensor device dtype shape,Tensor device dtype shape)
-- batch_norm_stats _input _eps = unsafePerformIO $ (cast2 ATen.batch_norm_stats_td) _input _eps

-- batch_norm_elemt :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Double -> Tensor device dtype shape
-- batch_norm_elemt _input _weight _bias _mean _invstd _eps = unsafePerformIO $ (cast6 ATen.batch_norm_elemt_tttttd) _input _weight _bias _mean _invstd _eps

-- batch_norm_gather_stats :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Double -> Double -> Int -> (Tensor device dtype shape,Tensor device dtype shape)
-- batch_norm_gather_stats _input _mean _invstd _running_mean _running_var _momentum _eps _count = unsafePerformIO $ (cast8 ATen.batch_norm_gather_stats_tttttddl) _input _mean _invstd _running_mean _running_var _momentum _eps _count

-- batch_norm_gather_stats_with_counts :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Double -> Double -> [Int] -> (Tensor device dtype shape,Tensor device dtype shape)
-- batch_norm_gather_stats_with_counts _input _mean _invstd _running_mean _running_var _momentum _eps _counts = unsafePerformIO $ (cast8 ATen.batch_norm_gather_stats_with_counts_tttttddl) _input _mean _invstd _running_mean _running_var _momentum _eps _counts

-- batch_norm_update_stats :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Double -> (Tensor device dtype shape,Tensor device dtype shape)
-- batch_norm_update_stats _input _running_mean _running_var _momentum = unsafePerformIO $ (cast4 ATen.batch_norm_update_stats_tttd) _input _running_mean _running_var _momentum

-- | onesLike
-- >>> dtype &&& shape $ onesLike (ones :: CPUTensor 'D.Float '[3,4,5])
-- (Float,[3,4,5])
onesLike
  :: forall shape dtype device
   . Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
onesLike input = unsafePerformIO $ cast1 ATen.ones_like_t input

-- pairwise_distance :: Tensor device dtype shape -> Tensor device dtype shape -> Double -> Double -> Bool -> Tensor device dtype shape
-- pairwise_distance _x1 _x2 _p _eps _keepdim = unsafePerformIO $ (cast5 ATen.pairwise_distance_ttddb) _x1 _x2 _p _eps _keepdim

-- cdist :: Tensor device dtype shape -> Tensor device dtype shape -> Double -> Tensor device dtype shape
-- cdist _x1 _x2 _p = unsafePerformIO $ (cast3 ATen.cdist_ttd) _x1 _x2 _p

-- pdist :: Tensor device dtype shape -> Double -> Tensor device dtype shape
-- pdist _input _p = unsafePerformIO $ (cast2 ATen.pdist_td) _input _p

-- cosine_similarity :: Tensor device dtype shape -> Tensor device dtype shape -> Int -> Double -> Tensor device dtype shape
-- cosine_similarity _x1 _x2 _dim _eps = unsafePerformIO $ (cast4 ATen.cosine_similarity_ttld) _x1 _x2 _dim _eps

-- pixel_shuffle :: Tensor device dtype shape -> Int -> Tensor device dtype shape
-- pixel_shuffle _input _upscale_factor = unsafePerformIO $ (cast2 ATen.pixel_shuffle_tl) _input _upscale_factor

-- pin_memory :: Tensor device dtype shape -> Tensor device dtype shape
-- pin_memory _input = unsafePerformIO $ (cast1 ATen.pin_memory_t) _input

-- pinverse :: Tensor device dtype shape -> Double -> Tensor device dtype shape
-- pinverse _input _rcond = unsafePerformIO $ (cast2 ATen.pinverse_td) _input _rcond

-- poisson_nll_loss :: Tensor device dtype shape -> Tensor device dtype shape -> Bool -> Bool -> Double -> Int -> Tensor device dtype shape
-- poisson_nll_loss _input _target _log_input _full _eps _reduction = unsafePerformIO $ (cast6 ATen.poisson_nll_loss_ttbbdl) _input _target _log_input _full _eps _reduction

-- | randLike
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
-- >>> t <- randLike (ones :: CPUTensor 'D.Float '[3,4,5])
-- >>> dtype &&& shape $ t
-- (Float,[3,4,5])
randLike
  :: forall shape dtype device
   . Tensor device dtype shape -- ^ input
  -> IO (Tensor device dtype shape) -- ^ output
randLike = cast1 ATen.rand_like_t

-- | randnLike
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
-- >>> t <- randnLike (ones :: CPUTensor 'D.Float '[3,4,5])
-- >>> dtype &&& shape $ t
-- (Float,[3,4,5])
randnLike
  :: forall shape dtype device
   . Tensor device dtype shape -- ^ input
  -> IO (Tensor device dtype shape) -- ^ output
randnLike = cast1 ATen.randn_like_t

-- reciprocal :: Tensor device dtype shape -> Tensor device dtype shape
-- reciprocal _input = unsafePerformIO $ (cast1 ATen.reciprocal_t) _input

-- | negate
-- TODO: probably not defined for `D.Bool` tensors
-- >>> dtype &&& shape $ neg (ones :: CPUTensor 'D.Float '[3,2])
-- (Float,[3,2])
neg
  :: forall shape dtype device
   . Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
neg input = unsafePerformIO $ cast1 ATen.neg_t input

-- | round
-- TODO: probably only defined for floating point tensors
-- >>> dtype &&& shape $ round (ones :: CPUTensor 'D.Float '[3,2])
-- (Float,[3,2])
round
  :: forall shape dtype device
   . Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
round input = unsafePerformIO $ cast1 ATen.round_t input

-- | prelu activation function
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
-- >>> dtype &&& shape $ prelu (ones :: CPUTensor 'D.Float '[]) (ones :: CPUTensor 'D.Float '[3,2])
-- (Float,[3,2])
prelu
  :: forall shape dtype device
   . Tensor device dtype '[] -- ^ weight
  -> Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
prelu weight input = unsafePerformIO $ cast2 ATen.prelu_tt input weight

type family GeluDTypeIsValid (device :: (D.DeviceType, Nat)) (dtype :: D.DType) :: Constraint where
  GeluDTypeIsValid '( 'D.CPU, 0)            dtype = ( DTypeIsFloatingPoint '( 'D.CPU, 0) dtype
                                                    , DTypeIsNotHalf '( 'D.CPU, 0) dtype
                                                    )
  GeluDTypeIsValid '( 'D.CUDA, deviceIndex) dtype = ( DTypeIsFloatingPoint '( 'D.CUDA, deviceIndex) dtype
                                                    , DTypeIsNotHalf '( 'D.CUDA, deviceIndex) dtype
                                                    )
  GeluDTypeIsValid '(deviceType, _)         dtype = UnsupportedDTypeForDevice deviceType dtype

-- | gelu activation function
-- >>> dtype &&& shape $ round (ones :: CPUTensor 'D.Float '[3,2])
-- (Float,[3,2])
gelu
  :: forall shape dtype device
   . (GeluDTypeIsValid device dtype)
  => Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
gelu input = unsafePerformIO $ cast1 ATen.gelu_t input

-- hardshrink :: Tensor device dtype shape -> Float -> Tensor device dtype shape
-- hardshrink _input _lambd = unsafePerformIO $ (cast2 ATen.hardshrink_ts) _input _lambd

-- | rsqrt
-- >>> dtype &&& shape $ rsqrt (ones :: CPUTensor 'D.Float '[3,2])
-- (Float,[3,2])
rsqrt
  :: forall shape dtype device
   . (StandardFloatingPointDTypeValidation device dtype)
  => Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
rsqrt input = unsafePerformIO $ cast1 ATen.rsqrt_t input

-- | celu activation function
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
-- >>> dtype &&& shape $ celu 3.0 (ones :: CPUTensor 'D.Float '[3,2])
-- (Float,[3,2])
celu
  :: forall shape dtype device
   . Float -- ^ alpha
  -> Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
celu alpha input = unsafePerformIO $ cast2 ATen.celu_ts input alpha

-- slice :: Tensor device dtype shape -> Int -> Int -> Int -> Int -> Tensor device dtype shape
-- slice _input _dim _start _end _step = unsafePerformIO $ (cast5 ATen.slice_tllll) _input _dim _start _end _step

-- slogdet :: Tensor device dtype shape -> (Tensor device dtype shape,Tensor device dtype shape)
-- slogdet _input = unsafePerformIO $ (cast1 ATen.slogdet_t) _input

-- smm :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape
-- smm _input _mat2 = unsafePerformIO $ (cast2 ATen.smm_tt) _input _mat2

-- split :: Tensor device dtype shape -> Int -> Int -> [Tensor device dtype shape]
-- split _input _split_size _dim = unsafePerformIO $ (cast3 ATen.split_tll) _input _split_size _dim

-- split_with_sizes :: Tensor device dtype shape -> [Int] -> Int -> [Tensor device dtype shape]
-- split_with_sizes _input _split_sizes _dim = unsafePerformIO $ (cast3 ATen.split_with_sizes_tll) _input _split_sizes _dim

-- sspaddmm :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Float -> Float -> Tensor device dtype shape
-- sspaddmm _input _mat1 _mat2 _beta _alpha = unsafePerformIO $ (cast5 ATen.sspaddmm_tttss) _input _mat1 _mat2 _beta _alpha

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
  MaybePair Nothing   _         _         = Nothing
  MaybePair _         Nothing   _         = Nothing
  MaybePair _         _         Nothing   = Nothing
  MaybePair (Just a') (Just b') (Just c') = Just '(a', b', c')

type family ComputeStackShape (shape :: [Nat]) (dim  :: Nat) (count :: Nat) :: Maybe [Nat] where
  ComputeStackShape _         _   0     = Nothing
  ComputeStackShape xs        0   count = Just (count ': xs)
  ComputeStackShape (x ': xs) dim count = AppendToMaybe x (ComputeStackShape xs (dim - 1) count)
  ComputeStackShape '[]       _   _     = Nothing

type family StackCheck (res :: Maybe ([Nat], D.DType, (D.DeviceType, Nat))) :: ([Nat], D.DType, (D.DeviceType, Nat)) where
  StackCheck 'Nothing                        = TypeError (Text "Stacking impossible.")
  StackCheck ('Just '(shape, dtype, device)) = '(shape, dtype, device)

-- | Stack
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
     , HFoldrM IO TensorListFold [D.ATenTensor] tensors
     , Apply TensorListUnfold [D.ATenTensor] (HUnfoldMRes IO [D.ATenTensor] tensors)
     , HUnfoldM IO TensorListUnfold (HUnfoldMRes IO [D.ATenTensor] tensors) tensors
     )
  => HList tensors -- ^ input list of tensors
  -> Tensor device dtype shape -- ^ output
stack tensors = unsafePerformIO $ cast2 ATen.stack_ll tensors (natValI @dim :: Int)

-- stft :: Tensor device dtype shape -> Int -> Int -> Int -> Tensor device dtype shape -> Bool -> Bool -> Tensor device dtype shape
-- stft _input _n_fft _hop_length _win_length _window _normalized _onesided = unsafePerformIO $ (cast7 ATen.stft_tllltbb) _input _n_fft _hop_length _win_length _window _normalized _onesided

-- stride :: Tensor device dtype shape -> Int -> Int
-- stride _input _dim = unsafePerformIO $ (cast2 ATen.stride_tl) _input _dim

-- t :: Tensor device dtype shape -> Tensor device dtype shape
-- t _input = unsafePerformIO $ (cast1 ATen.t_t) _input

-- | tan
-- >>> dtype &&& shape $ tan (ones :: CPUTensor 'D.Float '[3,2])
-- (Float,[3,2])
tan
  :: forall shape dtype device
   . (StandardFloatingPointDTypeValidation device dtype)
  => Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
tan input = unsafePerformIO $ cast1 ATen.tan_t input

-- tensordot :: Tensor device dtype shape -> Tensor device dtype shape -> [Int] -> [Int] -> Tensor device dtype shape
-- tensordot _input _other _dims_input _dims_other = unsafePerformIO $ (cast4 ATen.tensordot_ttll) _input _other _dims_input _dims_other

-- threshold :: Tensor device dtype shape -> Float -> Float -> Tensor device dtype shape
-- threshold _input _threshold _value = unsafePerformIO $ (cast3 ATen.threshold_tss) _input _threshold _value

-- one_hot :: Tensor device dtype shape -> Int -> Tensor device dtype shape
-- one_hot _input _num_classes = unsafePerformIO $ (cast2 ATen.one_hot_tl) _input _num_classes

-- flip :: Tensor device dtype shape -> [Int] -> Tensor device dtype shape
-- flip _input _dims = unsafePerformIO $ (cast2 ATen.flip_tl) _input _dims

-- roll :: Tensor device dtype shape -> Int -> Int -> Tensor device dtype shape
-- roll _input _shifts _dims = unsafePerformIO $ (cast3 ATen.roll_tll) _input _shifts _dims

-- rot90 :: Tensor device dtype shape -> Int -> [Int] -> Tensor device dtype shape
-- rot90 _input _k _dims = unsafePerformIO $ (cast3 ATen.rot90_tll) _input _k _dims

-- triplet_margin_loss :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Double -> Double -> Double -> Bool -> Int -> Tensor device dtype shape
-- triplet_margin_loss _anchor _positive _negative _margin _p _eps _swap _reduction = unsafePerformIO $ (cast8 ATen.triplet_margin_loss_tttdddbl) _anchor _positive _negative _margin _p _eps _swap _reduction

-- | trunc
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
-- >>> dtype &&& shape $ trunc (ones :: CPUTensor 'D.Float '[3,2])
-- (Float,[3,2])
trunc
  :: forall shape dtype device
   . (StandardFloatingPointDTypeValidation device dtype)
  => Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
trunc input = unsafePerformIO $ cast1 ATen.trunc_t input

-- unique_dim :: Tensor device dtype shape -> Int -> Bool -> Bool -> Bool -> (Tensor device dtype shape,Tensor device dtype shape,Tensor device dtype shape)
-- unique_dim _input _dim _sorted _return_inverse _return_counts = unsafePerformIO $ (cast5 ATen.unique_dim_tlbbb) _input _dim _sorted _return_inverse _return_counts

-- unique_consecutive :: Tensor device dtype shape -> Bool -> Bool -> Int -> (Tensor device dtype shape,Tensor device dtype shape,Tensor device dtype shape)
-- unique_consecutive _input _return_inverse _return_counts _dim = unsafePerformIO $ (cast4 ATen.unique_consecutive_tbbl) _input _return_inverse _return_counts _dim

-- unique_dim_consecutive :: Tensor device dtype shape -> Int -> Bool -> Bool -> (Tensor device dtype shape,Tensor device dtype shape,Tensor device dtype shape)
-- unique_dim_consecutive _input _dim _return_inverse _return_counts = unsafePerformIO $ (cast4 ATen.unique_dim_consecutive_tlbb) _input _dim _return_inverse _return_counts

-- | UnsqueezeImpl
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
unsqueeze input = unsafePerformIO $ cast2 ATen.unsqueeze_tl input (natValI @dim)

-- where' :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape
-- where' _condition _input _other = unsafePerformIO $ (cast3 ATen.where_ttt) _condition _input _other

-- where_ :: Tensor device dtype shape -> [Tensor device dtype shape]
-- where_ _condition = unsafePerformIO $ (cast1 ATen.where_t) _condition

-- norm_except_dim :: Tensor device dtype shape -> Int -> Int -> Tensor device dtype shape
-- norm_except_dim _v _pow _dim = unsafePerformIO $ (cast3 ATen.norm_except_dim_tll) _v _pow _dim

-- | zerosLike
-- >>> dtype &&& shape $ zerosLike (ones :: CPUTensor 'D.Float '[3,4,5])
-- (Float,[3,4,5])
zerosLike
  :: forall shape dtype device
   . Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
zerosLike input = unsafePerformIO $ cast1 ATen.zeros_like_t input

-- native_norm :: Tensor device dtype shape -> Float -> Tensor device dtype shape
-- native_norm _input _p = unsafePerformIO $ (cast2 ATen.native_norm_ts) _input _p

-- | clone
-- >>> t <- clone (ones :: CPUTensor 'D.Float '[3,2])
-- >>> dtype &&& shape $ t
-- (Float,[3,2])
clone
  :: forall shape dtype device
   . Tensor device dtype shape
  -> IO (Tensor device dtype shape)
clone input = cast1 ATen.clone_t input

-- s_native_addmm :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Float -> Float -> Tensor device dtype shape
-- s_native_addmm _input _mat1 _mat2 _beta _alpha = unsafePerformIO $ (cast5 ATen.s_native_addmm_tttss) _input _mat1 _mat2 _beta _alpha

-- | addmm
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
-- TODO: can we use D.Scalar here for beta and alpha?
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
addmm beta alpha mat1 mat2 input = unsafePerformIO $ cast5 ATen.addmm_tttss input mat1 mat2 beta alpha

-- hspmm :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape
-- hspmm _mat1 _mat2 = unsafePerformIO $ (cast2 ATen.hspmm_tt) _mat1 _mat2

-- | numel
-- TODO: since this is decidable at compile time, this should probably be calculated from the tensor type instead
numel
  :: forall shape dtype device
   . Tensor device dtype shape -- ^ input
  -> Int -- ^ output
numel input = unsafePerformIO $ cast1 ATen.numel_t input

-- unbind :: Tensor device dtype shape -> Int -> [Tensor device dtype shape]
-- unbind _input _dim = unsafePerformIO $ (cast2 ATen.unbind_tl) _input _dim

-- mkldnn_reorder_conv2d_weight :: Tensor device dtype shape -> (Int,Int) -> (Int,Int) -> (Int,Int) -> Int -> Tensor device dtype shape
-- mkldnn_reorder_conv2d_weight _input _padding _stride _dilation _groups = unsafePerformIO $ (cast5 ATen.mkldnn_reorder_conv2d_weight_tllll) _input _padding _stride _dilation _groups

--quantize_linear :: Tensor device dtype shape -> Double -> Int -> DType -> Tensor device dtype shape
--quantize_linear _input _scale _zero_point _dtype = unsafePerformIO $ (cast4 ATen.quantize_linear_tdls) _input _scale _zero_point _dtype

--quantize_linear_per_channel :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> [Int] -> DType -> Tensor device dtype shape
--quantize_linear_per_channel _input _scales _zero_points _axis _dtype = unsafePerformIO $ (cast5 ATen.quantize_linear_per_channel_tttls) _input _scales _zero_points _axis _dtype

-- dequantize :: Tensor device dtype shape -> Tensor device dtype shape
-- dequantize _input = unsafePerformIO $ (cast1 ATen.dequantize_t) _input

-- | qScale
-- TODO: are there any restrictions on the dtype?
qScale
  :: forall shape dtype device
   . Tensor device dtype shape -- ^ input
  -> Double -- ^ output
qScale input = unsafePerformIO $ cast1 ATen.q_scale_t input

-- | qZeroPoint
-- TODO: are there any restrictions on the dtype?
qZeroPoint
  :: forall shape dtype device
   . Tensor device dtype shape -- ^ input
  -> Int -- ^ output
qZeroPoint input = unsafePerformIO $ cast1 ATen.q_zero_point_t input

-- int_repr :: Tensor device dtype shape -> Tensor device dtype shape
-- int_repr _input = unsafePerformIO $ (cast1 ATen.int_repr_t) _input

-- fake_quantize_per_tensor_affine :: Tensor device dtype shape -> Double -> Int -> Int -> Int -> Tensor device dtype shape
-- fake_quantize_per_tensor_affine _input _scale _zero_point _quant_min _quant_max = unsafePerformIO $ (cast5 ATen.fake_quantize_per_tensor_affine_tdlll) _input _scale _zero_point _quant_min _quant_max

-- meshgrid :: [Tensor device dtype shape] -> [Tensor device dtype shape]
-- meshgrid _tensors = unsafePerformIO $ (cast1 ATen.meshgrid_l) _tensors

-- cartesian_prod :: [Tensor device dtype shape] -> Tensor device dtype shape
-- cartesian_prod _tensors = unsafePerformIO $ (cast1 ATen.cartesian_prod_l) _tensors

-- combinations :: Tensor device dtype shape -> Int -> Bool -> Tensor device dtype shape
-- combinations _input _r _with_replacement = unsafePerformIO $ (cast3 ATen.combinations_tlb) _input _r _with_replacement

-- | lstmCell
-- >>> dtype &&& shape $ fst $ lstmCell (ones :: CPUTensor 'D.Float '[12,2]) (ones :: CPUTensor 'D.Float '[12,3]) (ones :: CPUTensor 'D.Float '[12]) (ones :: CPUTensor 'D.Float '[12]) ((ones :: CPUTensor 'D.Float '[2,3]), (ones :: CPUTensor 'D.Float '[2,3])) (ones :: CPUTensor 'D.Float '[2,2])
-- (Float,[2,3])
lstmCell
  :: forall inputDim hiddenSize batchSize dtype device 
   . Tensor device dtype '[4 * hiddenSize, inputDim]
  -> Tensor device dtype '[4 * hiddenSize, hiddenSize]
  -> Tensor device dtype '[4 * hiddenSize]
  -> Tensor device dtype '[4 * hiddenSize]
  -> ( Tensor device dtype '[batchSize, hiddenSize]
     , Tensor device dtype '[batchSize, hiddenSize]
     )
  -> Tensor device dtype '[batchSize, inputDim]
  -> ( Tensor device dtype '[batchSize, hiddenSize]
     , Tensor device dtype '[batchSize, hiddenSize]
     )
lstmCell _w_ih _w_hh _b_ih _b_hh (_cc, _hc) _input =
  unsafePerformIO
    $ cast6 ATen.lstm_cell_tltttt _input _hx _w_ih _w_hh _b_ih _b_hh
  where _hx = [_cc, _hc] :: [Tensor device dtype '[batchSize, hiddenSize]]

-- gru_cell :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape
-- gru_cell _input _hx _w_ih _w_hh _b_ih _b_hh = unsafePerformIO $ (cast6 ATen.gru_cell_tttttt) _input _hx _w_ih _w_hh _b_ih _b_hh

-- rnn_tanh_cell :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape
-- rnn_tanh_cell _input _hx _w_ih _w_hh _b_ih _b_hh = unsafePerformIO $ (cast6 ATen.rnn_tanh_cell_tttttt) _input _hx _w_ih _w_hh _b_ih _b_hh

-- rnn_relu_cell :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape
-- rnn_relu_cell _input _hx _w_ih _w_hh _b_ih _b_hh = unsafePerformIO $ (cast6 ATen.rnn_relu_cell_tttttt) _input _hx _w_ih _w_hh _b_ih _b_hh

-- quantized_lstm :: Tensor device dtype shape -> [Tensor device dtype shape] -> [Tensor device dtype shape] -> Bool -> Int -> Double -> Bool -> Bool -> Bool -> DType -> (Tensor device dtype shape,Tensor device dtype shape,Tensor device dtype shape)
-- quantized_lstm _input _hx _params _has_biases _num_layers _dropout _train _bidirectional _batch_first _dtype = unsafePerformIO $ (cast10 ATen.quantized_lstm_tllbldbbbs) _input _hx _params _has_biases _num_layers _dropout _train _bidirectional _batch_first _dtype

-- quantized_lstm_cell :: Tensor device dtype shape -> [Tensor device dtype shape] -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Float -> Float -> Float -> Float -> (Tensor device dtype shape,Tensor device dtype shape)
-- quantized_lstm_cell _input _hx _w_ih _w_hh _b_ih _b_hh _packed_ih _packed_hh _col_offsets_ih _col_offsets_hh _scale_ih _scale_hh _zero_point_ih _zero_point_hh = unsafePerformIO $ (cast14 ATen.quantized_lstm_cell_tlttttttttssss) _input _hx _w_ih _w_hh _b_ih _b_hh _packed_ih _packed_hh _col_offsets_ih _col_offsets_hh _scale_ih _scale_hh _zero_point_ih _zero_point_hh

-- quantized_gru_cell :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Float -> Float -> Float -> Float -> Tensor device dtype shape
-- quantized_gru_cell _input _hx _w_ih _w_hh _b_ih _b_hh _packed_ih _packed_hh _col_offsets_ih _col_offsets_hh _scale_ih _scale_hh _zero_point_ih _zero_point_hh = unsafePerformIO $ (cast14 ATen.quantized_gru_cell_ttttttttttssss) _input _hx _w_ih _w_hh _b_ih _b_hh _packed_ih _packed_hh _col_offsets_ih _col_offsets_hh _scale_ih _scale_hh _zero_point_ih _zero_point_hh

-- quantized_rnn_relu_cell :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Float -> Float -> Float -> Float -> Tensor device dtype shape
-- quantized_rnn_relu_cell _input _hx _w_ih _w_hh _b_ih _b_hh _packed_ih _packed_hh _col_offsets_ih _col_offsets_hh _scale_ih _scale_hh _zero_point_ih _zero_point_hh = unsafePerformIO $ (cast14 ATen.quantized_rnn_relu_cell_ttttttttttssss) _input _hx _w_ih _w_hh _b_ih _b_hh _packed_ih _packed_hh _col_offsets_ih _col_offsets_hh _scale_ih _scale_hh _zero_point_ih _zero_point_hh

-- quantized_rnn_tanh_cell :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Float -> Float -> Float -> Float -> Tensor device dtype shape
-- quantized_rnn_tanh_cell _input _hx _w_ih _w_hh _b_ih _b_hh _packed_ih _packed_hh _col_offsets_ih _col_offsets_hh _scale_ih _scale_hh _zero_point_ih _zero_point_hh = unsafePerformIO $ (cast14 ATen.quantized_rnn_tanh_cell_ttttttttttssss) _input _hx _w_ih _w_hh _b_ih _b_hh _packed_ih _packed_hh _col_offsets_ih _col_offsets_hh _scale_ih _scale_hh _zero_point_ih _zero_point_hh

-- masked_scatter :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape
-- masked_scatter _input _mask _source = unsafePerformIO $ (cast3 ATen.masked_scatter_ttt) _input _mask _source

-- index_add :: Tensor device dtype shape -> Int -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape
-- index_add _input _dim _index _source = unsafePerformIO $ (cast4 ATen.index_add_tltt) _input _dim _index _source

-- scatter_add :: Tensor device dtype shape -> Int -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape
-- scatter_add _input _dim _index _src = unsafePerformIO $ (cast4 ATen.scatter_add_tltt) _input _dim _index _src

-- addbmm :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Float -> Float -> Tensor device dtype shape
-- addbmm _input _batch1 _batch2 _beta _alpha = unsafePerformIO $ (cast5 ATen.addbmm_tttss) _input _batch1 _batch2 _beta _alpha

-- cross :: Tensor device dtype shape -> Tensor device dtype shape -> Int -> Tensor device dtype shape
-- cross _input _other _dim = unsafePerformIO $ (cast3 ATen.cross_ttl) _input _other _dim

type family MatrixOrMatrixBatch (shape :: [Nat]) :: [Nat] where
  MatrixOrMatrixBatch (n : m : '[])     = '[n, m]
  MatrixOrMatrixBatch (b : n : m : '[]) = '[b, n, m]
  MatrixOrMatrixBatch _                 = TypeError (Text "The input must be matrix or a batch of matrices.")

-- | triu
-- TODO: triu is not implemented for D.Bool, or maybe numeric type is lifted?
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
triu diagonal input = unsafePerformIO $ cast2 ATen.triu_tl input diagonal

-- | tril
-- TODO: tril is not implemented for D.Bool, or maybe numeric type is lifted?
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
tril diagonal input = unsafePerformIO $ cast2 ATen.tril_tl input diagonal

-- trace :: Tensor device dtype shape -> Tensor device dtype shape
-- trace _input = unsafePerformIO $ (cast1 ATen.trace_t) _input

-- take :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape
-- take _input _index = unsafePerformIO $ (cast2 ATen.take_tt) _input _index

-- index_select :: Tensor device dtype shape -> Int -> Tensor device dtype shape -> Tensor device dtype shape
-- index_select _input _dim _index = unsafePerformIO $ (cast3 ATen.index_select_tlt) _input _dim _index

-- masked_select :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape
-- masked_select _input _mask = unsafePerformIO $ (cast2 ATen.masked_select_tt) _input _mask

-- nonzero :: Tensor device dtype shape -> Tensor device dtype shape
-- nonzero _input = unsafePerformIO $ (cast1 ATen.nonzero_t) _input

-- nonzero_numpy :: Tensor device dtype shape -> [Tensor device dtype shape]
-- nonzero_numpy _input = unsafePerformIO $ (cast1 ATen.nonzero_numpy_t) _input

-- gather :: Tensor device dtype shape -> Int -> Tensor device dtype shape -> Bool -> Tensor device dtype shape
-- gather _input _dim _index _sparse_grad = unsafePerformIO $ (cast4 ATen.gather_tltb) _input _dim _index _sparse_grad

-- addcmul :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Float -> Tensor device dtype shape
-- addcmul _input _tensor1 _tensor2 _value = unsafePerformIO $ (cast4 ATen.addcmul_ttts) _input _tensor1 _tensor2 _value

-- addcdiv :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Float -> Tensor device dtype shape
-- addcdiv _input _tensor1 _tensor2 _value = unsafePerformIO $ (cast4 ATen.addcdiv_ttts) _input _tensor1 _tensor2 _value

-- lstsq :: Tensor device dtype shape -> Tensor device dtype shape -> (Tensor device dtype shape,Tensor device dtype shape)
-- lstsq _input _A = unsafePerformIO $ (cast2 ATen.lstsq_tt) _input _A

-- triangular_solve :: Tensor device dtype shape -> Tensor device dtype shape -> Bool -> Bool -> Bool -> (Tensor device dtype shape,Tensor device dtype shape)
-- triangular_solve _input _A _upper _transpose _unitriangular = unsafePerformIO $ (cast5 ATen.triangular_solve_ttbbb) _input _A _upper _transpose _unitriangular

-- qr :: Tensor device dtype shape -> Bool -> (Tensor device dtype shape,Tensor device dtype shape)
-- qr _input _some = unsafePerformIO $ (cast2 ATen.qr_tb) _input _some

-- ormqr :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Bool -> Bool -> Tensor device dtype shape
-- ormqr _input _input2 _input3 _left _transpose = unsafePerformIO $ (cast5 ATen.ormqr_tttbb) _input _input2 _input3 _left _transpose

-- lu_solve :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape
-- lu_solve _input _LU_data _LU_pivots = unsafePerformIO $ (cast3 ATen.lu_solve_ttt) _input _LU_data _LU_pivots

-- | lgamma function
-- >>> dtype &&& shape $ lgamma (ones :: CPUTensor 'D.Float '[3,2])
-- (Float,[3,2])
lgamma
  :: forall shape dtype device
   . (StandardFloatingPointDTypeValidation device dtype)
  => Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
lgamma input = unsafePerformIO $ cast1 ATen.lgamma_t input

-- | digamma function
-- >>> dtype &&& shape $ digamma (ones :: CPUTensor 'D.Float '[3,2])
-- (Float,[3,2])
digamma
  :: forall shape dtype device
   . (StandardFloatingPointDTypeValidation device dtype)
  => Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
digamma input = unsafePerformIO $ cast1 ATen.digamma_t input

-- | polygamma function
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
polygamma
  :: forall shape dtype device
   . Int -- ^ order
  -> Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
polygamma n input = unsafePerformIO $ cast2 ATen.polygamma_lt n input

-- | inverse of the error function
-- >>> dtype &&& shape $ erfinv (ones :: CPUTensor 'D.Float '[3,2])
-- (Float,[3,2])
erfinv
  :: forall shape dtype device
   . (StandardFloatingPointDTypeValidation device dtype)
  => Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
erfinv input = unsafePerformIO $ cast1 ATen.erfinv_t input

-- dist :: Tensor device dtype shape -> Tensor device dtype shape -> Float -> Tensor device dtype shape
-- dist _input _other _p = unsafePerformIO $ (cast3 ATen.dist_tts) _input _other _p

-- atan2 :: Tensor device dtype shape -> Tensor device dtype shape -> Tensor device dtype shape
-- atan2 _input _other = unsafePerformIO $ (cast2 ATen.atan2_tt) _input _other

-- histc :: Tensor device dtype shape -> Int -> Float -> Float -> Tensor device dtype shape
-- histc _input _bins _min _max = unsafePerformIO $ (cast4 ATen.histc_tlss) _input _bins _min _max

-- | minAll
-- >>> dtype &&& shape $ minAll (ones :: CPUTensor 'D.Float '[2,2])
-- (Float,[])
minAll
  :: forall shape dtype device
   . Tensor device dtype shape -- ^ input
  -> Tensor device dtype '[] -- ^ output
minAll input = unsafePerformIO $ cast1 ATen.min_t input

type family DropValue (shape :: [Nat]) (i :: Nat) :: [Nat] where
  DropValue '[]     _ = TypeError (Text "Can not find a element in the list.")
  DropValue (x: xs) 0 = xs
  DropValue (x: xs) i = x ': DropValue xs (i-1)

-- | minDim
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
minDim input = unsafePerformIO $ cast2 ATen.min_tl input (natValI @d)

-- | maxAll
-- >>> dtype &&& shape $ maxAll (ones :: CPUTensor 'D.Float '[2,2])
-- (Float,[])
maxAll
  :: forall shape dtype device
   . Tensor device dtype shape -- ^ input
  -> Tensor device dtype '[] -- ^ output
maxAll input = unsafePerformIO $ cast1 ATen.max_t input

-- | maxDim
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
maxDim input = unsafePerformIO $ cast2 ATen.max_tl input (natValI @d)

-- | medianAll
-- >>> dtype &&& shape $ medianAll (ones :: CPUTensor 'D.Float '[2,2])
-- (Float,[])
medianAll
  :: forall shape dtype device
   . Tensor device dtype shape -- ^ input
  -> Tensor device dtype '[] -- ^ output
medianAll input = unsafePerformIO $ cast1 ATen.median_t input

-- | medianDim
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
medianDim input = unsafePerformIO $ cast2 ATen.median_tl input (natValI @d)

-- | median
-- See https://pytorch.org/docs/stable/torch.html#torch.median.
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
median' input = unsafePerformIO $ cast3 ATen.median_tlb
                                        input
                                        (natValI @dim)
                                        (keepOrDropDimVal @keepOrDropDim)

-- sort :: Tensor device dtype shape -> Int -> Bool -> (Tensor device dtype shape,Tensor device dtype shape)
-- sort _input _dim _descending = unsafePerformIO $ (cast3 ATen.sort_tlb) _input _dim _descending

-- argsort :: Tensor device dtype shape -> Int -> Bool -> Tensor device dtype shape
-- argsort _input _dim _descending = unsafePerformIO $ (cast3 ATen.argsort_tlb) _input _dim _descending

-- topk :: Tensor device dtype shape -> Int -> Int -> Bool -> Bool -> (Tensor device dtype shape,Tensor device dtype shape)
-- topk _input _k _dim _largest _sorted = unsafePerformIO $ (cast5 ATen.topk_tllbb) _input _k _dim _largest _sorted

-- renorm :: Tensor device dtype shape -> Float -> Int -> Float -> Tensor device dtype shape
-- renorm _input _p _dim _maxnorm = unsafePerformIO $ (cast4 ATen.renorm_tsls) _input _p _dim _maxnorm

-- equal :: Tensor device dtype shape -> Tensor device dtype shape -> Bool
-- equal _input _other = unsafePerformIO $ (cast2 ATen.equal_tt) _input _other

-- alias :: Tensor device dtype shape -> Tensor device dtype shape
-- alias _input = unsafePerformIO $ (cast1 ATen.alias_t) _input

-- | L1 loss
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
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
  $ cast3 ATen.l1_loss_ttl prediction target (reductionVal @reduction)

-- multi_margin_loss :: Tensor device dtype shape -> Tensor device dtype shape -> Float -> Float -> Tensor device dtype shape -> Int -> Tensor device dtype shape
-- multi_margin_loss _input _target _p _margin _weight _reduction = unsafePerformIO $ (cast6 ATen.multi_margin_loss_ttsstl) _input _target _p _margin _weight _reduction

-- multilabel_margin_loss :: Tensor device dtype shape -> Tensor device dtype shape -> Int -> Tensor device dtype shape
-- multilabel_margin_loss _input _target _reduction = unsafePerformIO $ (cast3 ATen.multilabel_margin_loss_ttl) _input _target _reduction

-- | negative log likelihood loss
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
-- See https://pytorch.org/docs/stable/nn.functional.html?highlight=nll_loss#torch.nn.functional.nll_loss.
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
  [] -> unsafePerformIO $ cast5 ATen.nll_loss_tttll
                                prediction
                                target
                                weight
                                (reductionVal @reduction)
                                ignoreIndex
  [_h, _w] -> unsafePerformIO $ cast5 ATen.nll_loss2d_tttll
                                      prediction
                                      target
                                      weight
                                      (reductionVal @reduction)
                                      ignoreIndex
  h : t -> case reductionVal @reduction of
    0 -> UnsafeMkTensor . D.reshape out $ (natValI @n) : h : t
    _ -> UnsafeMkTensor out
   where
    t'      = [1, foldl (*) h t]
    input'  = D.reshape (toDynamic prediction) (natValI @n : natValI @c : t')
    target' = D.reshape (toDynamic target) (natValI @n : t')
    out     = unsafePerformIO $ cast5 ATen.nll_loss2d_tttll
                                      input'
                                      target'
                                      weight
                                      (reductionVal @reduction)
                                      ignoreIndex

-- | smooth L1 loss
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
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
  $ cast3 ATen.smooth_l1_loss_ttl prediction target (reductionVal @reduction)

-- | soft margin loss
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
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
softMarginLoss prediciton target = unsafePerformIO $ cast3
  ATen.soft_margin_loss_ttl
  prediciton
  target
  (reductionVal @reduction)

-- | elu
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
-- >>> dtype &&& shape $ elu 0.1 0.1 0.3 (ones :: CPUTensor 'D.Float '[3,2])
-- (Float,[3,2])
elu
  :: forall shape dtype device
   . Float -- ^ alpha
  -> Float -- ^ scale
  -> Float -- ^ input scale
  -> Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
elu alpha scale inputScale input =
  unsafePerformIO $ cast4 ATen.elu_tsss input alpha scale inputScale

-- |
-- -- >>> dtype &&& shape $ glu (ones :: CPUTensor 'D.Float '[3,2]) 1
-- -- (Float,[3,1])
-- -- >>> dtype &&& shape $ glu (ones :: CPUTensor 'D.Float '[3,2]) 3
-- -- (Float,[3,2])
-- glu :: Tensor device dtype shape -> Int -> Tensor device dtype shape
-- glu _input _dim = unsafePerformIO $ (cast2 ATen.glu_tl) _input _dim

-- | hard tanh
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
-- >>> dtype &&& shape $ hardTanh 0 1 (ones :: CPUTensor 'D.Float '[3,2])
-- (Float,[3,2])
hardTanh
  :: forall shape dtype device
   . Float -- ^ minimum value
  -> Float -- ^ maximum value
  -> Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
hardTanh min_val max_val input =
  unsafePerformIO $ cast3 ATen.hardtanh_tss input min_val max_val

-- leaky_relu :: Tensor device dtype shape -> Float -> Tensor device dtype shape
-- leaky_relu _input _negative_slope = unsafePerformIO $ (cast2 ATen.leaky_relu_ts) _input _negative_slope

-- | logarithm of the sigmoid
-- >>> dtype &&& shape $ logSigmoid (ones :: CPUTensor 'D.Float '[3,2])
-- (Float,[3,2])
logSigmoid
  :: forall shape dtype device
   . (StandardFloatingPointDTypeValidation device dtype)
  => Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
logSigmoid input = unsafePerformIO $ cast1 ATen.log_sigmoid_t input

-- -- | softPlus
-- -- TODO: what's wrong with this, why is it commented?
-- -- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
-- -- >>> dtype &&& shape $ softPlus (ones :: CPUTensor 'D.Float '[3,2])
-- -- (Float,[3,2])
-- softPlus :: Tensor device dtype shape -> Float -> Float -> Tensor device dtype shape
-- softPlus _input _beta _threshold = unsafePerformIO $ (cast3 ATen.softplus_tss) _input _beta _threshold

-- | soft shrink
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
-- >>> dtype &&& shape $ softShrink 0.2 (ones :: CPUTensor 'D.Float '[3,2])
-- (Float,[3,2])
softShrink
  :: forall shape dtype device
   . Float -- ^ lambda
  -> Tensor device dtype shape -- ^ input
  -> Tensor device dtype shape -- ^ output
softShrink lambda input =
  unsafePerformIO $ cast2 ATen.softshrink_ts input lambda

-- | adaptive averaged 2-D pooling
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
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
                     , Fst outputSize, Snd outputSize
                     ]
     )
  => Tensor device dtype '[batchSize, channelSize, inputSize0, inputSize1] -- ^ input
  -> Tensor device dtype '[batchSize, channelSize, Fst outputSize, Snd outputSize] -- ^ output
adaptiveAvgPool2d input = unsafePerformIO $ cast2
  ATen.adaptive_avg_pool2d_tl
  input
  ([natValI @(Fst outputSize), natValI @(Snd outputSize)] :: [Int])

-- | MKLDNN adaptive averaged 2-D pooling
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
-- TODO: broken?
-- TODO: only defined for MKLDNN device?
-- TODO: test for availability of MKLDNN device?
-- TODO: merge with adaptiveAvgPool2d and dispatch based on (availability of MKLDNN) device in the function body?
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
                     , Fst outputSize, Snd outputSize
                     ]
     )
  => Tensor device dtype '[batchSize, channelSize, inputSize0, inputSize1] -- ^ input
  -> Tensor device dtype '[batchSize, channelSize, Fst outputSize, Snd outputSize] -- ^ output
mkldnnAdaptiveAvgPool2d input = unsafePerformIO $ cast2
  ATen.adaptive_avg_pool2d_tl
  input
  ([natValI @(Fst outputSize), natValI @(Snd outputSize)] :: [Int])

-- | adaptive averaged 3-D pooling
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
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
adaptiveAvgPool3d input = unsafePerformIO $ cast2
  ATen.adaptive_avg_pool3d_tl
  input
  ([ natValI @(Fst3 outputSize)
   , natValI @(Snd3 outputSize)
   , natValI @(Trd3 outputSize)
   ] :: [Int]
  )

-- | adaptive 2-D max-pool
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
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
                     , Fst outputSize, Snd outputSize
                     ]
     )
  => Tensor device dtype '[batchSize, channelSize, inputSize0, inputSize1] -- ^ input
  -> ( Tensor device dtype    '[batchSize, channelSize, Fst outputSize, Snd outputSize]
     , Tensor device 'D.Int64 '[batchSize, channelSize, Fst outputSize, Snd outputSize]
     ) -- ^ output
adaptiveMaxPool2d input = unsafePerformIO $ cast2
  ATen.adaptive_max_pool2d_tl
  input
  ([natValI @(Fst outputSize), natValI @(Snd outputSize)] :: [Int])

-- | adaptive 3-D max-pool
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
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
adaptiveMaxPool3d input = unsafePerformIO $ (cast2 ATen.adaptive_max_pool3d_tl)
  input
  ([ natValI @(Fst3 outputSize)
   , natValI @(Snd3 outputSize)
   , natValI @(Trd3 outputSize)
   ] :: [Int]
  )

-- | averaged 2-D pooling
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
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
   . ( All KnownNat '[ Fst kernelSize, Snd kernelSize
                     , Fst stride, Snd stride
                     , Fst padding, Snd padding
                     , channelSize
                     , inputSize0, inputSize1
                     , batchSize
                     ]
     , ConvSideCheck inputSize0 (Fst kernelSize) (Fst stride) (Fst padding) outputSize0
     , ConvSideCheck inputSize1 (Snd kernelSize) (Snd stride) (Snd padding) outputSize1
     )
  => Tensor device dtype '[batchSize, channelSize, inputSize0, inputSize1] -- ^ input
  -> Tensor device dtype '[batchSize, channelSize, outputSize0, outputSize1] -- ^ output
avgPool2d input = unsafePerformIO $ cast7
  ATen.avg_pool2d_tlllbbl
  input
  ([natValI @(Fst kernelSize), natValI @(Snd kernelSize)] :: [Int])
  ([natValI @(Fst stride), natValI @(Snd stride)] :: [Int])
  ([natValI @(Fst padding), natValI @(Snd padding)] :: [Int])
  False
  True
  (1 :: Int)

-- | averaged 3-D pooling
-- TODO: probably only defined for floating point tensors, or maybe numeric type is lifted?
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
avgPool3d input = unsafePerformIO $ cast7
  ATen.avg_pool3d_tlllbbl
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
-- fractional_max_pool2d _input _kernel_size _output_size _random_samples = unsafePerformIO $ (cast4 ATen.fractional_max_pool2d_tllt) _input _kernel_size _output_size _random_samples

-- fractional_max_pool3d :: Tensor device dtype shape -> (Int,Int,Int) -> (Int,Int,Int) -> Tensor device dtype shape -> (Tensor device dtype shape,Tensor device dtype shape)
-- fractional_max_pool3d _input _kernel_size _output_size _random_samples = unsafePerformIO $ (cast4 ATen.fractional_max_pool3d_tllt) _input _kernel_size _output_size _random_samples

-- max_pool2d_with_indices :: Tensor device dtype shape -> (Int,Int) -> (Int,Int) -> (Int,Int) -> (Int,Int) -> Bool -> (Tensor device dtype shape,Tensor device dtype shape)
-- max_pool2d_with_indices _input _kernel_size _stride _padding _dilation _ceil_mode = unsafePerformIO $ (cast6 ATen.max_pool2d_with_indices_tllllb) _input _kernel_size _stride _padding _dilation _ceil_mode

-- max_pool3d_with_indices :: Tensor device dtype shape -> (Int,Int,Int) -> (Int,Int,Int) -> (Int,Int,Int) -> (Int,Int,Int) -> Bool -> (Tensor device dtype shape,Tensor device dtype shape)
-- max_pool3d_with_indices _input _kernel_size _stride _padding _dilation _ceil_mode = unsafePerformIO $ (cast6 ATen.max_pool3d_with_indices_tllllb) _input _kernel_size _stride _padding _dilation _ceil_mode

-- max_unpool2d :: Tensor device dtype shape -> Tensor device dtype shape -> (Int,Int) -> Tensor device dtype shape
-- max_unpool2d _input _indices _output_size = unsafePerformIO $ (cast3 ATen.max_unpool2d_ttl) _input _indices _output_size

-- max_unpool3d :: Tensor device dtype shape -> Tensor device dtype shape -> (Int,Int,Int) -> (Int,Int,Int) -> (Int,Int,Int) -> Tensor device dtype shape
-- max_unpool3d _input _indices _output_size _stride _padding = unsafePerformIO $ (cast5 ATen.max_unpool3d_ttlll) _input _indices _output_size _stride _padding

-- reflection_pad1d :: Tensor device dtype shape -> (Int,Int) -> Tensor device dtype shape
-- reflection_pad1d _input _padding = unsafePerformIO $ (cast2 ATen.reflection_pad1d_tl) _input _padding

-- reflection_pad2d :: Tensor device dtype shape -> (Int,Int,Int,Int) -> Tensor device dtype shape
-- reflection_pad2d _input _padding = unsafePerformIO $ (cast2 ATen.reflection_pad2d_tl) _input _padding

-- replication_pad1d :: Tensor device dtype shape -> (Int,Int) -> Tensor device dtype shape
-- replication_pad1d _input _padding = unsafePerformIO $ (cast2 ATen.replication_pad1d_tl) _input _padding

-- replication_pad2d :: Tensor device dtype shape -> (Int,Int,Int,Int) -> Tensor device dtype shape
-- replication_pad2d _input _padding = unsafePerformIO $ (cast2 ATen.replication_pad2d_tl) _input _padding

-- replication_pad3d :: Tensor device dtype shape -> (Int,Int,Int,Int,Int,Int) -> Tensor device dtype shape
-- replication_pad3d _input _padding = unsafePerformIO $ (cast2 ATen.replication_pad3d_tl) _input _padding

-- upsample_linear1d :: Tensor device dtype shape -> Int -> Bool -> Tensor device dtype shape
-- upsample_linear1d _input _output_size _align_corners = unsafePerformIO $ (cast3 ATen.upsample_linear1d_tlb) _input _output_size _align_corners

-- upsample_bilinear2d :: Tensor device dtype shape -> (Int,Int) -> Bool -> Tensor device dtype shape
-- upsample_bilinear2d _input _output_size _align_corners = unsafePerformIO $ (cast3 ATen.upsample_bilinear2d_tlb) _input _output_size _align_corners

-- upsample_bicubic2d :: Tensor device dtype shape -> (Int,Int) -> Bool -> Tensor device dtype shape
-- upsample_bicubic2d _input _output_size _align_corners = unsafePerformIO $ (cast3 ATen.upsample_bicubic2d_tlb) _input _output_size _align_corners

-- upsample_trilinear3d :: Tensor device dtype shape -> (Int,Int,Int) -> Bool -> Tensor device dtype shape
-- upsample_trilinear3d _input _output_size _align_corners = unsafePerformIO $ (cast3 ATen.upsample_trilinear3d_tlb) _input _output_size _align_corners

-- upsample_nearest1d :: Tensor device dtype shape -> Int -> Tensor device dtype shape
-- upsample_nearest1d _input _output_size = unsafePerformIO $ (cast2 ATen.upsample_nearest1d_tl) _input _output_size

-- upsample_nearest2d :: Tensor device dtype shape -> (Int,Int) -> Tensor device dtype shape
-- upsample_nearest2d _input _output_size = unsafePerformIO $ (cast2 ATen.upsample_nearest2d_tl) _input _output_size

-- upsample_nearest3d :: Tensor device dtype shape -> (Int,Int,Int) -> Tensor device dtype shape
-- upsample_nearest3d _input _output_size = unsafePerformIO $ (cast2 ATen.upsample_nearest3d_tl) _input _output_size

-- conv_dilated2d :: Tensor device dtype shape -> Tensor device dtype shape -> (Int,Int) -> Tensor device dtype shape -> (Int,Int) -> (Int,Int) -> (Int,Int) -> Tensor device dtype shape
-- conv_dilated2d _input _weight _kernel_size _bias _stride _padding _dilation = unsafePerformIO $ (cast7 ATen.conv_dilated2d_ttltlll) _input _weight _kernel_size _bias _stride _padding _dilation

-- conv_dilated3d :: Tensor device dtype shape -> Tensor device dtype shape -> (Int,Int,Int) -> Tensor device dtype shape -> (Int,Int,Int) -> (Int,Int,Int) -> (Int,Int,Int) -> Tensor device dtype shape
-- conv_dilated3d _input _weight _kernel_size _bias _stride _padding _dilation = unsafePerformIO $ (cast7 ATen.conv_dilated3d_ttltlll) _input _weight _kernel_size _bias _stride _padding _dilation

-- col2im :: Tensor device dtype shape -> (Int,Int) -> (Int,Int) -> (Int,Int) -> (Int,Int) -> (Int,Int) -> Tensor device dtype shape
-- col2im _input _output_size _kernel_size _dilation _padding _stride = unsafePerformIO $ (cast6 ATen.col2im_tlllll) _input _output_size _kernel_size _dilation _padding _stride

-- im2col :: Tensor device dtype shape -> (Int,Int) -> (Int,Int) -> (Int,Int) -> (Int,Int) -> Tensor device dtype shape
-- im2col _input _kernel_size _dilation _padding _stride = unsafePerformIO $ (cast5 ATen.im2col_tllll) _input _kernel_size _dilation _padding _stride
