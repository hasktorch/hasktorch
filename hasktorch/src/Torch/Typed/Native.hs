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
                                                )
import           Data.Finite
import qualified Data.Int                      as I
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
import qualified Torch.Scalar                  as D
import           Torch.Functions                ( Reduction(..)
                                                , Tri(..)
                                                , isUpper
                                                , kOne
                                                )
import           Torch.Typed.Device
import           Torch.Typed.Factories
import           Torch.Typed.Tensor


type family DTypeIsNotHalf (dtype :: D.DType) :: Constraint where
  DTypeIsNotHalf D.Half = TypeError (Text "This operation does not support " :<>: ShowType D.Half :<>: Text " tensors.")
  DTypeIsNotHalf _      = ()

type family SumDType (dtype :: D.DType) :: D.DType where
  SumDType D.Bool = D.Int64
  SumDType D.UInt8 = D.Int64
  SumDType D.Int8 = D.Int64
  SumDType D.Int16 = D.Int64
  SumDType D.Int32 = D.Int64
  SumDType D.Int64 = D.Int64
  SumDType D.Float = D.Float
  SumDType D.Double = D.Double

-- | sumAll
-- >>> dtype &&& shape &&& (\t' -> D.asValue (toDynamic t') :: Int) $ sumAll (ones @'D.Bool @'[2, 3])
-- (Int64,([],6))
-- >>> dtype &&& shape &&& (\t' -> D.asValue (toDynamic t') :: Int) $ sumAll (ones @'D.UInt8 @'[2, 3])
-- (Int64,([],6))
-- >>> dtype &&& shape &&& (\t' -> D.asValue (toDynamic t') :: Int) $ sumAll (ones @'D.Int8 @'[2, 3])
-- (Int64,([],6))
-- >>> dtype &&& shape &&& (\t' -> D.asValue (toDynamic t') :: Int) $ sumAll (ones @'D.Int16 @'[2, 3])
-- (Int64,([],6))
-- >>> dtype &&& shape &&& (\t' -> D.asValue (toDynamic t') :: Int) $ sumAll (ones @'D.Int32 @'[2, 3])
-- (Int64,([],6))
-- >>> dtype &&& shape &&& (\t' -> D.asValue (toDynamic t') :: Int) $ sumAll (ones @'D.Int64 @'[2, 3])
-- (Int64,([],6))
-- >>> dtype &&& shape &&& (\t' -> D.asValue (toDynamic t') :: Float) $ sumAll (ones @'D.Float @'[2, 3])
-- (Float,([],6.0))
-- >>> dtype &&& shape &&& (\t' -> D.asValue (toDynamic t') :: Double) $ sumAll (ones @'D.Double @'[2, 3])
-- (Double,([],6.0))
sumAll
  :: forall device dtype shape
   . (DTypeIsNotHalf dtype)
  => Tensor device dtype shape
  -> Tensor device (SumDType dtype) '[]
sumAll input = unsafePerformIO $ cast1 ATen.sum_t input

-- | sumDim
-- >>> dtype &&& shape $ sumDim @0 (ones :: Tensor 'D.Float '[3,4,5])
-- (Float,[4,5])
-- >>> sumDim @1 (ones :: Tensor 'D.Float '[2,4])
-- Tensor Float [2] [ 4.0000   ,  4.0000   ]
sumDim
  :: forall d device dtype shape
   . (KnownNat d)
  => Tensor device dtype shape
  -> Tensor device dtype (DropValue shape d)
sumDim input = unsafePerformIO $ cast2 ATen.sum_tl input (natValI @d)

-- | abs
-- >>> dtype &&& shape $ abs (ones :: Tensor 'D.Float '[2,2])
-- (Float,[2,2])
abs
  :: forall device dtype shape
   . Tensor device dtype shape
  -> Tensor device dtype shape
abs input = unsafePerformIO $ cast1 ATen.abs_t input

-- | floor
ceil
  :: forall device dtype shape
   . Tensor device dtype shape
  -> Tensor device dtype shape
ceil input = unsafePerformIO $ cast1 ATen.ceil_t input

-- | floor
floor
  :: forall device dtype shape
   . Tensor device dtype shape
  -> Tensor device dtype shape
floor input = unsafePerformIO $ cast1 ATen.floor_t input

-- | min
-- >>> dtype &&& shape $ min (ones :: Tensor 'D.Float '[2,2])
-- (Float,[])
min
  :: forall device dtype shape
   . Tensor device dtype shape
  -> Tensor device dtype '[]
min input = unsafePerformIO $ cast1 ATen.min_t input

-- | max
-- >>> dtype &&& shape $ max (ones :: Tensor 'D.Float '[2,2])
-- (Float,[])
max
  :: forall device dtype shape
   . Tensor device dtype shape
  -> Tensor device  dtype '[]
max input = unsafePerformIO $ cast1 ATen.max_t input

-- | median
-- >>> dtype &&& shape $ median (ones :: Tensor 'D.Float '[2,2])
-- (Float,[])
median
  :: forall device dtype shape
   . Tensor device dtype shape
  -> Tensor device dtype '[]
median input = unsafePerformIO $ cast1 ATen.median_t input

-- | cmul
cmul
  :: forall a device dtype shape
   . D.Scalar a
  => a
  -> Tensor device dtype shape
  -> Tensor device dtype shape
cmul a input = unsafePerformIO $ cast2 ATen.mul_ts input a

-- | erf
-- >>> dtype &&& shape $ erf (ones :: Tensor 'D.Float '[3,2])
-- (Float,[3,2])
erf
  :: forall device dtype shape
   . Tensor device dtype shape
  -> Tensor device dtype shape
erf input = unsafePerformIO $ cast1 ATen.erf_t 

-- | exp
-- >>> dtype &&& shape $ exp (ones :: Tensor 'D.Float '[3,2])
-- (Float,[3,2])
exp
  :: forall device dtype shape
   . Tensor device dtype shape
  -> Tensor device dtype shape
exp input = unsafePerformIO $ cast1 ATen.exp_t input

-- | log1p
-- >>> dtype &&& shape $ log1p (ones :: Tensor 'D.Float '[3,2])
-- (Float,[3,2])
log1p
  :: forall device dtype shape
   . Tensor device dtype shape
  -> Tensor device dtype shape
log1p input = unsafePerformIO $ cast1 ATen.log1p_t input

-- | log2
-- >>> dtype &&& shape $ log2 (ones :: Tensor 'D.Float '[3,2])
-- (Float,[3,2])
log2
  :: forall device dtype shape
   . Tensor device dtype shape
  -> Tensor device dtype shape
log2 input = unsafePerformIO $ cast1 ATen.log2_t input

-- | log10
-- >>> dtype &&& shape $ log10 (ones :: Tensor 'D.Float '[3,2])
-- (Float,[3,2])
log10
  :: forall device dtype shape
   . Tensor device dtype shape
  -> Tensor device dtype shape
log10 input = unsafePerformIO $ cast1 ATen.log10_t input

-- | pow
-- >>> dtype &&& shape $ pow 2 (ones @'D.Float @'[3,2])
-- (Float,[3,2])
pow
  :: forall a device dtype shape
   . D.Scalar a
  => a
  -> Tensor device dtype shape
  -> Tensor device dtype shape
pow a input = unsafePerformIO $ cast2 ATen.pow_ts input a

-- | relu
-- >>> dtype &&& shape $ relu (ones :: Tensor 'D.Float '[3,2])
-- (Float,[3,2])
relu
  :: forall device dtype shape
   . Tensor device dtype shape
  -> Tensor device dtype shape
relu input = unsafePerformIO $ cast1 ATen.relu_t input

-- | selu
-- >>> dtype &&& shape $ selu (ones :: Tensor 'D.Float '[3,2])
-- (Float,[3,2])
selu
  :: forall device dtype shape
   . Tensor device dtype shape
  -> Tensor device dtype shape
selu input = unsafePerformIO $ cast1 ATen.selu_t input

-- | sigmoid
-- >>> dtype &&& shape $ sigmoid (ones :: Tensor 'D.Float '[3,2])
-- (Float,[3,2])
sigmoid
  :: forall device dtype shape
   . Tensor device dtype shape
  -> Tensor device dtype shape
sigmoid input = unsafePerformIO $ cast1 ATen.sigmoid_t input

-- | sin
-- >>> dtype &&& shape $ sin (ones :: Tensor 'D.Float '[3,2])
-- (Float,[3,2])
sin
  :: forall device dtype shape
   . Tensor device dtype shape
  -> Tensor device dtype shape
sin input = unsafePerformIO $ cast1 ATen.sin_t input

-- | sinh
-- >>> dtype &&& shape $ sinh (ones :: Tensor 'D.Float '[3,2])
-- (Float,[3,2])
sinh
  :: forall device dtype shape
   . Tensor device dtype shape
  -> Tensor device dtype shape
sinh input = unsafePerformIO $ cast1 ATen.sinh_t input

-- | cos
-- >>> dtype &&& shape $ cos (ones :: Tensor 'D.Float '[3,2])
-- (Float,[3,2])
cos
  :: forall device dtype shape
   . Tensor device dtype shape
  -> Tensor device dtype shape
cos input = unsafePerformIO $ cast1 ATen.cos_t input

-- | sqrt
sqrt
  :: forall device dtype shape
   . Tensor device dtype shape
  -> Tensor device dtype shape
sqrt input = unsafePerformIO $ cast1 ATen.sqrt_t input

-- | tanh
tanh 
  :: forall device dtype shape
   . Tensor device dtype shape
  -> Tensor device dtype shape
tanh input = unsafePerformIO $ cast1 ATen.tanh_t input

-- | toDType
-- >>> dtype &&& shape $ (toDType (ones :: Tensor 'D.Float '[2,2]) :: Tensor 'D.Double '[2,2])
-- (Double,[2,2])
toDType
  :: forall dtype' device dtype shape
   . (KnownDType dtype')
  => Tensor device dtype shape
  -> Tensor device dtype' shape
toDType input = unsafePerformIO $ cast4 ATen.tensor_to_sbb input (dtypeVal @dtype') False False

type family SqueezeAll (shape :: [Nat]) :: [Nat] where
  SqueezeAll '[] = '[]
  SqueezeAll (1: xs) = SqueezeAll xs
  SqueezeAll (x: xs) = x ': SqueezeAll xs

-- | squeezeAll
-- >>> dtype &&& shape $ (squeezeAll (ones :: Tensor 'D.Float '[2,1,2,1,2]) :: Tensor 'D.Float '[2,2,2])
-- (Float,[2,2,2])
squeezeAll
  :: forall device dtype shape
   . Tensor device dtype shape
  -> Tensor device dtype (SqueezeAll shape)
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
-- >>> tt = ones :: Tensor 'D.Float '[2,2]
-- >>> dtype &&& shape $ (binary_cross_entropy @ReduceNone tt tt tt :: Tensor 'D.Float '[2,2])
-- (Float,[2,2])
-- >>> dtype &&& shape $ (binary_cross_entropy @ReduceMean tt tt tt :: Tensor 'D.Float '[])
-- (Float,[])
-- >>> dtype &&& shape $ (binary_cross_entropy @ReduceSum tt tt tt :: Tensor 'D.Float '[])
-- (Float,[])
binaryCrossEntropy
  :: forall (reduction :: Reduction) device dtype shape
   . (KnownReduction reduction)
  => Tensor device dtype shape
  -> Tensor device dtype shape
  -> Tensor device dtype shape
  -> Tensor device dtype (ConditionalReduction shape reduction)
binaryCrossEntropy weight predictions targets = unsafePerformIO $ cast4
  ATen.binary_cross_entropy_tttl
  input
  target
  weight
  (reductionVal @reduction)

-- | mseLoss
-- >>> dtype &&& shape $ mse_loss (ones :: Tensor 'D.Float '[2,2]) (ones :: Tensor 'D.Float '[2,2])
-- (Float,[])
mseLoss
  :: forall device dtype shape
   . Tensor device dtype shape
  -> Tensor device dtype shape
  -> Tensor device dtype '[]
mseLoss a b = unsafePerformIO $ cast3 ATen.mse_loss_ttl a b ATen.kMean

-- | softmax
-- >>> dtype &&& shape $ softmax @0 (ones @D.Float @[2,2])
-- (Float,[2,2])
-- >>> dtype &&& shape $ softmax @1 (ones @D.Float @[2,2])
-- (Float,[2,2])
softmax
  :: forall dim device dtype shape
   . (KnownNat dim, KnownDType dtype, DimOutOfBoundCheck shape dim)
  => Tensor device dtype shape
  -> Tensor device dtype shape
softmax input = unsafePerformIO
  $ cast3 ATen.softmax_tls input (natValI @dim) (dtypeVal @dtype)

-- | logSoftmax
-- >>> dtype &&& shape $ logSoftmax @0 (ones @D.Float @[2,2])
-- (Float,[2,2])
-- >>> dtype &&& shape $ logSoftmax @1 (ones @D.Float @[2,2])
-- (Float,[2,2])
logSoftmax
  :: forall dim device dtype shape
   . (KnownNat dim, KnownDType dtype, DimOutOfBoundCheck shape dim)
  => Tensor device dtype shape
  -> Tensor device dtype shape
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

type family FstDim (shape :: [Nat]) :: Nat where
  FstDim (n:m:'[])   = n
  FstDim (b:n:m:'[]) = n
  FstDim _           = TypeError (Text "Can not get first dimention of matrix or batch + matrix.")

-- | inverse
-- >>> t <- randn :: IO (Tensor 'D.Float '[3,2,2])
-- >>> dtype &&& shape $ inverse t
-- (Float,[3,2,2])
-- >>> t <- randn :: IO (Tensor 'D.Float '[2,2])
-- >>> dtype &&& shape $ inverse t
-- (Float,[2,2])
inverse
  :: forall device dtype shape
   . Tensor device dtype shape
  -> Tensor device dtype (Square shape)
inverse input = unsafePerformIO $ cast1 ATen.inverse_t input

-- | symeig
-- >>> (eigenVals,eigenVecs) = symeig (ones :: Tensor 'D.Float '[3,2,2]) True Upper
-- >>> dtype &&& shape $ eigenVals
-- (Float,[3,2])
-- >>> :t eigenVals
-- eigenVals :: Tensor 'D.Float '[3, 2]
-- >>> dtype &&& shape $ eigenVecs
-- (Float,[3,2,2])
-- >>> :t eigenVecs
-- eigenVecs :: Tensor 'D.Float '[3, 2, 2]
-- >>> (eigenVals,eigenVecs) = symeig (ones :: Tensor 'D.Float '[3,2,2]) False Upper
-- >>> dtype &&& shape $ eigenVals
-- (Float,[3,2])
-- >>> dtype &&& shape $ eigenVecs
-- (Float,[3,2,2])
symeig
  :: forall device dtype shape
   . Bool
  -> Tri
  -> Tensor device dtype shape
  -> ( Tensor device dtype (VectorOfSquare shape)
     , Tensor device dtype (Square shape)
     )
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
  ConditionalEigenVectors EnableEigenVectors n = '[n,n]
  ConditionalEigenVectors DisableEigenVectors _ = '[0]

-- | eig
-- >>> (eigenVals,eigenVecs) = eig @EnableEigenVectors (ones :: Tensor 'D.Float '[3,3])
-- >>> dtype &&& shape $ eigenVals
-- (Float,[3,2])
-- >>> :t eigenVals
-- eigenVals :: Tensor 'D.Float '[3, 2]
-- >>> dtype &&& shape $ eigenVecs
-- (Float,[3,3])
-- >>> :t eigenVecs
-- eigenVecs :: Tensor 'D.Float '[3, 3]
-- >>> (eigenVals,eigenVecs) = eig @DisableEigenVectors (ones :: Tensor 'D.Float '[3,3])
-- >>> dtype &&& shape $ eigenVals
-- (Float,[3,2])
-- >>> dtype &&& shape $ eigenVecs
-- (Float,[0])
-- >>> :t eigenVecs
-- eigenVecs :: Tensor 'D.Float '[0]
eig
  :: forall eigenvectors device dtype n
   . (KnownNat n, KnownEigenVectors eigenvectors)
  => Tensor device dtype '[n, n]
  -> ( Tensor device dtype '[n, 2]
     , Tensor device dtype (ConditionalEigenVectors eigenvectors n)
     )
eig input =
  unsafePerformIO $ cast2 ATen.eig_tb input (enableEigenVectors @eigenvectors)

-- svd :: Tensor dtype shape -> Bool -> Bool -> (Tensor dtype shape, Tensor dtype shape, Tensor dtype shape)
-- svd t some compute_uv = unsafePerformIO $ (cast3 ATen.svd_tbb) t some compute_uv

-- | cholesky
-- >>> t <- rand :: IO (Tensor 'D.Float '[2,2])
-- >>> c = cholesky (t `matmul` transpose2D t) Upper
-- >>> dtype &&& shape $ c
-- (Float,[2,2])
-- >>> :t c
-- c :: Tensor 'D.Float '[2, 2]
cholesky
  :: forall device dtype shape
   . Tri
  -> Tensor device dtype shape
  -> Tensor device dtype (Square shape)
cholesky upper input = unsafePerformIO $ cast2 ATen.cholesky_tb input boolUpper
  where boolUpper = isUpper upper

-- cholesky_solve :: Tensor dtype shape -> Tensor dtype shape -> Tri -> Tensor dtype shape
-- cholesky_solve t1 t2 upper = unsafePerformIO $ (cast3 ATen.cholesky_solve_ttb) t1 t2 boolUpper
--   where boolUpper = isUpper upper

-- | solve
-- >>> a <- rand :: IO (Tensor 'D.Float '[10,10])
-- >>> b <- rand :: IO (Tensor 'D.Float '[10,3])
-- >>> (c,lu) = solve b a
-- >>> dtype &&& shape $ c
-- (Float,[10,3])
-- >>> dtype &&& shape $ lu
-- (Float,[10,10])
-- >>> :t c
-- c :: Tensor 'D.Float '[10, 3]
-- >>> :t lu
-- lu :: Tensor 'D.Float '[10, 10]
solve
  :: forall device dtype m_k m_m
   . (Square m_m ~ m_m, FstDim m_m ~ FstDim m_k)
  => Tensor device dtype m_k
  -> Tensor device dtype m_m
  -> (Tensor device dtype m_k, Tensor device dtype m_m)
solve b a = unsafePerformIO $ cast2 ATen.solve_tt b a

-- cholesky_inverse :: Tensor dtype shape -> Tri -> Tensor dtype shape
-- cholesky_inverse t upper = unsafePerformIO $ (cast2 ATen.cholesky_inverse_tb) t boolUpper
--   where boolUpper = isUpper upper

-- geqrf :: Tensor dtype shape -> (Tensor dtype shape, Tensor dtype shape)
-- geqrf t = unsafePerformIO $ (cast1 ATen.geqrf_t) t

-- orgqr :: Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape
-- orgqr b a = unsafePerformIO $ (cast2 ATen.orgqr_tt) b a

-- | sign
-- >>> dtype &&& shape $ sign (ones :: Tensor 'D.Float '[3,2])
-- (Float,[3,2])
sign :: forall device dtype shape . Tensor dtype shape -> Tensor dtype shape
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
-- See ../../../../deps/pytorch/aten/src/ATen/native/TensorShape.cpp
-- >>> dtype &&& shape $ transpose @0 @1 (ones :: Tensor 'D.Float '[3,2])
-- (Float,[2,3])
-- >>> dtype &&& shape $ transpose @0 @1 (ones :: Tensor 'D.Float '[3,2,1])
-- (Float,[2,3,1])
-- >>> dtype &&& shape $ transpose @1 @2 (ones :: Tensor 'D.Float '[3,2,1])
-- (Float,[3,1,2])
transpose
  :: forall n m device dtype shape
   . (KnownNat n, KnownNat m)
  => Tensor device dtype shape
  -> Tensor device dtype (Transpose shape n m)
transpose input =
  unsafePerformIO $ cast3 ATen.transpose_tll t (natValI @n) (natValI @m)

-- | transpose2d, special case for a 2D tensor
-- >>> dtype &&& shape $ transpose2D (ones :: Tensor 'D.Float '[3,2])
-- (Float,[2,3])
transpose2D
  :: forall (i :: Nat) (j :: Nat) device dtype
   . Tensor device dtype '[i, j]
  -> Tensor device dtype '[j, i]
transpose2D = transpose @0 @1

-- diag :: Tensor dtype shape -> Int -> Tensor dtype shape
-- diag t index = unsafePerformIO $ (cast2 ATen.tensor_diag_l) t index

-- | all
-- See https://pytorch.org/docs/stable/tensors.html#torch.BoolTensor.all.
-- >>> t = all (fromJust [False, False] :: Tensor 'D.Bool '[2])
-- >>> toInt t == 1
-- False
--
-- >>> t = all (fromJust [False, True] :: Tensor 'D.Bool '[2])
-- >>> toInt t == 1
-- False
--
-- >>> t = all (fromJust [True, True] :: Tensor 'D.Bool '[2])
-- >>> toInt t == 1
-- True
all :: forall device shape . Tensor device 'D.Bool shape -> Tensor device 'D.Bool '[]
all input = unsafePerformIO $ cast1 ATen.all_t input

-- | any
-- See https://pytorch.org/docs/stable/tensors.html#torch.BoolTensor.any.
-- >>> t = any (fromJust [False, False] :: Tensor 'D.Bool '[2])
-- >>> toInt t == 1
-- False
--
-- >>> t = any (fromJust [False, True] :: Tensor 'D.Bool '[2])
-- >>> toInt t == 1
-- True
--
-- >>> t = any (fromJust [True, True] :: Tensor 'D.Bool '[2])
-- >>> toInt t == 1
-- True
any
  :: forall device shape
   . Tensor device 'D.Bool shape
  -> Tensor device 'D.Bool '[]
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
-- >>> t = fromJust [[True, True], [True, False], [True, True], [True, True]] :: Tensor 'D.Bool '[4, 2]
--
-- >>> dtype &&& shape &&& (\t' -> D.asValue (toDynamic t') :: [Bool]) $ (all' @1 @DropDim t :: Tensor 'D.Bool '[4])
-- (Bool,([4],[True,False,True,True]))
--
-- >>> dtype &&& shape &&& (\t' -> D.asValue (toDynamic t') :: [[Bool]]) $ (all' @1 @KeepDim t :: Tensor 'D.Bool '[4, 1])
-- (Bool,([4,1],[[True],[False],[True],[True]]))
--
-- >>> dtype &&& shape &&& (\t' -> D.asValue (toDynamic t') :: [Bool]) $ (all' @0 @DropDim t :: Tensor 'D.Bool '[2])
-- (Bool,([2],[True,False]))
--
-- >>> dtype &&& shape &&& (\t' -> D.asValue (toDynamic t') :: [[Bool]]) $ (all' @0 @KeepDim t :: Tensor 'D.Bool '[1, 2])
-- (Bool,([1,2],[[True,False]]))
all'
  :: forall dim keepOrDropDim device shape
   . (KnownNat dim, KnownKeepOrDropDim keepOrDropDim)
  => Tensor device 'D.Bool shape
  -> Tensor
       device
       'D.Bool
       (ConditionalDropDimension shape dim keepOrDropDim)
all' input = unsafePerformIO
  $ cast3 ATen.all_tlb input (natValI @dim) (keepOrDropDimVal @keepOrDropDim)

-- | any'
-- See https://pytorch.org/docs/stable/tensors.html#torch.BoolTensor.any.
-- >>> t = fromJust [[True, True], [True, False], [True, True], [True, True]] :: Tensor 'D.Bool '[4, 2]
--
-- >>> dtype &&& shape &&& (\t' -> D.asValue (toDynamic t') :: [Bool]) $ (any' @1 @DropDim t :: Tensor 'D.Bool '[4])
-- (Bool,([4],[True,True,True,True]))
--
-- >>> dtype &&& shape &&& (\t' -> D.asValue (toDynamic t') :: [[Bool]]) $ (any' @1 @KeepDim t :: Tensor 'D.Bool '[4, 1])
-- (Bool,([4,1],[[True],[True],[True],[True]]))
--
-- >>> dtype &&& shape &&& (\t' -> D.asValue (toDynamic t') :: [Bool]) $ (any' @0 @DropDim t :: Tensor 'D.Bool '[2])
-- (Bool,([2],[True,True]))
--
-- >>> dtype &&& shape &&& (\t' -> D.asValue (toDynamic t') :: [[Bool]]) $ (any' @0 @KeepDim t :: Tensor 'D.Bool '[1, 2])
-- (Bool,([1,2],[[True,True]]))
any'
  :: forall dim keepOrDropDim device shape
   . (KnownNat dim, KnownKeepOrDropDim keepOrDropDim)
  => Tensor device 'D.Bool shape
  -> Tensor device 'D.Bool (ConditionalDropDimension shape dim keepOrDropDim)
any' input = unsafePerformIO $ cast3 ATen.any_tlb input (natValI @dim) (keepOrDropDimVal @keepOrDropDim)

-- | dropout
-- >>> t = ones @D.Float @[3,2]
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
  :: forall device dtype shape
   . Double
  -> Bool
  -> Tensor device dtype shape
  -> IO (Tensor device dtype shape)
dropout p train input = cast3 ATen.dropout_tdb input p train

-- | featureDropout
-- >>> c = featureDropout (ones :: Tensor 'D.Float '[2,2]) 0.1 True
-- >>> dtype &&& shape $ c
-- (Float,[2,2])
featureDropout
  :: forall device dtype shape
   . Double
  -> Bool
  -> Tensor device dtype shape
  -> Tensor device dtype shape
featureDropout p train input =
  unsafePerformIO $ cast3 ATen.feature_dropout_tdb input p train

-- | alphaDropout
-- >>> c = alphaDropout (ones :: Tensor 'D.Float '[2,2]) 0.1 True
-- >>> dtype &&& shape $ c
-- (Float,[2,2])
alphaDropout
  :: forall device dtype shape
   . Double
  -> Bool
  -> Tensor device dtype shape
  -> Tensor device dtype shape
alphaDropout p train input =
  unsafePerformIO $ cast3 ATen.alpha_dropout_tdb input p train

-- | featureAlphaDropout
-- >>> c = featureAlphaDropout (ones :: Tensor 'D.Float '[2,2]) 0.1 True
-- >>> dtype &&& shape $ c
-- (Float,[2,2])
featureAlphaDropout
  :: forall device dtype shape
   . Double
  -> Bool
  -> Tensor device dtype shape
  -> Tensor device dtype shape
featureAlphaDropout p train input =
  unsafePerformIO $ cast3 ATen.feature_alpha_dropout_tdb input p train

-- | acos
-- >>> dtype &&& shape $ acos (ones :: Tensor 'D.Float '[3,2])
-- (Float,[3,2])
acos
  :: forall device dtype shape
   . Tensor device dtype shape
  -> Tensor device dtype shape
acos input = unsafePerformIO $ cast1 ATen.acos_t input

-- | avgPool1d
-- >>> t = avgPool1d @1 @1 @0 (ones::Tensor 'D.Float '[1,3,4])
-- >>> shape t
-- [1,3,4]
-- >>> :t t
-- t :: Tensor 'D.Float '[1, 3, 4]
avgPool1d
  :: forall
       kernelSize
       stride
       padding
       channelSize
       inputSize
       batchSize
       outputSize
       device
       dtype
   . ( All
         KnownNat
         '[kernelSize, stride, padding, channelSize, inputSize, batchSize]
     , ConvSideCheck inputSize kernelSize stride padding outputSize
     )
  => Tensor device dtype '[batchSize, channelSize, inputSize]
  -> Tensor device dtype '[batchSize, channelSize, outputSize]
avgPool1d input = unsafePerformIO $ (cast6 ATen.avg_pool1d_tlllbb)
  input
  (natValI @kernelSize)
  (natValI @stride)
  (natValI @padding)
  False
  True

-- | adaptiveAvgPool1d
-- >>> t = adaptiveAvgPool1d @8 (ones::Tensor 'D.Float '[1,3,16])
-- >>> shape t
-- [1,3,8]
-- >>> :t t
-- t :: Tensor 'D.Float '[1, 3, 8]
adaptiveAvgPool1d
  :: forall outputSize channelSize inputSize batchSize device dtype
   . (All KnownNat '[channelSize, inputSize, batchSize, outputSize])
  => Tensor device dtype '[batchSize, channelSize, inputSize]
  -> Tensor device dtype '[batchSize, channelSize, outputSize]
adaptiveAvgPool1d input = unsafePerformIO
  $ cast2 ATen.adaptive_avg_pool1d_tl input (natValI @outputSize)

-- | adaptiveMaxPool1d
-- >>> t = adaptiveMaxPool1d @8 (ones::Tensor 'D.Float '[1,3,16])
-- >>> shape t
-- [1,3,8]
-- >>> :t t
-- t :: Tensor 'D.Float '[1, 3, 8]
adaptiveMaxPool1d
  :: forall outputSize channelSize inputSize batchSize device dtype
   . (All KnownNat '[channelSize, inputSize, batchSize, outputSize])
  => Tensor device dtype '[batchSize, channelSize, inputSize]
  -> Tensor device dtype '[batchSize, channelSize, outputSize]
adaptiveMaxPool1d input =
  fst
    $ (unsafePerformIO
        $ (cast2 ATen.adaptive_max_pool1d_tl) input (natValI @outputSize) :: ( Tensor
            device
            dtype
            '[batchSize, channelSize, outputSize]
        , Tensor device 'D.Int64 '[batchSize, channelSize, outputSize]
        )
      )

-- | addmv
-- >>> t = addmv (ones :: Tensor 'D.Float '[]) (ones :: Tensor 'D.Float '[3,2]) (zeros :: Tensor 'D.Float '[2]) 1 1
-- >>> dtype &&& shape $ t
-- (Float,[3])
-- >>> :t t
-- t :: Tensor 'D.Float '[3]
addmv
  :: forall shape' shape n m device dtype
   . ( KnownNat n
     , KnownNat m
     , shape' ~ Broadcast shape '[n]
     )
  => Float
  -> Float
  -> Tensor device dtype '[n,m]
  -> Tensor device dtype '[m]
  -> Tensor device dtype shape
  -> Tensor device dtype shape'
addmv beta alpha mat vec input = unsafePerformIO $ cast5 ATen.addmv_tttss input mat vec beta alpha

-- | addr
-- >>> t = addr (ones :: Tensor 'D.Float '[]) (ones :: Tensor 'D.Float '[3]) (zeros :: Tensor 'D.Float '[2]) 1 1
-- >>> dtype &&& shape $ t
-- (Float,[3,2])
-- >>> :t t
-- t :: Tensor 'D.Float '[3, 2]
addr
  :: forall shape' shape n m device dtype
   . ( KnownNat n
     , KnownNat m
     , shape' ~ Broadcast shape '[n,m]
     )
  => Float
  -> Float
  -> Tensor device dtype '[n]
  -> Tensor device dtype '[m]
  -> Tensor device dtype shape
  -> Tensor device dtype shape'
addr beta alpha vec1 vec2 input = unsafePerformIO $ cast5 ATen.addr_tttss input vec1 vec2 beta alpha

-- affine_grid_generator :: Tensor dtype shape -> [Int] -> Tensor dtype shape
-- affine_grid_generator _theta _size = unsafePerformIO $ (cast2 ATen.affine_grid_generator_tl) _theta _size

-- allclose :: Tensor dtype shape -> Tensor dtype shape -> Double -> Double -> Bool -> Bool
-- allclose _input _other _rtol _atol _equal_nan = unsafePerformIO $ (cast5 ATen.allclose_ttddb) _input _other _rtol _atol _equal_nan

-- | argmax
-- See https://pytorch.org/docs/stable/torch.html#torch.argmax.
-- >>> t = fromJust [[0, 1], [-1, 2], [0, 1], [0, -2]] :: Tensor 'D.Float '[4, 2]
--
-- >>> dtype &&& shape &&& (\t' -> D.asValue (toDynamic t') :: [Int]) $ (argmax @1 @DropDim t :: Tensor 'D.Int64 '[4])
-- (Int64,([4],[1,1,1,0]))
--
-- >>> dtype &&& shape &&& (\t' -> D.asValue (toDynamic t') :: [[Int]]) $ (argmax @1 @KeepDim t :: Tensor 'D.Int64 '[4, 1])
-- (Int64,([4,1],[[1],[1],[1],[0]]))
--
-- >>> dtype &&& shape &&& (\t' -> D.asValue (toDynamic t') :: [Int]) $ (argmax @0 @DropDim t :: Tensor 'D.Int64 '[2])
-- (Int64,([2],[3,1]))
--
-- >>> dtype &&& shape &&& (\t' -> D.asValue (toDynamic t') :: [[Int]]) $ (argmax @0 @KeepDim t :: Tensor 'D.Int64 '[1, 2])
-- (Int64,([1,2],[[3,1]]))
argmax
  :: forall dim keepOrDropDim device dtype shape
   . (KnownNat dim, KnownKeepOrDropDim keepOrDropDim)
  => Tensor device dtype shape
  -> Tensor
       device
       'D.Int64
       (ConditionalDropDimension shape dim keepOrDropDim)
argmax input = unsafePerformIO $ cast3 ATen.argmax_tlb
                                       input
                                       (natValI @dim)
                                       (keepOrDropDimVal @keepOrDropDim)

-- | argmin
-- See https://pytorch.org/docs/stable/torch.html#torch.argmin.
-- >>> t = fromJust [[0, 1], [-1, 2], [0, 1], [0, -2]] :: Tensor 'D.Float '[4, 2]
--
-- >>> dtype &&& shape &&& (\t' -> D.asValue (toDynamic t') :: [Int]) $ (argmin @1 @DropDim t :: Tensor 'D.Int64 '[4])
-- (Int64,([4],[0,0,0,1]))
--
-- >>> dtype &&& shape &&& (\t' -> D.asValue (toDynamic t') :: [[Int]]) $ (argmin @1 @KeepDim t :: Tensor 'D.Int64 '[4, 1])
-- (Int64,([4,1],[[0],[0],[0],[1]]))
--
-- >>> dtype &&& shape &&& (\t' -> D.asValue (toDynamic t') :: [Int]) $ (argmin @0 @DropDim t :: Tensor 'D.Int64 '[2])
-- (Int64,([2],[1,3]))
--
-- >>> dtype &&& shape &&& (\t' -> D.asValue (toDynamic t') :: [[Int]]) $ (argmin @0 @KeepDim t :: Tensor 'D.Int64 '[1, 2])
-- (Int64,([1,2],[[1,3]]))
argmin
  :: forall dim keepOrDropDim device dtype shape
   . (KnownNat dim, KnownKeepOrDropDim keepOrDropDim)
  => Tensor device dtype shape
  -> Tensor
       device
       'D.Int64
       (ConditionalDropDimension shape dim keepOrDropDim)
argmin input = unsafePerformIO $ cast3 ATen.argmin_tlb
                                       input
                                       (natValI @dim)
                                       (keepOrDropDimVal @keepOrDropDim)

-- as_strided :: Tensor dtype shape -> [Int] -> [Int] -> Int -> Tensor dtype shape
-- as_strided _input _size _stride _storage_offset = unsafePerformIO $ (cast4 ATen.as_strided_tlll) _input _size _stride _storage_offset

-- | asin
-- >>> dtype &&& shape $ asin (ones :: Tensor 'D.Float '[3,2])
-- (Float,[3,2])
asin
  :: forall device dtype shape
   . Tensor device dtype shape
  -> Tensor device dtype shape
asin input = unsafePerformIO $ cast1 ATen.asin_t input

-- | atan
-- >>> dtype &&& shape $ atan (ones :: Tensor 'D.Float '[3,2])
-- (Float,[3,2])
atan
  :: forall device dtype shape
   . Tensor device dtype shape
  -> Tensor device dtype shape
atan input = unsafePerformIO $ cast1 ATen.atan_t input

-- | baddbmm
-- >>> t = baddbmm (ones :: Tensor 'D.Float '[]) (ones :: Tensor 'D.Float '[5,3,2]) (zeros :: Tensor 'D.Float '[5,2,4]) 1 1
-- >>> dtype &&& shape $ t
-- (Float,[5,3,4])
-- >>> :t t
-- t :: Tensor 'D.Float '[5, 3, 4]
baddbmm
  :: forall shape' shape  batchSize n m k device dtype
   . ( KnownNat n
     , KnownNat m
     , KnownNat k
     , shape' ~ Broadcast shape '[batchSize,n,m]
     )
  => Float
  -> Float
  -> Tensor device dtype '[batchSize,n,k]
  -> Tensor device dtype '[batchSize,k,m]
  -> Tensor device dtype shape
  -> Tensor device dtype shape'
baddbmm beta alpha batch1 batch2 input = unsafePerformIO $ cast5 ATen.baddbmm_tttss input batch1 batch2 beta alpha

-- batch_norm :: Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> Bool -> Double -> Double -> Bool -> Tensor dtype shape
-- batch_norm _input _weight _bias _running_mean _running_var _training _momentum _eps _cudnn_enabled = unsafePerformIO $ (cast9 ATen.batch_norm_tttttbddb) _input _weight _bias _running_mean _running_var _training _momentum _eps _cudnn_enabled

-- bilinear :: Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape
-- bilinear _input1 _input2 _weight _bias = unsafePerformIO $ (cast4 ATen.bilinear_tttt) _input1 _input2 _weight _bias

-- binary_cross_entropy_with_logits :: Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> Int -> Tensor dtype shape
-- binary_cross_entropy_with_logits _input _target _weight _pos_weight _reduction = unsafePerformIO $ (cast5 ATen.binary_cross_entropy_with_logits_ttttl) _input _target _weight _pos_weight _reduction

-- bincount :: Tensor dtype shape -> Tensor dtype shape -> Int -> Tensor dtype shape
-- bincount _input _weights _minlength = unsafePerformIO $ (cast3 ATen.bincount_ttl) _input _weights _minlength

-- bitwise_not :: Tensor dtype shape -> Tensor dtype shape
-- bitwise_not _input = unsafePerformIO $ (cast1 ATen.bitwise_not_t) _input

-- | batched matrix multiplication
-- >>> dtype &&& shape $ bmm (ones @'D.Float @'[5,3,2]) (zeros @'D.Float @'[5,2,4])
-- (Float,[5,3,4])
bmm
  :: forall batchSize n m k device dtype
   . Tensor device dtype '[batchSize, n, k]
  -> Tensor device dtype '[batchSize, k, m]
  -> Tensor device dtype '[batchSize, n, m]
bmm input mat2 = unsafePerformIO $ cast2 ATen.bmm_tt input mat2

-- | BroadcastTensorsImpl
-- >>> type Ty = BroadcastTensorsImpl '[] 'Nothing
-- >>> :kind! Ty
-- Ty :: Maybe (D.DType, [Nat])
-- = 'Nothing
-- >>> type Ty = BroadcastTensorsImpl '[Tensor 'D.Float '[1, 3], Tensor 'D.Float '[2, 1]] 'Nothing
-- >>> :kind! Ty
-- Ty :: Maybe (D.DType, [Nat])
-- = 'Just '( 'D.Float, '[2, 3])
-- >>> type Ty = BroadcastTensorsImpl '[Tensor 'D.Float '[1, 3], Tensor 'D.Float '[2, 1], Tensor 'D.Float '[5, 1, 1]] 'Nothing
-- >>> :kind! Ty
-- Ty :: Maybe (D.DType, [Nat])
-- = 'Just '( 'D.Float, '[5, 2, 3])
-- >>> type Ty = BroadcastTensorsImpl '[Tensor 'D.Float '[1, 3], Tensor 'D.Float '[2, 1], Tensor 'D.Float '[1, 5, 1]] 'Nothing
-- >>> :kind! Ty
-- Ty :: Maybe (D.DType, [Nat])
-- = 'Nothing
type family BroadcastTensorsImpl (tensors :: [a]) (acc :: Maybe ((DeviceType, Nat), D.DType, [Nat])) :: Maybe ((DeviceType, Nat), D.DType, [Nat]) where
  BroadcastTensorsImpl '[]                                    'Nothing                                = 'Nothing
  BroadcastTensorsImpl '[]                                    ('Just '(device, dtype, reverseShape))  = 'Just '(device, dtype, Reverse reverseShape)
  BroadcastTensorsImpl (Tensor device dtype shape ': tensors) 'Nothing                                = BroadcastTensorsImpl tensors ('Just '(device, dtype, Reverse shape))
  BroadcastTensorsImpl (Tensor device dtype shape ': tensors) ('Just '(device, dtype, reverseShape')) = BroadcastTensorsImpl tensors (MaybeTriple ('Just device) ('Just dtype) (ComputeBroadcast (Reverse shape) reverseShape'))
  BroadcastTensorsImpl (Tensor device dtype shape ': _)       ('Just _)                               = Nothing

type family BroadcastTensorsCheck (tensors :: [a]) (result :: Maybe ((DeviceType, Nat), D.DType, [Nat])) :: [a] where
  BroadcastTensorsCheck tensors 'Nothing                        = TypeError (    Text "Cannot broadcast tensors due to incompatible shapes and/or dtypes: "
                                                                            :<>: ShowType tensors
                                                                            )
  BroadcastTensorsCheck tensors ('Just '(device, dtype, shape)) = HReplicateR (ListLength tensors) (Tensor device dtype shape)

type BroadcastTensors tensors
  = BroadcastTensorsCheck tensors (BroadcastTensorsImpl tensors 'Nothing)

-- | broadcast tensors
-- TODO: broadcastTensors returns garbage data and is hence broken
-- See https://pytorch.org/docs/stable/_modules/torch/functional.html#broadcast_tensors.
-- >>> x = ones @'D.Float @'[1, 3]
-- >>> y = ones @'D.Float @'[2, 1]
-- >>> z = ones @'D.Float @'[5, 1, 1]
-- 
-- -- >>> x' :. y' :. z' :. HNil = broadcastTensors (x :. y :. z :. HNil)
-- -- >>> :type x'
-- -- x' :: Tensor 'D.Float '[5, 2, 3]
-- -- >>> dtype &&& shape &&& (\t -> D.asValue (toDynamic t) :: [[[Float]]]) $ x'
-- -- >>> :type y'
-- -- y' :: Tensor 'D.Float '[5, 2, 3]
-- -- >>> dtype &&& shape &&& (\t -> D.asValue (toDynamic t) :: [[[Float]]]) $ y'
-- -- >>> :type z'
-- -- z' :: Tensor 'D.Float '[5, 2, 3]
-- -- >>> dtype &&& shape &&& (\t -> D.asValue (toDynamic t) :: [[[Float]]]) $ z'
broadcastTensors
  :: forall tensors tensors'
   . ( tensors' ~ BroadcastTensors tensors
     , HFoldrM IO TensorListFolds [D.ATenTensor] tensors
     , Apply TensorListFolds [D.ATenTensor] (HUnfoldMRes IO [D.ATenTensor] tensors)
     , HUnfoldM IO TensorListFolds (HUnfoldMRes IO [D.ATenTensor] tensors) tensors
     , HFoldrM IO TensorListFolds [D.ATenTensor] tensors'
     , Apply TensorListFolds [D.ATenTensor] (HUnfoldMRes IO [D.ATenTensor] tensors')
     , HUnfoldM IO TensorListFolds (HUnfoldMRes IO [D.ATenTensor] tensors') tensors'
     )
  => HList tensors
  -> HList tensors'
broadcastTensors tensors = unsafePerformIO $ cast1 ATen.broadcast_tensors_l tensors

type family CatImpl (dim :: Nat) (tensors :: [a]) (acc :: Maybe ((DeviceType, Nat), D.DType, [Nat])) :: Maybe ((DeviceType, Nat), D.DType, [Nat]) where
  CatImpl _   '[]                                    acc = acc
  CatImpl dim (Tensor device dtype shape ': tensors) acc = CatImpl dim tensors (MaybeTriple (ComputeCatDevice device acc) (ComputeCatDType dtype acc) (ComputeCatShape dim shape acc))

type family ComputeCatShape (dim :: Nat) (shape :: [Nat]) (acc :: Maybe ((DeviceType, Nat), D.DType, [Nat])) :: Maybe [Nat] where
  ComputeCatShape 0   (x ': xs) Nothing                          = Just (x ': xs)
  ComputeCatShape dim (x ': xs) Nothing                          = AppendToMaybe x (ComputeCatShape (dim - 1) xs Nothing)
  ComputeCatShape 0   (x ': xs) (Just '(_, _, y ': xs))          = Just ((x + y) ': xs)
  ComputeCatShape dim (x ': xs) (Just '(device, dtype, x ': ys)) = AppendToMaybe x (ComputeCatShape (dim - 1) xs (Just '(device, dtype, ys)))
  ComputeCatShape _   _         _                                = Nothing

type family ComputeCatDType (dtype :: D.DType) (acc :: Maybe ((DeviceType, Nat), D.DType, [Nat])) :: Maybe D.DType where
  ComputeCatDType dtype Nothing               = Just dtype
  ComputeCatDType dtype (Just '(_, dtype, _)) = Just dtype
  ComputeCatDType _     _                     = Nothing

type family ComputeCatDevice (device :: (DeviceType, Nat)) (acc :: Maybe ((DeviceType, Nat), D.DType, [Nat])) :: Maybe (DeviceType, Nat) where
  ComputeCatDevice device Nothing                = Just device
  ComputeCatDevice device (Just '(device, _, _)) = Just device
  ComputeCatDevice _      _                      = Nothing

type family CatCheck (res :: Maybe ((DeviceType, Nat), D.DType, [Nat])) :: ((DeviceType, Nat), D.DType, [Nat]) where
  CatCheck 'Nothing                        = TypeError (Text "Concatenation impossible.")
  CatCheck ('Just '(device, dtype, shape)) = '(device, dtype, shape)

-- | Cat
-- >>> type Ty = Cat 0 '[Tensor 'D.Float '[1]]
-- >>> :kind! Ty
-- Ty :: (D.DType, [Nat])
-- = '( 'D.Float, '[1])
-- >>> type Ty = Cat 0 '[Tensor 'D.Float '[1], Tensor 'D.Float '[2]]
-- >>> :kind! Ty
-- Ty :: (D.DType, [Nat])
-- = '( 'D.Float, '[3])
-- >>> type Ty = Cat 0 '[Tensor 'D.Float '[1, 3], Tensor 'D.Float '[2, 3]]
-- >>> :kind! Ty
-- Ty :: (D.DType, [Nat])
-- = '( 'D.Float, '[3, 3])
-- >>> type Ty = Cat 1 '[Tensor 'D.Float '[3, 1], Tensor 'D.Float '[3, 2]]
-- >>> :kind! Ty
-- Ty :: (D.DType, [Nat])
-- = '( 'D.Float, '[3, 3])
-- >>> type Ty = Cat 1 '[Tensor 'D.Float '[2, 5, 4, 2], Tensor 'D.Float '[2, 1, 4, 2], Tensor 'D.Float '[2, 3, 4, 2], Tensor 'D.Float '[2, 1, 4, 2]]
-- >>> :kind! Ty
-- Ty :: (D.DType, [Nat])
-- = '( 'D.Float, '[2, 10, 4, 2])
type Cat dim tensors = CatCheck (CatImpl dim tensors Nothing)

-- | cat
-- >>> t = ones @'D.Float @'[2,2]
-- >>> t' = cat @0 (t :. HNil)
-- >>> :type t'
-- t' :: Tensor 'D.Float '[2, 2]
-- >>> dtype &&& shape &&& (\t'' -> D.asValue (toDynamic t'') :: [[Float]]) $ t'
-- (Float,([2,2],[[1.0,1.0],[1.0,1.0]]))
-- >>> t' = cat @1 (t :. HNil)
-- >>> :type t'
-- t' :: Tensor 'D.Float '[2, 2]
-- >>> dtype &&& shape &&& (\t'' -> D.asValue (toDynamic t'') :: [[Float]]) $ t'
-- (Float,([2,2],[[1.0,1.0],[1.0,1.0]]))
-- >>> t' = cat @0 (t :. t :. t :. HNil)
-- >>> :type t'
-- t' :: Tensor 'D.Float '[6, 2]
-- >>> dtype &&& shape &&& (\t'' -> D.asValue (toDynamic t'') :: [[Float]]) $ t'
-- (Float,([6,2],[[1.0,1.0],[1.0,1.0],[1.0,1.0],[1.0,1.0],[1.0,1.0],[1.0,1.0]]))
-- >>> t' = cat @1 (t :. t :. t :. HNil)
-- >>> :type t'
-- t' :: Tensor 'D.Float '[2, 6]
-- >>> dtype &&& shape &&& (\t'' -> D.asValue (toDynamic t'') :: [[Float]]) $ t'
-- (Float,([2,6],[[1.0,1.0,1.0,1.0,1.0,1.0],[1.0,1.0,1.0,1.0,1.0,1.0]]))
cat
  :: forall dim device dtype shape tensors
   . ( KnownNat dim
     , '(device, dtype, shape) ~ Cat dim tensors
     , HFoldrM IO TensorListFolds [D.ATenTensor] tensors
     , Apply TensorListFolds [D.ATenTensor] (HUnfoldMRes IO [D.ATenTensor] tensors)
     , HUnfoldM IO TensorListFolds (HUnfoldMRes IO [D.ATenTensor] tensors) tensors
     )
  => HList tensors
  -> Tensor device dtype shape
cat tensors = unsafePerformIO $ cast2 ATen.cat_ll tensors (natValI @dim :: Int)

-- chain_matmul :: [Tensor dtype shape] -> Tensor dtype shape
-- chain_matmul _matrices = unsafePerformIO $ (cast1 ATen.chain_matmul_l) _matrices

type family ChunkImpl (chunkShapes :: Maybe [[Nat]]) (device :: (DeviceType, Nat)) (dtype :: D.DType) :: Maybe a where
  ChunkImpl (Just '[])               _      _     = Just '[]
  ChunkImpl (Just (shape ': shapes)) device dtype = AppendToMaybe (Tensor device dtype shape) (ChunkImpl (Just shapes) device dtype)
  ChunkImpl Nothing                  _      _     = Nothing

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

type Chunk chunks dim device dtype shape = ChunkCheck shape dim (ChunkImpl (ChunkShapes chunks dim shape) device dtype)

-- | chunk
-- >>> :type chunk @3 @1 (ones @D.Float @[2, 2])
-- chunk @3 @1 (ones @D.Float @[2, 2])
--   :: HList '[Tensor 'D.Float '[2, 1], Tensor 'D.Float '[2, 1]]
-- >>> t0 :. t1 :. HNil = chunk @3 @1 (ones @D.Float @[2, 2])
-- >>> dtype &&& shape $ t0
-- (Float,[2,1])
-- >>> dtype &&& shape $ t1
-- (Float,[2,1])
-- >>> :type chunk @3 @1 (ones @D.Float @'[1, 0, 3])
-- chunk @3 @1 (ones @D.Float @'[1, 0, 3])
--   :: HList
--        '[Tensor 'D.Float '[1, 0, 3], Tensor 'D.Float '[1, 0, 3],
--          Tensor 'D.Float '[1, 0, 3]]
-- >>> t0 :. t1 :. t2 :. HNil = chunk @3 @1 (ones @D.Float @'[1, 0, 3])
-- >>> dtype &&& shape $ t0
-- (Float,[1,0,3])
-- >>> dtype &&& shape $ t1
-- (Float,[1,0,3])
-- >>> dtype &&& shape $ t2
-- (Float,[1,0,3])
-- >>> :type chunk @6 @0 (ones @D.Float @[19, 4])
-- chunk @6 @0 (ones @D.Float @[19, 4])
--   :: HList
--        '[Tensor 'D.Float '[4, 4], Tensor 'D.Float '[4, 4],
--          Tensor 'D.Float '[4, 4], Tensor 'D.Float '[4, 4],
--          Tensor 'D.Float '[3, 4]]
-- >>> t0 :. t1 :. t2 :. t3 :. t4 :. HNil = chunk @6 @0 (ones @D.Float @[19, 4])
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
  :: forall chunks dim device dtype shape tensorChunks
   . ( KnownNat chunks
     , KnownNat dim
     , tensorChunks ~ Chunk chunks dim device dtype shape
     , HFoldrM IO TensorListFolds [D.ATenTensor] tensorChunks
     , Apply TensorListFolds [D.ATenTensor] (HUnfoldMRes IO [D.ATenTensor] tensorChunks)
     , HUnfoldM IO TensorListFolds (HUnfoldMRes IO [D.ATenTensor] tensorChunks) tensorChunks
     )
  => Tensor device dtype shape
  -> HList tensorChunks
chunk input = unsafePerformIO
  $ cast3 ATen.chunk_tll input (natValI @chunks :: Int) (natValI @dim :: Int)

-- | clamp
-- >>> dtype &&& shape $ clamp (ones :: Tensor 'D.Float '[3,2]) 0 1
-- (Float,[3,2])
clamp
  :: forall device dtype shape
   . Float
  -> Float
  -> Tensor device dtype shape
  -> Tensor device dtype shape
clamp min max input = unsafePerformIO $ cast3 ATen.clamp_tss input min max

-- | clampMax
-- >>> dtype &&& shape $ clamp_max (ones :: Tensor 'D.Float '[3,2]) 1
-- (Float,[3,2])
clampMax
  :: forall device dtype shape
   . Float
  -> Tensor device dtype shape
  -> Tensor device dtype shape
clampMax max input = unsafePerformIO $ cast2 ATen.clamp_max_ts input max

-- | clampMin
-- >>> dtype &&& shape $ clamp_min (ones :: Tensor 'D.Float '[3,2]) 0
-- (Float,[3,2])
clampMin
  :: forall device dtype shape
   . Float
  -> Tensor device dtype shape
  -> Tensor device dtype shape
clampMin min input = unsafePerformIO $ cast2 ATen.clamp_min_ts input min

-- | cudnnIsAcceptable
cudnnIsAcceptable
  :: forall device dtype shape . Tensor device dtype shape -> Bool
cudnnIsAcceptable input =
  unsafePerformIO $ cast1 ATen.cudnn_is_acceptable_t input

-- constant_pad_nd :: Tensor dtype shape -> [Int] -> Float -> Tensor dtype shape
-- constant_pad_nd _input _pad _value = unsafePerformIO $ (cast3 ATen.constant_pad_nd_tls) _input _pad _value

-- convolution :: Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> [Int] -> [Int] -> [Int] -> Bool -> [Int] -> Int -> Tensor dtype shape
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
-- >>> :kind! ConvOutputSize 1 0 1 4
-- ConvOutputSize 1 0 1 4 :: Nat
-- = 4
type family ConvOutputSize (stride :: Nat) (padding :: Nat) (kernel_size :: Nat)  (input_size :: Nat) :: Nat where
    ConvOutputSize s p k i = (Div (i + 2 * p - k) s) + 1

-- | conv1d
-- >>> t = conv1d @1 @0 (ones @'D.Float @'[10, 3, 1]) (ones @'D.Float @'[10]) (ones @'D.Float @'[1, 3, 4])
-- >>> :type t
-- t :: Tensor 'D.Float '[1, 10, 4]
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
            device
            dtype
   . ( All KnownNat [ stride
                    , padding
                    , inputChannelSize, outputChannelSize
                    , kernelSize
                    , inputSize
                    , batchSize
                    , outputSize
                    ]
     , ConvSideCheck inputSize kernelSize stride padding outputSize
     )
  => Tensor device dtype '[outputChannelSize, inputChannelSize, kernelSize]
  -> Tensor device dtype '[outputChannelSize]
  -> Tensor device dtype '[batchSize, inputChannelSize, inputSize]
  -> Tensor device dtype '[batchSize, outputChannelSize, outputSize]
conv1d weight bias input = unsafePerformIO $ cast7 ATen.conv1d_tttllll
                                                   input
                                                   weight
                                                   bias
                                                   (natValI @stride :: Int)
                                                   (natValI @padding :: Int)
                                                   (1 :: Int)
                                                   (1 :: Int)

-- | conv2d
-- >>> t = conv2d @'(1, 1) @'(0, 0) (ones @'D.Float @'[10, 3, 1, 1]) (ones @'D.Float @'[10]) (ones @'D.Float @'[1, 3, 4, 5])
-- >>> :type t
-- t :: Tensor 'D.Float '[1, 10, 4, 5]
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
            device
            dtype
   . ( All KnownNat [ Fst stride, Snd stride
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
  => Tensor device dtype '[outputChannelSize, inputChannelSize, kernelSize0, kernelSize1]
  -> Tensor device dtype '[outputChannelSize]
  -> Tensor device dtype '[batchSize, inputChannelSize, inputSize0, inputSize1]
  -> Tensor device dtype '[batchSize, outputChannelSize, outputSize0, outputSize1]
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
-- >>> t = conv3d @'(1, 1, 1) @'(0, 0, 0) (ones @'D.Float @'[10, 3, 1, 1, 1]) (ones @'D.Float @'[10]) (ones @'D.Float @'[1, 3, 4, 5, 6])
-- >>> :type t
-- t :: Tensor 'D.Float '[1, 10, 4, 5, 6]
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
            device
            dtype
   . ( All KnownNat [ Fst3 stride, Snd3 stride, Trd3 stride
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
  => Tensor device dtype '[outputChannelSize, inputChannelSize, kernelSize0, kernelSize1, kernelSize2]
  -> Tensor device dtype '[outputChannelSize]
  -> Tensor device dtype '[batchSize, inputChannelSize, inputSize0, inputSize1, inputSize2]
  -> Tensor device dtype '[batchSize, outputChannelSize, outputSize0, outputSize1, outputSize2]
conv3d weight bias input = unsafePerformIO $ cast7
  ATen.conv3d_tttllll
  input
  weight
  bias
  ([natValI @(Fst3 stride), natValI @(Snd3 stride), natValI @(Trd3 stride)] :: [Int])
  ([natValI @(Fst3 padding), natValI @(Snd3 padding), natValI @(Trd3 padding)] :: [Int])
  ([1, 1, 1] :: [Int])
  (1 :: Int)

-- conv_tbc :: Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> Int -> Tensor dtype shape
-- conv_tbc _input _weight _bias _pad = unsafePerformIO $ (cast4 ATen.conv_tbc_tttl) _input _weight _bias _pad

-- conv_transpose1d :: Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> Int -> Int -> Int -> Int -> Int -> Tensor dtype shape
-- conv_transpose1d _input _weight _bias _stride _padding _output_padding _groups _dilation = unsafePerformIO $ (cast8 ATen.conv_transpose1d_tttlllll) _input _weight _bias _stride _padding _output_padding _groups _dilation

-- | cosh
-- >>> dtype &&& shape $ cosh (ones :: Tensor 'D.Float '[3,2])
-- (Float,[3,2])
cosh :: forall device dtype shape . Tensor device dtype shape -> Tensor device dtype shape
cosh input = unsafePerformIO $ cast1 ATen.cosh_t input

-- cosine_embedding_loss :: Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> Double -> Int -> Tensor dtype shape
-- cosine_embedding_loss _input1 _input2 _target _margin _reduction = unsafePerformIO $ (cast5 ATen.cosine_embedding_loss_tttdl) _input1 _input2 _target _margin _reduction

-- cudnn_affine_grid_generator :: Tensor dtype shape -> Int -> Int -> Int -> Int -> Tensor dtype shape
-- cudnn_affine_grid_generator _theta _N _C _H _W = unsafePerformIO $ (cast5 ATen.cudnn_affine_grid_generator_tllll) _theta _N _C _H _W

-- cudnn_batch_norm :: Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> Bool -> Double -> Double -> (Tensor dtype shape,Tensor dtype shape,Tensor dtype shape)
-- cudnn_batch_norm _input _weight _bias _running_mean _running_var _training _exponential_average_factor _epsilon = unsafePerformIO $ (cast8 ATen.cudnn_batch_norm_tttttbdd) _input _weight _bias _running_mean _running_var _training _exponential_average_factor _epsilon

-- cudnn_convolution :: Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> [Int] -> [Int] -> [Int] -> Int -> Bool -> Bool -> Tensor dtype shape
-- cudnn_convolution _input _weight _bias _padding _stride _dilation _groups _benchmark _deterministic = unsafePerformIO $ (cast9 ATen.cudnn_convolution_tttllllbb) _input _weight _bias _padding _stride _dilation _groups _benchmark _deterministic

-- cudnn_convolution_transpose :: Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> [Int] -> [Int] -> [Int] -> [Int] -> Int -> Bool -> Bool -> Tensor dtype shape
-- cudnn_convolution_transpose _input _weight _bias _padding _output_padding _stride _dilation _groups _benchmark _deterministic = unsafePerformIO $ (cast10 ATen.cudnn_convolution_transpose_tttlllllbb) _input _weight _bias _padding _output_padding _stride _dilation _groups _benchmark _deterministic

-- cudnn_grid_sampler :: Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape
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
-- >>> dtype &&& shape $ det (ones :: Tensor 'D.Float '[2,2])
-- (Float,[])
-- >>> dtype &&& shape $ det (ones :: Tensor 'D.Float '[3,2,2])
-- (Float,[3])
det
  :: forall device dtype shape
   . Tensor device dtype shape
  -> Tensor device dtype (Det shape)
det input = unsafePerformIO $ cast1 ATen.det_t input

-- diag_embed :: Tensor dtype shape -> Int -> Int -> Int -> Tensor dtype shape
-- diag_embed _input _offset _dim1 _dim2 = unsafePerformIO $ (cast4 ATen.diag_embed_tlll) _input _offset _dim1 _dim2

-- diagflat :: Tensor dtype shape -> Int -> Tensor dtype shape
-- diagflat _input _offset = unsafePerformIO $ (cast2 ATen.diagflat_tl) _input _offset

-- diagonal :: Tensor dtype shape -> Int -> Int -> Int -> Tensor dtype shape
-- diagonal _input _offset _dim1 _dim2 = unsafePerformIO $ (cast4 ATen.diagonal_tlll) _input _offset _dim1 _dim2

-- dot :: Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape
-- dot _input _tensor = unsafePerformIO $ (cast2 ATen.dot_tt) _input _tensor

-- einsum :: String -> [Tensor dtype shape] -> Tensor dtype shape
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
-- >>> weights = fromJust [[1, 1], [2, 2], [3, 3], [4, 4]] :: Tensor 'D.Float '[4, 2]
-- >>> indices = fromJust [[0], [2], [0], [1]] :: Tensor 'D.Int64 '[4, 1]
-- >>> t = embedding @('Just 0) False False weights indices
-- >>> :type t
-- t :: Tensor 'D.Float '[4, 1, 2]
-- >>> dtype &&& shape &&& (\t' -> D.asValue (toDynamic t') :: [[[Float]]]) $ t
-- (Float,([4,1,2],[[[1.0,1.0]],[[3.0,3.0]],[[1.0,1.0]],[[2.0,2.0]]]))
-- >>> t = embedding @'Nothing False False weights indices
-- >>> :type t
-- t :: Tensor 'D.Float '[4, 1, 2]
-- >>> dtype &&& shape &&& (\t' -> D.asValue (toDynamic t') :: [[[Float]]]) $ t
-- (Float,([4,1,2],[[[1.0,1.0]],[[3.0,3.0]],[[1.0,1.0]],[[2.0,2.0]]]))
embedding
  :: forall (paddingIdx :: Maybe Nat) dtype shape numEmbeds embedDim
   . ( KnownMaybeNat paddingIdx
     , PaddingIdxCheck paddingIdx numEmbeds
     )
  => Bool
  -> Bool
  -> Tensor dtype '[numEmbeds, embedDim]
  -> Tensor 'D.Int64 shape
  -> Tensor dtype (Reverse (embedDim ': Reverse shape))
embedding scaleGradByFreq sparse weights indices =
  unsafePerformIO $ cast5 ATen.embedding_ttlbb weights indices paddingIdx scaleGradByFreq sparse
 where paddingIdx :: Int
       paddingIdx = case maybeNatVal @paddingIdx of
                      Just idx -> fromIntegral idx
                      Nothing  -> -1

-- embedding_bag :: Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> Bool -> Int -> Bool -> Tensor dtype shape -> (Tensor dtype shape,Tensor dtype shape,Tensor dtype shape,Tensor dtype shape)
-- embedding_bag _weight _indices _offsets _scale_grad_by_freq _mode _sparse _per_sample_weights = unsafePerformIO $ (cast7 ATen.embedding_bag_tttblbt) _weight _indices _offsets _scale_grad_by_freq _mode _sparse _per_sample_weights

-- | emptyLike
-- >>> t <- emptyLike (ones :: Tensor 'D.Float '[3,4,5])
-- >>> dtype &&& shape $ t
-- (Float,[3,4,5])
emptyLike :: Tensor dtype shape -> IO (Tensor dtype shape)
emptyLike _input = (cast1 ATen.empty_like_t) _input

-- | erfc
-- >>> dtype &&& shape $ erfc (ones :: Tensor 'D.Float '[3,2])
-- (Float,[3,2])
erfc :: Tensor dtype shape -> Tensor dtype shape
erfc _input = unsafePerformIO $ (cast1 ATen.erfc_t) _input

-- | expm1
-- >>> dtype &&& shape $ expm1 (ones :: Tensor 'D.Float '[3,2])
-- (Float,[3,2])
expm1 :: Tensor dtype shape -> Tensor dtype shape
expm1 _input = unsafePerformIO $ (cast1 ATen.expm1_t) _input

-- | expand
-- TODO: figure out what the boolean value does
-- >>> t = ones :: Tensor 'D.Float '[2]
-- >>> t' = expand @[3, 1, 2] False t
-- >>> dtype &&& shape $ t'
-- (Float,[3,1,2])
-- >>> t'' = expand @[3, 1, 2] True t
-- >>> dtype &&& shape $ t''
-- (Float,[3,1,2])
-- >>> toInt (all (t' ==. t'')) == 1
-- True
expand
  :: forall shape' dtype shape
   . ( KnownShape shape'
     , shape' ~ Broadcast shape shape'
     )
  => Bool
  -> Tensor dtype shape
  -> Tensor dtype shape'
expand someBool input = unsafePerformIO $ cast3 ATen.tensor_expand_lb input (shapeVal @shape') someBool

-- flatten :: Tensor dtype shape -> Int -> Int -> Tensor dtype shape
-- flatten _input _start_dim _end_dim = unsafePerformIO $ (cast3 ATen.flatten_tll) _input _start_dim _end_dim

-- | flattenAll
-- >>> t = flattenAll (ones :: Tensor 'D.Float '[3,2])
-- >>> dtype &&& shape $ t
-- (Float,[6])
-- >>> :t t
-- t :: Tensor 'D.Float '[6]
flattenAll :: KnownShape shape => Tensor dtype shape -> Tensor dtype '[Product shape]
flattenAll _input = unsafePerformIO $ (cast3 ATen.flatten_tll) _input (0::Int) (-1::Int)

-- | frac
-- >>> dtype &&& shape $ frac (ones :: Tensor 'D.Float '[3,2])
-- (Float,[3,2])
frac :: Tensor dtype shape -> Tensor dtype shape
frac _input = unsafePerformIO $ (cast1 ATen.frac_t) _input

-- | full_like
-- >>> dtype &&& shape $ full_like (ones :: Tensor 'D.Float '[3,2]) 3.0
-- (Float,[3,2])
full_like :: Tensor dtype shape -> Float -> Tensor dtype shape
full_like _input _fill_value = unsafePerformIO $ (cast2 ATen.full_like_ts) _input _fill_value

-- grid_sampler :: Tensor dtype shape -> Tensor dtype shape -> Int -> Int -> Tensor dtype shape
-- grid_sampler _input _grid _interpolation_mode _padding_mode = unsafePerformIO $ (cast4 ATen.grid_sampler_ttll) _input _grid _interpolation_mode _padding_mode

-- grid_sampler_2d :: Tensor dtype shape -> Tensor dtype shape -> Int -> Int -> Tensor dtype shape
-- grid_sampler_2d _input _grid _interpolation_mode _padding_mode = unsafePerformIO $ (cast4 ATen.grid_sampler_2d_ttll) _input _grid _interpolation_mode _padding_mode

-- grid_sampler_3d :: Tensor dtype shape -> Tensor dtype shape -> Int -> Int -> Tensor dtype shape
-- grid_sampler_3d _input _grid _interpolation_mode _padding_mode = unsafePerformIO $ (cast4 ATen.grid_sampler_3d_ttll) _input _grid _interpolation_mode _padding_mode

-- hinge_embedding_loss :: Tensor dtype shape -> Tensor dtype shape -> Double -> Int -> Tensor dtype shape
-- hinge_embedding_loss _input _target _margin _reduction = unsafePerformIO $ (cast4 ATen.hinge_embedding_loss_ttdl) _input _target _margin _reduction

-- ger :: Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape
-- ger _input _vec2 = unsafePerformIO $ (cast2 ATen.ger_tt) _input _vec2

-- group_norm :: Tensor dtype shape -> Int -> Tensor dtype shape -> Tensor dtype shape -> Double -> Bool -> Tensor dtype shape
-- group_norm _input _num_groups _weight _bias _eps _cudnn_enabled = unsafePerformIO $ (cast6 ATen.group_norm_tlttdb) _input _num_groups _weight _bias _eps _cudnn_enabled

-- fft :: Tensor dtype shape -> Int -> Bool -> Tensor dtype shape
-- fft _input _signal_ndim _normalized = unsafePerformIO $ (cast3 ATen.fft_tlb) _input _signal_ndim _normalized

-- ifft :: Tensor dtype shape -> Int -> Bool -> Tensor dtype shape
-- ifft _input _signal_ndim _normalized = unsafePerformIO $ (cast3 ATen.ifft_tlb) _input _signal_ndim _normalized

-- rfft :: Tensor dtype shape -> Int -> Bool -> Bool -> Tensor dtype shape
-- rfft _input _signal_ndim _normalized _onesided = unsafePerformIO $ (cast4 ATen.rfft_tlbb) _input _signal_ndim _normalized _onesided

-- irfft :: Tensor dtype shape -> Int -> Bool -> Bool -> [Int] -> Tensor dtype shape
-- irfft _input _signal_ndim _normalized _onesided _signal_sizes = unsafePerformIO $ (cast5 ATen.irfft_tlbbl) _input _signal_ndim _normalized _onesided _signal_sizes

-- index :: Tensor dtype shape -> [Tensor dtype shape] -> Tensor dtype shape
-- index _input _indices = unsafePerformIO $ (cast2 ATen.index_tl) _input _indices

-- index_copy :: Tensor dtype shape -> Int -> Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape
-- index_copy _input _dim _index _source = unsafePerformIO $ (cast4 ATen.index_copy_tltt) _input _dim _index _source

-- index_put :: Tensor dtype shape -> [Tensor dtype shape] -> Tensor dtype shape -> Bool -> Tensor dtype shape
-- index_put _input _indices _values _accumulate = unsafePerformIO $ (cast4 ATen.index_put_tltb) _input _indices _values _accumulate

-- instance_norm :: Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> Bool -> Double -> Double -> Bool -> Tensor dtype shape
-- instance_norm _input _weight _bias _running_mean _running_var _use_input_stats _momentum _eps _cudnn_enabled = unsafePerformIO $ (cast9 ATen.instance_norm_tttttbddb) _input _weight _bias _running_mean _running_var _use_input_stats _momentum _eps _cudnn_enabled

-- | isclose
-- >>> dtype &&& shape $ isclose (ones :: Tensor 'D.Float '[3,2]) (ones :: Tensor 'D.Float '[3,2]) 0.1 0.1 False
-- (Bool,[3,2])
isclose :: Tensor dtype shape -> Tensor dtype shape -> Double -> Double -> Bool -> Tensor D.Bool shape
isclose _input _other _rtol _atol _equal_nan = unsafePerformIO $ (cast5 ATen.isclose_ttddb) _input _other _rtol _atol _equal_nan

-- | isnan
-- >>> dtype &&& shape $ isnan (ones :: Tensor 'D.Float '[3,2])
-- (Bool,[3,2])
isnan :: Tensor dtype shape -> Tensor 'D.Bool shape
isnan _input = unsafePerformIO $ (cast1 ATen.isnan_t) _input

is_distributed :: Tensor dtype shape -> Bool
is_distributed _input = unsafePerformIO $ (cast1 ATen.is_distributed_t) _input

is_floating_point :: Tensor dtype shape -> Bool
is_floating_point _input = unsafePerformIO $ (cast1 ATen.is_floating_point_t) _input

is_complex :: Tensor dtype shape -> Bool
is_complex _input = unsafePerformIO $ (cast1 ATen.is_complex_t) _input

is_nonzero :: Tensor dtype shape -> Bool
is_nonzero _input = unsafePerformIO $ (cast1 ATen.is_nonzero_t) _input

is_same_size :: Tensor dtype shape -> Tensor dtype shape2 -> Bool
is_same_size _input _other = unsafePerformIO $ (cast2 ATen.is_same_size_tt) _input _other

is_signed :: Tensor dtype shape -> Bool
is_signed _input = unsafePerformIO $ (cast1 ATen.is_signed_t) _input

-- kl_div :: Tensor dtype shape -> Tensor dtype shape -> Int -> Tensor dtype shape
-- kl_div _input _target _reduction = unsafePerformIO $ (cast3 ATen.kl_div_ttl) _input _target _reduction

-- kthvalue :: Tensor dtype shape -> Int -> Int -> Bool -> (Tensor dtype shape,Tensor dtype shape)
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
-- >>> t = layerNorm @'[1, 2] @'D.Float @'[2, 1, 2] ones ones 0.01 ones
-- >>> :type t
-- t :: Tensor 'D.Float '[2, 1, 2]
-- >>> dtype &&& shape $ t
-- (Float,[2,1,2])
layerNorm
  :: forall normalizedShape dtype shape
   . ( KnownShape normalizedShape
     , EndsWith shape normalizedShape
     )
  => Tensor dtype normalizedShape -- ^ weight
  -> Tensor dtype normalizedShape -- ^ bias
  -> Double -- ^ eps
  -> Tensor dtype shape -- ^ input tensor
  -> Tensor dtype shape -- ^ output tensor
layerNorm weight bias eps input = unsafePerformIO $ cast6
  ATen.layer_norm_tlttdb
  input
  (shapeVal @normalizedShape)
  weight
  bias
  eps
  (  cudnn_is_acceptable weight
  && cudnn_is_acceptable bias
  && cudnn_is_acceptable input
  )

-- native_layer_norm :: Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> Int -> Int -> Double -> (Tensor dtype shape,Tensor dtype shape,Tensor dtype shape)
-- native_layer_norm _input _weight _bias _M _N _eps = unsafePerformIO $ (cast6 ATen.native_layer_norm_tttlld) _input _weight _bias _M _N _eps

-- | linear
-- https://pytorch.org/docs/stable/_modules/torch/nn/functional.html#linear
-- >>> w = fromJust [[-0.5, -2,  0.5], [1.5, -0.5, 0.5]] :: Tensor 'D.Float '[2, 3]
-- >>> b = fromJust [0, 0.5] :: Tensor 'D.Float '[2]
-- >>> t = fromJust [[-2, 0.5, 1], [0.5, 0, 0], [0, 1, 0], [0, 0, 0], [1, -1, 0]] :: Tensor 'D.Float '[5, 3]
-- >>> t' = linear w b t
-- >>> :type t'
-- t' :: Tensor 'D.Float '[5, 2]
-- >>> dtype &&& shape &&& (\t'' -> D.asValue (toDynamic t'') :: [[Float]]) $ t'
-- (Float,([5,2],[[0.5,-2.25],[-0.25,1.25],[-2.0,0.0],[0.0,0.5],[1.5,2.5]]))
linear
  :: forall dtype batchSize inputFeatures outputFeatures
   . Tensor dtype '[outputFeatures, inputFeatures]
  -> Tensor dtype '[outputFeatures]
  -> Tensor dtype '[batchSize, inputFeatures]
  -> Tensor dtype '[batchSize, outputFeatures]
linear weight bias input = unsafePerformIO $ cast3 ATen.linear_ttt input weight bias

-- | linear'
-- https://pytorch.org/docs/stable/_modules/torch/nn/functional.html#linear
-- >>> w = fromJust [[-0.5, -2,  0.5], [1.5, -0.5, 0.5]] :: Tensor 'D.Float '[2, 3]
-- >>> b = fromJust [0, 0.5] :: Tensor 'D.Float '[2]
-- >>> t = fromJust [[-2, 0.5, 1], [0.5, 0, 0], [0, 1, 0], [0, 0, 0], [1, -1, 0]] :: Tensor 'D.Float '[5, 3]
-- >>> t' = linear' w b t
-- >>> :type t'
-- t' :: Tensor 'D.Float '[5, 2]
-- >>> dtype &&& shape &&& (\t'' -> D.asValue (toDynamic t'') :: [[Float]]) $ t'
-- (Float,([5,2],[[0.5,-2.25],[-0.25,1.25],[-2.0,0.0],[0.0,0.5],[1.5,2.5]]))
-- >>> t = fromJust [[[[-2, 0.5, 1], [0.5, 0, 0], [0, 1, 0], [0, 0, 0], [1, -1, 0]], [[-2, 0.5, 1], [0.5, 0, 0], [0, 1, 0], [0, 0, 0], [1, -1, 0]]]] :: Tensor 'D.Float '[1, 2, 5, 3]
-- >>> t' = linear' w b t
-- >>> :type t'
-- t' :: Tensor 'D.Float '[1, 2, 5, 2]
-- >>> dtype &&& shape &&& (\t' -> D.asValue (toDynamic t') :: [[[[Float]]]]) $ t'
-- (Float,([1,2,5,2],[[[[0.5,-2.25],[-0.25,1.25],[-2.0,0.0],[0.0,0.5],[1.5,2.5]],[[0.5,-2.25],[-0.25,1.25],[-2.0,0.0],[0.0,0.5],[1.5,2.5]]]]))
linear'
  :: forall dtype (inputFeatures :: Nat) (outputFeatures :: Nat) (shape :: [Nat]) (shape' :: [Nat])
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
  => Tensor dtype '[outputFeatures, inputFeatures]
  -> Tensor dtype '[outputFeatures]
  -> Tensor dtype shape
  -> Tensor dtype shape'
-- linear' weight bias input = Torch.Typed.add (matmul input $ transpose @0 @1 weight) bias
linear' weight bias input = unsafePerformIO $ cast3 ATen.linear_ttt input weight bias

-- | mkldnnLinear
-- TODO: mkldnnLinear does not return a usuable tensor value and is hence broken
-- >>> w = fromJust [[-0.5, -2,  0.5], [1.5, -0.5, 0.5]] :: Tensor 'D.Float '[2, 3]
-- >>> b = fromJust [0, 0.5] :: Tensor 'D.Float '[2]
-- >>> t = fromJust [[-2, 0.5, 1], [0.5, 0, 0], [0, 1, 0], [0, 0, 0], [1, -1, 0]] :: Tensor 'D.Float '[5, 3]
-- 
-- -- >>> t' = mkldnnLinear (toMKLDNN w) (toMKLDNN b) (toMKLDNN t)
-- -- >>> :type t'
-- -- t' :: Tensor 'D.Float '[5, 2]
-- -- >>> dtype &&& shape &&& (\t'' -> D.asValue (toDynamic t'') :: [[Float]]) $ t'
-- -- (Float,([5,2],[[0.5,-2.25],[-0.25,1.25],[-2.0,0.0],[0.0,0.5],[1.5,2.5]]))
mkldnnLinear
  :: forall dtype batchSize inputFeatures outputFeatures
   . Tensor dtype '[outputFeatures, inputFeatures]
  -> Tensor dtype '[outputFeatures]
  -> Tensor dtype '[batchSize, inputFeatures]
  -> Tensor dtype '[batchSize, outputFeatures]
mkldnnLinear weight bias input = unsafePerformIO $ cast3 ATen.mkldnn_linear_ttt input weight bias

-- fbgemm_linear_int8_weight :: Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> Float -> Float -> Tensor dtype shape -> Tensor dtype shape
-- fbgemm_linear_int8_weight _input _weight _packed _col_offsets _weight_scale _weight_zero_point _bias = unsafePerformIO $ (cast7 ATen.fbgemm_linear_int8_weight_ttttsst) _input _weight _packed _col_offsets _weight_scale _weight_zero_point _bias

-- fbgemm_linear_quantize_weight :: Tensor dtype shape -> (Tensor dtype shape,Tensor dtype shape,Double,Int)
-- fbgemm_linear_quantize_weight _input = unsafePerformIO $ (cast1 ATen.fbgemm_linear_quantize_weight_t) _input

-- fbgemm_pack_gemm_matrix_fp16 :: Tensor dtype shape -> Tensor dtype shape
-- fbgemm_pack_gemm_matrix_fp16 _input = unsafePerformIO $ (cast1 ATen.fbgemm_pack_gemm_matrix_fp16_t) _input

-- fbgemm_linear_fp16_weight :: Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape
-- fbgemm_linear_fp16_weight _input _packed_weight _bias = unsafePerformIO $ (cast3 ATen.fbgemm_linear_fp16_weight_ttt) _input _packed_weight _bias

-- fbgemm_pack_quantized_matrix :: Tensor dtype shape -> Int -> Int -> Tensor dtype shape
-- fbgemm_pack_quantized_matrix _input _K _N = unsafePerformIO $ (cast3 ATen.fbgemm_pack_quantized_matrix_tll) _input _K _N

-- fbgemm_is_cpu_supported :: Bool
-- fbgemm_is_cpu_supported  = unsafePerformIO $ (cast0 ATen.fbgemm_is_cpu_supported) 

-- | log
-- >>> dtype &&& shape $ log (ones @'D.Float @'[3,2])
-- (Float,[3,2])
log :: Tensor dtype shape -> Tensor dtype shape
log input = unsafePerformIO $ cast1 ATen.log_t input

-- | logdet
-- >>> dtype &&& shape $ logdet (ones @'D.Float @'[2,2])
-- (Float,[])
-- >>> dtype &&& shape $ logdet (ones @'D.Float @'[3,2,2])
-- (Float,[3])
logdet :: Tensor dtype shape -> Tensor dtype (Det shape)
logdet input = unsafePerformIO $ cast1 ATen.logdet_t input

type family IsFloatingPoint (dtype :: D.DType) :: Constraint where
  IsFloatingPoint 'D.Half   = ()
  IsFloatingPoint 'D.Float  = ()
  IsFloatingPoint 'D.Double = ()
  IsFloatingPoint dtype     = TypeError (    Text "Data type "
                                        :<>: ShowType dtype
                                        :<>: Text " is not floating point and hence not supported"
                                        )

type family IsIntegral (dtype :: D.DType) :: Constraint where
  IsIntegral 'D.Bool  = ()
  IsIntegral 'D.UInt8 = ()
  IsIntegral 'D.Int8  = ()
  IsIntegral 'D.Int16 = ()
  IsIntegral 'D.Int32 = ()
  IsIntegral 'D.Int64 = ()
  IsIntegral dtype    = TypeError (    Text "Data type "
                                  :<>: ShowType dtype
                                  :<>: Text " is not integral and hence not supported"
                                  )                         

-- | logsumexp
-- See https://pytorch.org/docs/stable/torch.html#torch.logsumexp.
-- >>> t = fromJust [[5, 1], [3, 2], [4, 1], [2, 7]] :: Tensor 'D.Float '[4, 2]
--
-- >>> dtype &&& shape &&& (\t' -> D.asValue (toDynamic t') :: [Float]) $ (logsumexp @1 @DropDim t :: Tensor 'D.Float '[4])
-- (Float,([4],[5.01815,3.3132617,4.0485873,7.0067153]))
--
-- >>> dtype &&& shape &&& (\t' -> D.asValue (toDynamic t') :: [[Float]]) $ (logsumexp @1 @KeepDim t :: Tensor 'D.Float '[4, 1])
-- (Float,([4,1],[[5.01815],[3.3132617],[4.0485873],[7.0067153]]))
--
-- >>> dtype &&& shape &&& (\t' -> D.asValue (toDynamic t') :: [Float]) $ (logsumexp @0 @DropDim t :: Tensor 'D.Float '[2])
-- (Float,([2],[5.44019,7.0116277]))
--
-- >>> dtype &&& shape &&& (\t' -> D.asValue (toDynamic t') :: [[Float]]) $ (logsumexp @0 @KeepDim t :: Tensor 'D.Float '[1, 2])
-- (Float,([1,2],[[5.44019,7.0116277]]))
logsumexp
  :: forall dim keepOrDropDim dtype shape
   . (KnownNat dim, KnownKeepOrDropDim keepOrDropDim, Reifies dtype D.DType, IsFloatingPoint dtype)
  => Tensor dtype shape
  -> Tensor dtype (ConditionalDropDimension shape dim keepOrDropDim)
logsumexp t = unsafePerformIO $ cast3 ATen.logsumexp_tlb t (natValI @dim) (keepOrDropDimVal @keepOrDropDim)

-- margin_ranking_loss :: Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> Double -> Int -> Tensor dtype shape
-- margin_ranking_loss _input1 _input2 _target _margin _reduction = unsafePerformIO $ (cast5 ATen.margin_ranking_loss_tttdl) _input1 _input2 _target _margin _reduction

-- | matrixPower
-- >>> dtype &&& shape $ matrixPower 2 (ones @'D.Float @'[3,4,4])
-- (Float,[3,4,4])
matrixPower :: Int -> Tensor dtype shape -> Tensor dtype (Square shape)
matrixPower n input = unsafePerformIO $ cast2 ATen.matrix_power_tl input n

-- | maxValues
-- >>> dtype &&& shape $ maxValues @0 @KeepDim (ones @'D.Float @'[3,4,5])
-- (Float,[1,4,5])
-- >>> dtype &&& shape $ maxValues @0 @DropDim (ones @'D.Float @'[3,4,5])
-- (Float,[4,5])
-- >>> dtype &&& shape $ maxValues @1 @KeepDim (ones @'D.Float @'[3,4,5])
-- (Float,[3,1,5])
-- >>> dtype &&& shape $ maxValues @1 @DropDim (ones @'D.Float @'[3,4,5])
-- (Float,[3,5])
maxValues
  :: forall dim keepOrDropDim dtype shape
   . (KnownNat dim, KnownKeepOrDropDim keepOrDropDim)
  => Tensor dtype shape
  -> Tensor dtype (ConditionalDropDimension shape dim keepOrDropDim)
maxValues input = unsafePerformIO $ cast3 ATen.max_values_tlb input (natValI @dim) (keepOrDropDimVal @keepOrDropDim)

-- max_pool1d_with_indices :: Tensor dtype shape -> Int -> Int -> Int -> Int -> Bool -> (Tensor dtype shape,Tensor dtype shape)
-- max_pool1d_with_indices _input _kernel_size _stride _padding _dilation _ceil_mode = unsafePerformIO $ (cast6 ATen.max_pool1d_with_indices_tllllb) _input _kernel_size _stride _padding _dilation _ceil_mode

-- | maxPool1d
-- >>> t = maxPool1d @1 @1 @0 (ones @'D.Float @'[1,3,4])
-- >>> shape t
-- [1,3,4]
-- >>> :t t
-- t :: Tensor 'D.Float '[1, 3, 4]
maxPool1d
  :: forall kernelSize stride padding channelSize inputSize batchSize dtype outputSize
   . ( All KnownNat [kernelSize, stride, padding, channelSize, inputSize, batchSize]
     , ConvSideCheck inputSize kernelSize stride padding outputSize
     )
  => Tensor dtype '[batchSize, channelSize, inputSize]
  -> Tensor dtype '[batchSize, channelSize, outputSize]
maxPool1d input = unsafePerformIO $ cast6 ATen.max_pool1d_tllllb
                                          input
                                          (natValI @kernelSize)
                                          (natValI @stride)
                                          (natValI @padding)
                                          (1 :: Int)
                                          False

-- | maxPool2d
-- >>> t = maxPool2d @'(1,1) @'(1,1) @'(0,0) (ones @'D.Float @'[1,3,4,5])
-- >>> shape t
-- [1,3,4,5]
-- >>> :t t
-- t :: Tensor 'D.Float '[1, 3, 4, 5]
maxPool2d
  :: forall kernelSize stride padding channelSize inputSize0 inputSize1 batchSize dtype outputSize0 outputSize1
   . (All KnownNat [ Fst kernelSize, Snd kernelSize
                   , Fst stride, Snd stride
                   , Fst padding, Snd padding
                   , channelSize
                   , inputSize0, inputSize1
                   , batchSize
                   ]
     , ConvSideCheck inputSize0 (Fst kernelSize) (Fst stride) (Fst padding) outputSize0
     , ConvSideCheck inputSize1 (Snd kernelSize) (Snd stride) (Snd padding) outputSize1
     )
  => Tensor dtype '[batchSize, channelSize, inputSize0, inputSize1]
  -> Tensor dtype '[batchSize, channelSize, outputSize0, outputSize1]
maxPool2d input = unsafePerformIO $ cast6
  ATen.max_pool2d_tllllb
  input
  ([natValI @(Fst kernelSize), natValI @(Snd kernelSize)] :: [Int])
  ([natValI @(Fst stride), natValI @(Snd stride)] :: [Int])
  ([natValI @(Fst padding), natValI @(Snd padding)] :: [Int])
  ([1, 1] :: [Int])
  False

-- | mkldnnMaxPool2d
-- >>> t = mkldnnMaxPool2d @'(1,1) @'(1,1) @'(0,0) (toMKLDNN (ones::Tensor 'D.Float '[1,3,4,5]))
-- >>> shape t
-- [1,3,4,5]
-- >>> :t t
-- t :: Tensor 'D.Float '[1, 3, 4, 5]
mkldnnMaxPool2d
  :: forall kernelSize stride padding channelSize inputSize0 inputSize1 batchSize dtype outputSize0 outputSize1
   . (All KnownNat [Fst kernelSize, Snd kernelSize
                   , Fst stride, Snd stride
                   , Fst padding, Snd padding
                   , channelSize
                   , inputSize0, inputSize1
                   , batchSize
                   ]
     , ConvSideCheck inputSize0 (Fst kernelSize) (Fst stride) (Fst padding) outputSize0
     , ConvSideCheck inputSize1 (Snd kernelSize) (Snd stride) (Snd padding) outputSize1
     )
  => Tensor dtype '[batchSize, channelSize, inputSize0, inputSize1]
  -> Tensor dtype '[batchSize, channelSize, outputSize0, outputSize1]
mkldnnMaxPool2d input = unsafePerformIO $ cast6
  ATen.mkldnn_max_pool2d_tllllb
  input
  ([natValI @(Fst kernelSize), natValI @(Snd kernelSize)] :: [Int])
  ([natValI @(Fst stride), natValI @(Snd stride)] :: [Int])
  ([natValI @(Fst padding), natValI @(Snd padding)] :: [Int])
  ([1, 1] :: [Int])
  False

-- | quantizedMaxPool2d
-- TODO: quantized functions are not avaiable on libtorch 1.2.
-- -- >>> t = quantizedMaxPool2d @'(1,1) @'(1,1) @'(0,0) (ones::Tensor 'D.Float '[1,3,4,5])
-- -- >>> shape t
-- -- [1,3,4,5]
-- -- >>> :t t
-- -- t :: Tensor 'D.Float '[1, 3, 4, 5]
quantizedMaxPool2d
  :: forall kernelSize stride padding channelSize inputSize0 inputSize1 batchSize dtype outputSize0 outputSize1
   . ( All KnownNat [ Fst kernelSize, Snd kernelSize
                    , Fst stride, Snd stride
                    , Fst padding, Snd padding
                    , channelSize
                    , inputSize0, inputSize1
                    , batchSize
                    ]
     , ConvSideCheck inputSize0 (Fst kernelSize) (Fst stride) (Fst padding) outputSize0
     , ConvSideCheck inputSize1 (Snd kernelSize) (Snd stride) (Snd padding) outputSize1
     )
  => Tensor dtype '[batchSize, channelSize, inputSize0, inputSize1]
  -> Tensor dtype '[batchSize, channelSize, outputSize0, outputSize1]
quantizedMaxPool2d input = unsafePerformIO $ cast5
  ATen.quantized_max_pool2d_tllll
  input
  ([natValI @(Fst kernelSize), natValI @(Snd kernelSize)] :: [Int])
  ([natValI @(Fst stride), natValI @(Snd stride)] :: [Int])
  ([natValI @(Fst padding), natValI @(Snd padding)] :: [Int])
  ([1, 1] :: [Int])

-- | maskedFill
-- >>> t = ones @'D.Float @'[2, 1, 3]
-- >>> m = fromJust [[False], [True], [False]] :: Tensor 'D.Bool '[3, 1]
-- >>> t' = maskedFill @Float m 0.5 t
-- >>> :type t'
-- t' :: Tensor 'D.Float '[2, 3, 3]
-- >>> dtype &&& shape &&& (\u -> D.asValue (toDynamic u) :: [[[Float]]]) $ t'
-- (Float,([2,3,3],[[[1.0,1.0,1.0],[0.5,0.5,0.5],[1.0,1.0,1.0]],[[1.0,1.0,1.0],[0.5,0.5,0.5],[1.0,1.0,1.0]]]))
maskedFill
  :: forall a dtype shape shape' shape''
   . (D.Scalar a, shape'' ~ Broadcast shape shape')
  => Tensor 'D.Bool shape'
  -> a
  -> Tensor dtype shape
  -> Tensor dtype shape''
maskedFill mask value input =
  unsafePerformIO $ cast3 ATen.masked_fill_tts input mask value

-- | maxPool3d
-- >>> t = maxPool3d @'(1,1,1) @'(1,1,1) @'(0,0,0) (ones @'D.Float @'[1,3,4,5,6])
-- >>> shape t
-- [1,3,4,5,6]
-- >>> :t t
-- t :: Tensor 'D.Float '[1, 3, 4, 5, 6]
maxPool3d
  :: forall kernelSize stride padding channelSize
            inputSize0 inputSize1 inputSize2
            batchSize dtype
            outputSize0 outputSize1 outputSize2
   . ( All KnownNat [ Fst3 kernelSize, Snd3 kernelSize, Trd3 kernelSize
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
  => Tensor dtype '[batchSize, channelSize, inputSize0, inputSize1, inputSize2]
  -> Tensor dtype '[batchSize, channelSize, outputSize0, outputSize1, outputSize2]
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

-- | minValues
-- >>> dtype &&& shape $ minValues @0 @KeepDim (ones @'D.Float @'[3,4,5])
-- (Float,[1,4,5])
-- >>> dtype &&& shape $ minValues @0 @DropDim (ones @'D.Float @'[3,4,5])
-- (Float,[4,5])
-- >>> dtype &&& shape $ minValues @1 @KeepDim (ones @'D.Float @'[3,4,5])
-- (Float,[3,1,5])
-- >>> dtype &&& shape $ minValues @1 @DropDim (ones @'D.Float @'[3,4,5])
-- (Float,[3,5])
minValues
  :: forall dim keepOrDropDim dtype shape
   . (KnownNat dim, KnownKeepOrDropDim keepOrDropDim)
  => Tensor dtype shape
  -> Tensor dtype (ConditionalDropDimension shape dim keepOrDropDim)
minValues input = unsafePerformIO $ cast3 ATen.min_values_tlb
                                          input
                                          (natValI @dim)
                                          (keepOrDropDimVal @keepOrDropDim)

-- mkldnn_convolution :: Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> [Int] -> [Int] -> [Int] -> Int -> Tensor dtype shape
-- mkldnn_convolution _input _weight _bias _padding _stride _dilation _groups = unsafePerformIO $ (cast7 ATen.mkldnn_convolution_tttllll) _input _weight _bias _padding _stride _dilation _groups

-- miopen_batch_norm :: Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> Bool -> Double -> Double -> (Tensor dtype shape,Tensor dtype shape,Tensor dtype shape)
-- miopen_batch_norm _input _weight _bias _running_mean _running_var _training _exponential_average_factor _epsilon = unsafePerformIO $ (cast8 ATen.miopen_batch_norm_tttttbdd) _input _weight _bias _running_mean _running_var _training _exponential_average_factor _epsilon

-- miopen_convolution :: Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> [Int] -> [Int] -> [Int] -> Int -> Bool -> Bool -> Tensor dtype shape
-- miopen_convolution _input _weight _bias _padding _stride _dilation _groups _benchmark _deterministic = unsafePerformIO $ (cast9 ATen.miopen_convolution_tttllllbb) _input _weight _bias _padding _stride _dilation _groups _benchmark _deterministic

-- miopen_convolution_transpose :: Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> [Int] -> [Int] -> [Int] -> [Int] -> Int -> Bool -> Bool -> Tensor dtype shape
-- miopen_convolution_transpose _input _weight _bias _padding _output_padding _stride _dilation _groups _benchmark _deterministic = unsafePerformIO $ (cast10 ATen.miopen_convolution_transpose_tttlllllbb) _input _weight _bias _padding _output_padding _stride _dilation _groups _benchmark _deterministic

-- miopen_depthwise_convolution :: Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> [Int] -> [Int] -> [Int] -> Int -> Bool -> Bool -> Tensor dtype shape
-- miopen_depthwise_convolution _input _weight _bias _padding _stride _dilation _groups _benchmark _deterministic = unsafePerformIO $ (cast9 ATen.miopen_depthwise_convolution_tttllllbb) _input _weight _bias _padding _stride _dilation _groups _benchmark _deterministic

-- miopen_rnn :: Tensor dtype shape -> [Tensor dtype shape] -> Int -> Tensor dtype shape -> Tensor dtype shape -> Int -> Int -> Int -> Bool -> Double -> Bool -> Bool -> [Int] -> Tensor dtype shape -> (Tensor dtype shape,Tensor dtype shape,Tensor dtype shape,Tensor dtype shape,Tensor dtype shape)
-- miopen_rnn _input _weight _weight_stride0 _hx _cx _mode _hidden_size _num_layers _batch_first _dropout _train _bidirectional _batch_sizes _dropout_state = unsafePerformIO $ (cast14 ATen.miopen_rnn_tllttlllbdbblt) _input _weight _weight_stride0 _hx _cx _mode _hidden_size _num_layers _batch_first _dropout _train _bidirectional _batch_sizes _dropout_state

-- | mm
-- >>> dtype &&& shape $ mm (ones @'D.Float @'[3,2]) (zeros @'D.Float @'[2,4])
-- (Float,[3,4])
mm :: Tensor dtype '[n,k] -> Tensor dtype '[k,m] -> Tensor dtype '[n,m]
mm a b = unsafePerformIO $ cast2 ATen.mm_tt a b

-- mode :: Tensor dtype shape -> Int -> Bool -> (Tensor dtype shape,Tensor dtype shape)
-- mode _input _dim _keepdim = unsafePerformIO $ (cast3 ATen.mode_tlb) _input _dim _keepdim

-- | mv
-- >>> dtype &&& shape $ mv (ones @'D.Float @'[3,2]) (zeros @'D.Float @'[2])
-- (Float,[3])
mv :: Tensor dtype '[n,m] -> Tensor dtype '[m] -> Tensor dtype '[n]
mv input vec = unsafePerformIO $ cast2 ATen.mv_tt input vec

-- mvlgamma :: Tensor dtype shape -> Int -> Tensor dtype shape
-- mvlgamma _input _p = unsafePerformIO $ (cast2 ATen.mvlgamma_tl) _input _p

-- narrow :: Tensor dtype shape -> Int -> Int -> Int -> Tensor dtype shape
-- narrow _input _dim _start _length = unsafePerformIO $ (cast4 ATen.narrow_tlll) _input _dim _start _length

-- native_batch_norm :: Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> Bool -> Double -> Double -> (Tensor dtype shape,Tensor dtype shape,Tensor dtype shape)
-- native_batch_norm _input _weight _bias _running_mean _running_var _training _momentum _eps = unsafePerformIO $ (cast8 ATen.native_batch_norm_tttttbdd) _input _weight _bias _running_mean _running_var _training _momentum _eps

-- batch_norm_stats :: Tensor dtype shape -> Double -> (Tensor dtype shape,Tensor dtype shape)
-- batch_norm_stats _input _eps = unsafePerformIO $ (cast2 ATen.batch_norm_stats_td) _input _eps

-- batch_norm_elemt :: Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> Double -> Tensor dtype shape
-- batch_norm_elemt _input _weight _bias _mean _invstd _eps = unsafePerformIO $ (cast6 ATen.batch_norm_elemt_tttttd) _input _weight _bias _mean _invstd _eps

-- batch_norm_gather_stats :: Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> Double -> Double -> Int -> (Tensor dtype shape,Tensor dtype shape)
-- batch_norm_gather_stats _input _mean _invstd _running_mean _running_var _momentum _eps _count = unsafePerformIO $ (cast8 ATen.batch_norm_gather_stats_tttttddl) _input _mean _invstd _running_mean _running_var _momentum _eps _count

-- batch_norm_gather_stats_with_counts :: Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> Double -> Double -> [Int] -> (Tensor dtype shape,Tensor dtype shape)
-- batch_norm_gather_stats_with_counts _input _mean _invstd _running_mean _running_var _momentum _eps _counts = unsafePerformIO $ (cast8 ATen.batch_norm_gather_stats_with_counts_tttttddl) _input _mean _invstd _running_mean _running_var _momentum _eps _counts

-- batch_norm_update_stats :: Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> Double -> (Tensor dtype shape,Tensor dtype shape)
-- batch_norm_update_stats _input _running_mean _running_var _momentum = unsafePerformIO $ (cast4 ATen.batch_norm_update_stats_tttd) _input _running_mean _running_var _momentum

-- | onesLike
-- >>> dtype &&& shape $ onesLike (ones @'D.Float @'[3,4,5])
-- (Float,[3,4,5])
onesLike :: Tensor dtype shape -> Tensor dtype shape
onesLike input = unsafePerformIO $ cast1 ATen.ones_like_t input

-- pairwise_distance :: Tensor dtype shape -> Tensor dtype shape -> Double -> Double -> Bool -> Tensor dtype shape
-- pairwise_distance _x1 _x2 _p _eps _keepdim = unsafePerformIO $ (cast5 ATen.pairwise_distance_ttddb) _x1 _x2 _p _eps _keepdim

-- cdist :: Tensor dtype shape -> Tensor dtype shape -> Double -> Tensor dtype shape
-- cdist _x1 _x2 _p = unsafePerformIO $ (cast3 ATen.cdist_ttd) _x1 _x2 _p

-- pdist :: Tensor dtype shape -> Double -> Tensor dtype shape
-- pdist _input _p = unsafePerformIO $ (cast2 ATen.pdist_td) _input _p

-- cosine_similarity :: Tensor dtype shape -> Tensor dtype shape -> Int -> Double -> Tensor dtype shape
-- cosine_similarity _x1 _x2 _dim _eps = unsafePerformIO $ (cast4 ATen.cosine_similarity_ttld) _x1 _x2 _dim _eps

-- pixel_shuffle :: Tensor dtype shape -> Int -> Tensor dtype shape
-- pixel_shuffle _input _upscale_factor = unsafePerformIO $ (cast2 ATen.pixel_shuffle_tl) _input _upscale_factor

-- pin_memory :: Tensor dtype shape -> Tensor dtype shape
-- pin_memory _input = unsafePerformIO $ (cast1 ATen.pin_memory_t) _input

-- pinverse :: Tensor dtype shape -> Double -> Tensor dtype shape
-- pinverse _input _rcond = unsafePerformIO $ (cast2 ATen.pinverse_td) _input _rcond

-- poisson_nll_loss :: Tensor dtype shape -> Tensor dtype shape -> Bool -> Bool -> Double -> Int -> Tensor dtype shape
-- poisson_nll_loss _input _target _log_input _full _eps _reduction = unsafePerformIO $ (cast6 ATen.poisson_nll_loss_ttbbdl) _input _target _log_input _full _eps _reduction

-- | randLike
-- >>> t <- randLike (ones @'D.Float @'[3,4,5])
-- >>> dtype &&& shape $ t
-- (Float,[3,4,5])
randLike :: Tensor dtype shape -> IO (Tensor dtype shape)
randLike = cast1 ATen.rand_like_t

-- | randnLike
-- >>> t <- randnLike (ones @'D.Float @'[3,4,5])
-- >>> dtype &&& shape $ t
-- (Float,[3,4,5])
randnLike :: Tensor dtype shape -> IO (Tensor dtype shape)
randnLike = cast1 ATen.randn_like_t

-- reciprocal :: Tensor dtype shape -> Tensor dtype shape
-- reciprocal _input = unsafePerformIO $ (cast1 ATen.reciprocal_t) _input

-- | neg
-- >>> dtype &&& shape $ neg (ones @'D.Float @'[3,2])
-- (Float,[3,2])
neg :: Tensor dtype shape -> Tensor dtype shape
neg input = unsafePerformIO $ cast1 ATen.neg_t input

-- | round
-- >>> dtype &&& shape $ round (ones @'D.Float @'[3,2])
-- (Float,[3,2])
round :: Tensor dtype shape -> Tensor dtype shape
round input = unsafePerformIO $ cast1 ATen.round_t input

-- | prelu
-- >>> dtype &&& shape $ prelu (ones @'D.Float @'[]) (ones @'D.Float @'[3,2])
-- (Float,[3,2])
prelu :: Tensor dtype '[] -> Tensor dtype shape -> Tensor dtype shape
prelu weight input = unsafePerformIO $ cast2 ATen.prelu_tt input weight

-- | gelu
-- >>> dtype &&& shape $ round (ones @'D.Float @'[3,2])
-- (Float,[3,2])
gelu :: Tensor dtype shape -> Tensor dtype shape
gelu input = unsafePerformIO $ cast1 ATen.gelu_t input

-- hardshrink :: Tensor dtype shape -> Float -> Tensor dtype shape
-- hardshrink _input _lambd = unsafePerformIO $ (cast2 ATen.hardshrink_ts) _input _lambd

-- | rsqrt
-- >>> dtype &&& shape $ rsqrt (ones @'D.Float @'[3,2])
-- (Float,[3,2])
rsqrt :: Tensor dtype shape -> Tensor dtype shape
rsqrt input = unsafePerformIO $ cast1 ATen.rsqrt_t input

-- | celu
-- >>> dtype &&& shape $ celu 3.0 (ones @'D.Float @'[3,2])
-- (Float,[3,2])
celu :: Float -> Tensor dtype shape -> Tensor dtype shape
celu alpha input = unsafePerformIO $ cast2 ATen.celu_ts input alpha

-- slice :: Tensor dtype shape -> Int -> Int -> Int -> Int -> Tensor dtype shape
-- slice _input _dim _start _end _step = unsafePerformIO $ (cast5 ATen.slice_tllll) _input _dim _start _end _step

-- slogdet :: Tensor dtype shape -> (Tensor dtype shape,Tensor dtype shape)
-- slogdet _input = unsafePerformIO $ (cast1 ATen.slogdet_t) _input

-- smm :: Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape
-- smm _input _mat2 = unsafePerformIO $ (cast2 ATen.smm_tt) _input _mat2

-- split :: Tensor dtype shape -> Int -> Int -> [Tensor dtype shape]
-- split _input _split_size _dim = unsafePerformIO $ (cast3 ATen.split_tll) _input _split_size _dim

-- split_with_sizes :: Tensor dtype shape -> [Int] -> Int -> [Tensor dtype shape]
-- split_with_sizes _input _split_sizes _dim = unsafePerformIO $ (cast3 ATen.split_with_sizes_tll) _input _split_sizes _dim

-- sspaddmm :: Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> Float -> Float -> Tensor dtype shape
-- sspaddmm _input _mat1 _mat2 _beta _alpha = unsafePerformIO $ (cast5 ATen.sspaddmm_tttss) _input _mat1 _mat2 _beta _alpha

type family StackImpl (dim :: Nat) (tensors :: [a]) (count :: Nat) :: Maybe ((DeviceType, Nat), D.DType, [Nat]) where
  StackImpl dim '[]                                                                 count = Nothing
  StackImpl dim (Tensor device dtype shape ': '[])                                  count = MaybeTriple (Just device) (Just dtype) (ComputeStackShape shape dim count)
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

type family StackCheck (res :: Maybe ((DeviceType, Nat), D.DType, [Nat])) :: ((DeviceType, Nat), D.DType, [Nat]) where
  StackCheck 'Nothing                        = TypeError (Text "Stacking impossible.")
  StackCheck ('Just '(device, dtype, shape)) = '(device, dtype, shape)

-- | Stack
-- >>> type Ty = Stack 0 '[Tensor 'D.Float '[]]
-- >>> :kind! Ty
-- Ty :: (D.DType, [Nat])
-- = '( 'D.Float, '[1])
-- >>> type Ty = Stack 0 '[Tensor 'D.Float '[2,2]]
-- >>> :kind! Ty
-- Ty :: (D.DType, [Nat])
-- = '( 'D.Float, '[1, 2, 2])
-- >>> type Ty = Stack 1 '[Tensor 'D.Float '[2,2]]
-- >>> :kind! Ty
-- Ty :: (D.DType, [Nat])
-- = '( 'D.Float, '[2, 1, 2])
-- >>> type Ty = Stack 2 '[Tensor 'D.Float '[2,2]]
-- >>> :kind! Ty
-- Ty :: (D.DType, [Nat])
-- = '( 'D.Float, '[2, 2, 1])
-- >>> type Ty = Stack 2 '[Tensor 'D.Float '[2,2], Tensor 'D.Float '[2,2], Tensor 'D.Float '[2,2]]
-- >>> :kind! Ty
-- Ty :: (D.DType, [Nat])
-- = '( 'D.Float, '[2, 2, 3])
type Stack dim tensors = StackCheck (StackImpl dim tensors 1)

-- | stack
-- >>> t = ones @'D.Float @'[]
-- >>> t' = stack @0 (t :. HNil)
-- >>> :type t'
-- t' :: Tensor 'D.Float '[1]
-- >>> dtype &&& shape &&& (\t'' -> D.asValue (toDynamic t'') :: [Float]) $ t'
-- (Float,([1],[1.0]))
-- >>> t = ones @'D.Float @'[2,2]
-- >>> t' = stack @0 (t :. HNil)
-- >>> :type t'
-- t' :: Tensor 'D.Float '[1, 2, 2]
-- >>> dtype &&& shape &&& (\t'' -> D.asValue (toDynamic t'') :: [[[Float]]]) $ t'
-- (Float,([1,2,2],[[[1.0,1.0],[1.0,1.0]]]))
-- >>> t' = stack @1 (t :. HNil)
-- >>> :type t'
-- t' :: Tensor 'D.Float '[2, 1, 2]
-- >>> dtype &&& shape &&& (\t'' -> D.asValue (toDynamic t'') :: [[[Float]]]) $ t'
-- (Float,([2,1,2],[[[1.0,1.0]],[[1.0,1.0]]]))
-- >>> t' = stack @2 (t :. HNil)
-- >>> :type t'
-- t' :: Tensor 'D.Float '[2, 2, 1]
-- >>> dtype &&& shape &&& (\t'' -> D.asValue (toDynamic t'') :: [[[Float]]]) $ t'
-- (Float,([2,2,1],[[[1.0],[1.0]],[[1.0],[1.0]]]))
-- >>> t' = stack @2 (t :. t :. t :. HNil)
-- >>> :type t'
-- t' :: Tensor 'D.Float '[2, 2, 3]
-- >>> dtype &&& shape &&& (\t'' -> D.asValue (toDynamic t'') :: [[[Float]]]) $ t'
-- (Float,([2,2,3],[[[1.0,1.0,1.0],[1.0,1.0,1.0]],[[1.0,1.0,1.0],[1.0,1.0,1.0]]]))
stack
  :: forall dim device dtype shape tensors
   . ( KnownNat dim
     , '(device, dtype, shape) ~ Stack dim tensors
     , HFoldrM IO TensorListFolds [D.ATenTensor] tensors
     , Apply TensorListFolds [D.ATenTensor] (HUnfoldMRes IO [D.ATenTensor] tensors)
     , HUnfoldM IO TensorListFolds (HUnfoldMRes IO [D.ATenTensor] tensors) tensors
     )
  => HList tensors
  -> Tensor device dtype shape
stack tensors = unsafePerformIO $ cast2 ATen.stack_ll tensors (natValI @dim :: Int)

-- stft :: Tensor dtype shape -> Int -> Int -> Int -> Tensor dtype shape -> Bool -> Bool -> Tensor dtype shape
-- stft _input _n_fft _hop_length _win_length _window _normalized _onesided = unsafePerformIO $ (cast7 ATen.stft_tllltbb) _input _n_fft _hop_length _win_length _window _normalized _onesided

-- stride :: Tensor dtype shape -> Int -> Int
-- stride _input _dim = unsafePerformIO $ (cast2 ATen.stride_tl) _input _dim

-- t :: Tensor dtype shape -> Tensor dtype shape
-- t _input = unsafePerformIO $ (cast1 ATen.t_t) _input

-- |
-- >>> dtype &&& shape $ tan (ones :: Tensor 'D.Float '[3,2])
-- (Float,[3,2])
tan :: Tensor dtype shape -> Tensor dtype shape
tan _input = unsafePerformIO $ (cast1 ATen.tan_t) _input

-- tensordot :: Tensor dtype shape -> Tensor dtype shape -> [Int] -> [Int] -> Tensor dtype shape
-- tensordot _input _other _dims_input _dims_other = unsafePerformIO $ (cast4 ATen.tensordot_ttll) _input _other _dims_input _dims_other

-- threshold :: Tensor dtype shape -> Float -> Float -> Tensor dtype shape
-- threshold _input _threshold _value = unsafePerformIO $ (cast3 ATen.threshold_tss) _input _threshold _value

-- one_hot :: Tensor dtype shape -> Int -> Tensor dtype shape
-- one_hot _input _num_classes = unsafePerformIO $ (cast2 ATen.one_hot_tl) _input _num_classes

-- flip :: Tensor dtype shape -> [Int] -> Tensor dtype shape
-- flip _input _dims = unsafePerformIO $ (cast2 ATen.flip_tl) _input _dims

-- roll :: Tensor dtype shape -> Int -> Int -> Tensor dtype shape
-- roll _input _shifts _dims = unsafePerformIO $ (cast3 ATen.roll_tll) _input _shifts _dims

-- rot90 :: Tensor dtype shape -> Int -> [Int] -> Tensor dtype shape
-- rot90 _input _k _dims = unsafePerformIO $ (cast3 ATen.rot90_tll) _input _k _dims

-- triplet_margin_loss :: Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> Double -> Double -> Double -> Bool -> Int -> Tensor dtype shape
-- triplet_margin_loss _anchor _positive _negative _margin _p _eps _swap _reduction = unsafePerformIO $ (cast8 ATen.triplet_margin_loss_tttdddbl) _anchor _positive _negative _margin _p _eps _swap _reduction

-- |
-- >>> dtype &&& shape $ trunc (ones :: Tensor 'D.Float '[3,2])
-- (Float,[3,2])
trunc :: Tensor dtype shape -> Tensor dtype shape
trunc _input = unsafePerformIO $ (cast1 ATen.trunc_t) _input

-- unique_dim :: Tensor dtype shape -> Int -> Bool -> Bool -> Bool -> (Tensor dtype shape,Tensor dtype shape,Tensor dtype shape)
-- unique_dim _input _dim _sorted _return_inverse _return_counts = unsafePerformIO $ (cast5 ATen.unique_dim_tlbbb) _input _dim _sorted _return_inverse _return_counts

-- unique_consecutive :: Tensor dtype shape -> Bool -> Bool -> Int -> (Tensor dtype shape,Tensor dtype shape,Tensor dtype shape)
-- unique_consecutive _input _return_inverse _return_counts _dim = unsafePerformIO $ (cast4 ATen.unique_consecutive_tbbl) _input _return_inverse _return_counts _dim

-- unique_dim_consecutive :: Tensor dtype shape -> Int -> Bool -> Bool -> (Tensor dtype shape,Tensor dtype shape,Tensor dtype shape)
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
-- >>> t = fromJust [1, 2, 3, 4] :: Tensor 'D.Int64 '[4]
-- >>> t' = unsqueeze @0 t
-- >>> :type t'
-- t' :: Tensor 'D.Int64 '[1, 4]
-- >>> dtype &&& shape &&& (\u -> D.asValue (toDynamic u) :: [[Int]]) $ t'
-- (Int64,([1,4],[[1,2,3,4]]))
-- >>> t'' = unsqueeze @1 t
-- >>> :type t''
-- t'' :: Tensor 'D.Int64 '[4, 1]
-- >>> dtype &&& shape &&& (\u -> D.asValue (toDynamic u) :: [[Int]]) $ t''
-- (Int64,([4,1],[[1],[2],[3],[4]]))
unsqueeze
  :: forall dim dtype shape shape'
   . (KnownNat dim, shape' ~ Unsqueeze shape dim)
  => Tensor dtype shape
  -> Tensor dtype shape'
unsqueeze input = unsafePerformIO $ cast2 ATen.unsqueeze_tl input (natValI @dim)

-- where' :: Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape
-- where' _condition _input _other = unsafePerformIO $ (cast3 ATen.where_ttt) _condition _input _other

-- where_ :: Tensor dtype shape -> [Tensor dtype shape]
-- where_ _condition = unsafePerformIO $ (cast1 ATen.where_t) _condition

-- norm_except_dim :: Tensor dtype shape -> Int -> Int -> Tensor dtype shape
-- norm_except_dim _v _pow _dim = unsafePerformIO $ (cast3 ATen.norm_except_dim_tll) _v _pow _dim

-- |
-- >>> dtype &&& shape $ zerosLike (ones :: Tensor 'D.Float '[3,4,5])
-- (Float,[3,4,5])
zerosLike :: Tensor dtype shape -> Tensor dtype shape
zerosLike _input = unsafePerformIO $ (cast1 ATen.zeros_like_t) _input

-- native_norm :: Tensor dtype shape -> Float -> Tensor dtype shape
-- native_norm _input _p = unsafePerformIO $ (cast2 ATen.native_norm_ts) _input _p

-- |
-- >>> t <- clone (ones :: Tensor 'D.Float '[3,2])
-- >>> dtype &&& shape $ t
-- (Float,[3,2])
clone :: Tensor dtype shape -> IO (Tensor dtype shape)
clone _input = (cast1 ATen.clone_t) _input

-- s_native_addmm :: Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> Float -> Float -> Tensor dtype shape
-- s_native_addmm _input _mat1 _mat2 _beta _alpha = unsafePerformIO $ (cast5 ATen.s_native_addmm_tttss) _input _mat1 _mat2 _beta _alpha

-- |
-- >>> t = addmm (ones :: Tensor 'D.Float '[]) (ones :: Tensor 'D.Float '[3,2]) (zeros :: Tensor 'D.Float '[2,4]) 1 1
-- >>> dtype &&& shape $ t
-- (Float,[3,4])
-- >>> :t t
-- t :: Tensor 'D.Float '[3, 4]
addmm
  :: forall shape' shape n k m dtype
   . (KnownNat n, KnownNat m, KnownNat k,
      shape' ~ Broadcast shape '[n,m])
  => Tensor dtype shape
  -> Tensor dtype '[n,k]
  -> Tensor dtype '[k,m]
  -> Float
  -> Float
  -> Tensor dtype shape'
addmm _input _mat1 _mat2 _beta _alpha = unsafePerformIO $ (cast5 ATen.addmm_tttss) _input _mat1 _mat2 _beta _alpha

-- hspmm :: Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape
-- hspmm _mat1 _mat2 = unsafePerformIO $ (cast2 ATen.hspmm_tt) _mat1 _mat2

numel :: Tensor dtype shape -> Int
numel _input = unsafePerformIO $ (cast1 ATen.numel_t) _input

-- unbind :: Tensor dtype shape -> Int -> [Tensor dtype shape]
-- unbind _input _dim = unsafePerformIO $ (cast2 ATen.unbind_tl) _input _dim

-- mkldnn_reorder_conv2d_weight :: Tensor dtype shape -> (Int,Int) -> (Int,Int) -> (Int,Int) -> Int -> Tensor dtype shape
-- mkldnn_reorder_conv2d_weight _input _padding _stride _dilation _groups = unsafePerformIO $ (cast5 ATen.mkldnn_reorder_conv2d_weight_tllll) _input _padding _stride _dilation _groups

--quantize_linear :: Tensor dtype shape -> Double -> Int -> DType -> Tensor dtype shape
--quantize_linear _input _scale _zero_point _dtype = unsafePerformIO $ (cast4 ATen.quantize_linear_tdls) _input _scale _zero_point _dtype

--quantize_linear_per_channel :: Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> [Int] -> DType -> Tensor dtype shape
--quantize_linear_per_channel _input _scales _zero_points _axis _dtype = unsafePerformIO $ (cast5 ATen.quantize_linear_per_channel_tttls) _input _scales _zero_points _axis _dtype

-- dequantize :: Tensor dtype shape -> Tensor dtype shape
-- dequantize _input = unsafePerformIO $ (cast1 ATen.dequantize_t) _input

q_scale :: Tensor dtype shape -> Double
q_scale _input = unsafePerformIO $ (cast1 ATen.q_scale_t) _input

q_zero_point :: Tensor dtype shape -> Int
q_zero_point _input = unsafePerformIO $ (cast1 ATen.q_zero_point_t) _input

-- int_repr :: Tensor dtype shape -> Tensor dtype shape
-- int_repr _input = unsafePerformIO $ (cast1 ATen.int_repr_t) _input

-- fake_quantize_per_tensor_affine :: Tensor dtype shape -> Double -> Int -> Int -> Int -> Tensor dtype shape
-- fake_quantize_per_tensor_affine _input _scale _zero_point _quant_min _quant_max = unsafePerformIO $ (cast5 ATen.fake_quantize_per_tensor_affine_tdlll) _input _scale _zero_point _quant_min _quant_max

-- meshgrid :: [Tensor dtype shape] -> [Tensor dtype shape]
-- meshgrid _tensors = unsafePerformIO $ (cast1 ATen.meshgrid_l) _tensors

-- cartesian_prod :: [Tensor dtype shape] -> Tensor dtype shape
-- cartesian_prod _tensors = unsafePerformIO $ (cast1 ATen.cartesian_prod_l) _tensors

-- combinations :: Tensor dtype shape -> Int -> Bool -> Tensor dtype shape
-- combinations _input _r _with_replacement = unsafePerformIO $ (cast3 ATen.combinations_tlb) _input _r _with_replacement

-- | lstm_cell
-- >>> dtype &&& shape $ fst $ lstm_cell (ones :: Tensor 'D.Float '[2,2]) ((ones :: Tensor 'D.Float '[2,3]), (ones :: Tensor 'D.Float '[2,3])) (ones :: Tensor 'D.Float '[12,2]) (ones :: Tensor 'D.Float '[12,3]) (ones :: Tensor 'D.Float '[12]) (ones :: Tensor 'D.Float '[12])
-- (Float,[2,3])
lstm_cell :: forall dtype batchSize inputDim hiddenSize . 
  Tensor dtype '[batchSize, inputDim] 
  -> (Tensor dtype '[batchSize, hiddenSize], Tensor dtype '[batchSize, hiddenSize])
  -> Tensor dtype '[4 * hiddenSize, inputDim] 
  -> Tensor dtype '[4 * hiddenSize, hiddenSize] 
  -> Tensor dtype '[4 * hiddenSize] 
  -> Tensor dtype '[4 * hiddenSize]  
  -> (Tensor dtype '[batchSize, hiddenSize],Tensor dtype '[batchSize, hiddenSize])
lstm_cell _input (_cc, _hc) _w_ih _w_hh _b_ih _b_hh = unsafePerformIO $ (cast6 ATen.lstm_cell_tltttt) _input _hx _w_ih _w_hh _b_ih _b_hh
  where 
    _hx = [_cc, _hc] :: [Tensor dtype '[batchSize, hiddenSize]]


-- gru_cell :: Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape
-- gru_cell _input _hx _w_ih _w_hh _b_ih _b_hh = unsafePerformIO $ (cast6 ATen.gru_cell_tttttt) _input _hx _w_ih _w_hh _b_ih _b_hh

-- rnn_tanh_cell :: Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape
-- rnn_tanh_cell _input _hx _w_ih _w_hh _b_ih _b_hh = unsafePerformIO $ (cast6 ATen.rnn_tanh_cell_tttttt) _input _hx _w_ih _w_hh _b_ih _b_hh

-- rnn_relu_cell :: Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape
-- rnn_relu_cell _input _hx _w_ih _w_hh _b_ih _b_hh = unsafePerformIO $ (cast6 ATen.rnn_relu_cell_tttttt) _input _hx _w_ih _w_hh _b_ih _b_hh

--quantized_lstm :: Tensor dtype shape -> [Tensor dtype shape] -> [Tensor dtype shape] -> Bool -> Int -> Double -> Bool -> Bool -> Bool -> DType -> (Tensor dtype shape,Tensor dtype shape,Tensor dtype shape)
--quantized_lstm _input _hx _params _has_biases _num_layers _dropout _train _bidirectional _batch_first _dtype = unsafePerformIO $ (cast10 ATen.quantized_lstm_tllbldbbbs) _input _hx _params _has_biases _num_layers _dropout _train _bidirectional _batch_first _dtype

-- quantized_lstm_cell :: Tensor dtype shape -> [Tensor dtype shape] -> Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> Float -> Float -> Float -> Float -> (Tensor dtype shape,Tensor dtype shape)
-- quantized_lstm_cell _input _hx _w_ih _w_hh _b_ih _b_hh _packed_ih _packed_hh _col_offsets_ih _col_offsets_hh _scale_ih _scale_hh _zero_point_ih _zero_point_hh = unsafePerformIO $ (cast14 ATen.quantized_lstm_cell_tlttttttttssss) _input _hx _w_ih _w_hh _b_ih _b_hh _packed_ih _packed_hh _col_offsets_ih _col_offsets_hh _scale_ih _scale_hh _zero_point_ih _zero_point_hh

-- quantized_gru_cell :: Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> Float -> Float -> Float -> Float -> Tensor dtype shape
-- quantized_gru_cell _input _hx _w_ih _w_hh _b_ih _b_hh _packed_ih _packed_hh _col_offsets_ih _col_offsets_hh _scale_ih _scale_hh _zero_point_ih _zero_point_hh = unsafePerformIO $ (cast14 ATen.quantized_gru_cell_ttttttttttssss) _input _hx _w_ih _w_hh _b_ih _b_hh _packed_ih _packed_hh _col_offsets_ih _col_offsets_hh _scale_ih _scale_hh _zero_point_ih _zero_point_hh

-- quantized_rnn_relu_cell :: Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> Float -> Float -> Float -> Float -> Tensor dtype shape
-- quantized_rnn_relu_cell _input _hx _w_ih _w_hh _b_ih _b_hh _packed_ih _packed_hh _col_offsets_ih _col_offsets_hh _scale_ih _scale_hh _zero_point_ih _zero_point_hh = unsafePerformIO $ (cast14 ATen.quantized_rnn_relu_cell_ttttttttttssss) _input _hx _w_ih _w_hh _b_ih _b_hh _packed_ih _packed_hh _col_offsets_ih _col_offsets_hh _scale_ih _scale_hh _zero_point_ih _zero_point_hh

-- quantized_rnn_tanh_cell :: Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> Float -> Float -> Float -> Float -> Tensor dtype shape
-- quantized_rnn_tanh_cell _input _hx _w_ih _w_hh _b_ih _b_hh _packed_ih _packed_hh _col_offsets_ih _col_offsets_hh _scale_ih _scale_hh _zero_point_ih _zero_point_hh = unsafePerformIO $ (cast14 ATen.quantized_rnn_tanh_cell_ttttttttttssss) _input _hx _w_ih _w_hh _b_ih _b_hh _packed_ih _packed_hh _col_offsets_ih _col_offsets_hh _scale_ih _scale_hh _zero_point_ih _zero_point_hh

-- masked_scatter :: Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape
-- masked_scatter _input _mask _source = unsafePerformIO $ (cast3 ATen.masked_scatter_ttt) _input _mask _source

-- index_add :: Tensor dtype shape -> Int -> Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape
-- index_add _input _dim _index _source = unsafePerformIO $ (cast4 ATen.index_add_tltt) _input _dim _index _source

-- scatter_add :: Tensor dtype shape -> Int -> Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape
-- scatter_add _input _dim _index _src = unsafePerformIO $ (cast4 ATen.scatter_add_tltt) _input _dim _index _src

-- addbmm :: Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> Float -> Float -> Tensor dtype shape
-- addbmm _input _batch1 _batch2 _beta _alpha = unsafePerformIO $ (cast5 ATen.addbmm_tttss) _input _batch1 _batch2 _beta _alpha

-- cross :: Tensor dtype shape -> Tensor dtype shape -> Int -> Tensor dtype shape
-- cross _input _other _dim = unsafePerformIO $ (cast3 ATen.cross_ttl) _input _other _dim

type family MatrixOrMatrixBatch (shape :: [Nat]) :: [Nat] where
  MatrixOrMatrixBatch (n : m : '[])     = '[n, m]
  MatrixOrMatrixBatch (b : n : m : '[]) = '[b, n, m]
  MatrixOrMatrixBatch _                 = TypeError (Text "The input must be matrix or a batch of matrices.")

-- | triu
-- TODO: triu is not implemented for D.Bool
-- >>> t = ones @'D.Float @'[3, 4]
-- >>> dtype &&& shape &&& (\t' -> D.asValue (toDynamic t') :: [[Float]]) $ triu 0 t
-- (Float,([3,4],[[1.0,1.0,1.0,1.0],[0.0,1.0,1.0,1.0],[0.0,0.0,1.0,1.0]]))
-- >>> dtype &&& shape &&& (\t' -> D.asValue (toDynamic t') :: [[Float]]) $ triu 1 t
-- (Float,([3,4],[[0.0,1.0,1.0,1.0],[0.0,0.0,1.0,1.0],[0.0,0.0,0.0,1.0]]))
-- >>> dtype &&& shape &&& (\t' -> D.asValue (toDynamic t') :: [[Float]]) $ triu (-1) t
-- (Float,([3,4],[[1.0,1.0,1.0,1.0],[1.0,1.0,1.0,1.0],[0.0,1.0,1.0,1.0]]))
triu
  :: forall dtype shape
   . (shape ~ MatrixOrMatrixBatch shape)
  => Int
  -> Tensor dtype shape
  -> Tensor dtype shape
triu diagonal input = unsafePerformIO $ cast2 ATen.triu_tl input diagonal

-- | tril
-- TODO: tril is not implemented for D.Bool
-- >>> t = ones @'D.Float @'[3, 4]
-- >>> dtype &&& shape &&& (\t' -> D.asValue (toDynamic t') :: [[Float]]) $ tril 0 t
-- (Float,([3,4],[[1.0,0.0,0.0,0.0],[1.0,1.0,0.0,0.0],[1.0,1.0,1.0,0.0]]))
-- >>> dtype &&& shape &&& (\t' -> D.asValue (toDynamic t') :: [[Float]]) $ tril 1 t
-- (Float,([3,4],[[1.0,1.0,0.0,0.0],[1.0,1.0,1.0,0.0],[1.0,1.0,1.0,1.0]]))
-- >>> dtype &&& shape &&& (\t' -> D.asValue (toDynamic t') :: [[Float]]) $ tril (-1) t
-- (Float,([3,4],[[0.0,0.0,0.0,0.0],[1.0,0.0,0.0,0.0],[1.0,1.0,0.0,0.0]]))
tril
  :: forall dtype shape
   . (shape ~ MatrixOrMatrixBatch shape)
  => Int
  -> Tensor dtype shape
  -> Tensor dtype shape
tril diagonal input = unsafePerformIO $ cast2 ATen.tril_tl input diagonal


-- trace :: Tensor dtype shape -> Tensor dtype shape
-- trace _input = unsafePerformIO $ (cast1 ATen.trace_t) _input

-- take :: Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape
-- take _input _index = unsafePerformIO $ (cast2 ATen.take_tt) _input _index

-- index_select :: Tensor dtype shape -> Int -> Tensor dtype shape -> Tensor dtype shape
-- index_select _input _dim _index = unsafePerformIO $ (cast3 ATen.index_select_tlt) _input _dim _index

-- masked_select :: Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape
-- masked_select _input _mask = unsafePerformIO $ (cast2 ATen.masked_select_tt) _input _mask

-- nonzero :: Tensor dtype shape -> Tensor dtype shape
-- nonzero _input = unsafePerformIO $ (cast1 ATen.nonzero_t) _input

-- nonzero_numpy :: Tensor dtype shape -> [Tensor dtype shape]
-- nonzero_numpy _input = unsafePerformIO $ (cast1 ATen.nonzero_numpy_t) _input

-- gather :: Tensor dtype shape -> Int -> Tensor dtype shape -> Bool -> Tensor dtype shape
-- gather _input _dim _index _sparse_grad = unsafePerformIO $ (cast4 ATen.gather_tltb) _input _dim _index _sparse_grad

-- addcmul :: Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> Float -> Tensor dtype shape
-- addcmul _input _tensor1 _tensor2 _value = unsafePerformIO $ (cast4 ATen.addcmul_ttts) _input _tensor1 _tensor2 _value

-- addcdiv :: Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> Float -> Tensor dtype shape
-- addcdiv _input _tensor1 _tensor2 _value = unsafePerformIO $ (cast4 ATen.addcdiv_ttts) _input _tensor1 _tensor2 _value

-- lstsq :: Tensor dtype shape -> Tensor dtype shape -> (Tensor dtype shape,Tensor dtype shape)
-- lstsq _input _A = unsafePerformIO $ (cast2 ATen.lstsq_tt) _input _A

-- triangular_solve :: Tensor dtype shape -> Tensor dtype shape -> Bool -> Bool -> Bool -> (Tensor dtype shape,Tensor dtype shape)
-- triangular_solve _input _A _upper _transpose _unitriangular = unsafePerformIO $ (cast5 ATen.triangular_solve_ttbbb) _input _A _upper _transpose _unitriangular

-- qr :: Tensor dtype shape -> Bool -> (Tensor dtype shape,Tensor dtype shape)
-- qr _input _some = unsafePerformIO $ (cast2 ATen.qr_tb) _input _some

-- ormqr :: Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> Bool -> Bool -> Tensor dtype shape
-- ormqr _input _input2 _input3 _left _transpose = unsafePerformIO $ (cast5 ATen.ormqr_tttbb) _input _input2 _input3 _left _transpose

-- lu_solve :: Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape
-- lu_solve _input _LU_data _LU_pivots = unsafePerformIO $ (cast3 ATen.lu_solve_ttt) _input _LU_data _LU_pivots

-- |
-- >>> dtype &&& shape $ lgamma (ones :: Tensor 'D.Float '[3,2])
-- (Float,[3,2])
lgamma :: Tensor dtype shape -> Tensor dtype shape
lgamma _input = unsafePerformIO $ (cast1 ATen.lgamma_t) _input

-- |
-- >>> dtype &&& shape $ digamma (ones :: Tensor 'D.Float '[3,2])
-- (Float,[3,2])
digamma :: Tensor dtype shape -> Tensor dtype shape
digamma _input = unsafePerformIO $ (cast1 ATen.digamma_t) _input

polygamma :: Int -> Tensor dtype shape -> Tensor dtype shape
polygamma _n _input = unsafePerformIO $ (cast2 ATen.polygamma_lt) _n _input

-- |
-- >>> dtype &&& shape $ erfinv (ones :: Tensor 'D.Float '[3,2])
-- (Float,[3,2])
erfinv :: Tensor dtype shape -> Tensor dtype shape
erfinv _input = unsafePerformIO $ (cast1 ATen.erfinv_t) _input

-- dist :: Tensor dtype shape -> Tensor dtype shape -> Float -> Tensor dtype shape
-- dist _input _other _p = unsafePerformIO $ (cast3 ATen.dist_tts) _input _other _p

-- atan2 :: Tensor dtype shape -> Tensor dtype shape -> Tensor dtype shape
-- atan2 _input _other = unsafePerformIO $ (cast2 ATen.atan2_tt) _input _other

-- histc :: Tensor dtype shape -> Int -> Float -> Float -> Tensor dtype shape
-- histc _input _bins _min _max = unsafePerformIO $ (cast4 ATen.histc_tlss) _input _bins _min _max

-- |
-- >>> dtype &&& shape $ minAll (ones :: Tensor 'D.Float '[2,2])
-- (Float,[])
minAll :: Tensor dtype shape -> Tensor dtype '[]
minAll _input = unsafePerformIO $ (cast1 ATen.min_t) _input


type family DropValue (shape :: [Nat]) (i :: Nat) :: [Nat] where
    DropValue '[] _ = TypeError (Text "Can not find a element in the list.")
    DropValue (x: xs) 0 = xs
    DropValue (x: xs) i = x ': DropValue xs (i-1)

-- |
-- >>> dtype &&& shape $ (minDim @0 (ones :: Tensor 'D.Float '[3,4,5]) :: Tensor 'D.Float '[4,5])
-- (Float,[4,5])
-- >>> dtype &&& shape $ (minDim @1 (ones :: Tensor 'D.Float '[3,4,5]) :: Tensor 'D.Float '[3,5])
-- (Float,[3,5])
-- >>> dtype &&& shape $ (minDim @2 (ones :: Tensor 'D.Float '[3,4,5]) :: Tensor 'D.Float '[3,4])
-- (Float,[3,4])
minDim :: forall d dtype shape. (KnownNat d) => Tensor dtype shape -> Tensor dtype (DropValue shape d)
minDim _input = fst $ (unsafePerformIO $ (cast2 ATen.min_tl) _input (natValI @d) :: (Tensor dtype  (DropValue shape d), Tensor 'D.Int64 (DropValue shape d)))

maxAll :: Tensor dtype shape -> Tensor dtype '[]
maxAll _input = unsafePerformIO $ (cast1 ATen.max_t) _input

maxDim :: forall d dtype shape. (KnownNat d) => Tensor dtype shape -> Tensor dtype (DropValue shape d)
maxDim _input = fst $ (unsafePerformIO $ (cast2 ATen.max_tl) _input (natValI @d) :: (Tensor dtype  (DropValue shape d), Tensor 'D.Int64 (DropValue shape d)))

medianAll :: Tensor dtype shape -> Tensor dtype '[]
medianAll _input = unsafePerformIO $ (cast1 ATen.median_t) _input

medianDim :: forall d dtype shape. (KnownNat d) => Tensor dtype shape -> Tensor dtype (DropValue shape d)
medianDim _input = fst $ (unsafePerformIO $ (cast2 ATen.median_tl) _input (natValI @d) :: (Tensor dtype  (DropValue shape d), Tensor 'D.Int64 (DropValue shape d)))

-- | See https://pytorch.org/docs/stable/torch.html#torch.median.
-- >>> t = fromJust [[5, 1], [3, 2], [4, 1], [2, 7]] :: Tensor 'D.Float '[4, 2]
-- >>> median' @0 @KeepDim t :: (Tensor 'D.Float '[1, 2], Tensor 'D.Int64 '[1, 2])
-- (Tensor Float [1,2] [[ 3.0000   ,  1.0000   ]],Tensor Int64 [1,2] [[ 1,  0]])
median'
  :: forall dim keepOrDropDim dtype shape
   . (KnownNat dim, KnownKeepOrDropDim keepOrDropDim)
  => Tensor dtype shape
  -> (Tensor dtype (ConditionalDropDimension shape dim keepOrDropDim), Tensor 'D.Int64 (ConditionalDropDimension shape dim keepOrDropDim))
median' t = unsafePerformIO $ cast3 ATen.median_tlb t (natValI @dim) (keepOrDropDimVal @keepOrDropDim)

-- sort :: Tensor dtype shape -> Int -> Bool -> (Tensor dtype shape,Tensor dtype shape)
-- sort _input _dim _descending = unsafePerformIO $ (cast3 ATen.sort_tlb) _input _dim _descending

-- argsort :: Tensor dtype shape -> Int -> Bool -> Tensor dtype shape
-- argsort _input _dim _descending = unsafePerformIO $ (cast3 ATen.argsort_tlb) _input _dim _descending

-- topk :: Tensor dtype shape -> Int -> Int -> Bool -> Bool -> (Tensor dtype shape,Tensor dtype shape)
-- topk _input _k _dim _largest _sorted = unsafePerformIO $ (cast5 ATen.topk_tllbb) _input _k _dim _largest _sorted

-- renorm :: Tensor dtype shape -> Float -> Int -> Float -> Tensor dtype shape
-- renorm _input _p _dim _maxnorm = unsafePerformIO $ (cast4 ATen.renorm_tsls) _input _p _dim _maxnorm

-- equal :: Tensor dtype shape -> Tensor dtype shape -> Bool
-- equal _input _other = unsafePerformIO $ (cast2 ATen.equal_tt) _input _other

-- alias :: Tensor dtype shape -> Tensor dtype shape
-- alias _input = unsafePerformIO $ (cast1 ATen.alias_t) _input


-- |
-- >>> dtype &&& shape $ (l1_loss @ReduceNone (ones :: Tensor 'D.Float '[2,2]) (ones :: Tensor 'D.Float '[2,2]) :: Tensor 'D.Float '[2,2])
-- (Float,[2,2])
-- >>> dtype &&& shape $ (l1_loss @ReduceSum (ones :: Tensor 'D.Float '[2,2]) (ones :: Tensor 'D.Float '[2,2]) :: Tensor 'D.Float '[])
-- (Float,[])
l1_loss :: forall reduction dtype shape. (KnownReduction reduction) => Tensor dtype shape -> Tensor dtype shape -> Tensor dtype (ConditionalReduction shape reduction)
l1_loss _input _target = unsafePerformIO $ (cast3 ATen.l1_loss_ttl) _input _target (reductionVal @reduction)

-- multi_margin_loss :: Tensor dtype shape -> Tensor dtype shape -> Float -> Float -> Tensor dtype shape -> Int -> Tensor dtype shape
-- multi_margin_loss _input _target _p _margin _weight _reduction = unsafePerformIO $ (cast6 ATen.multi_margin_loss_ttsstl) _input _target _p _margin _weight _reduction

-- multilabel_margin_loss :: Tensor dtype shape -> Tensor dtype shape -> Int -> Tensor dtype shape
-- multilabel_margin_loss _input _target _reduction = unsafePerformIO $ (cast3 ATen.multilabel_margin_loss_ttl) _input _target _reduction

-- | The negative log likelihood loss.
-- See https://pytorch.org/docs/stable/nn.functional.html?highlight=nll_loss#torch.nn.functional.nll_loss.
-- >>> input <- randn @'D.Float @[3, 5]
-- >>> target = fromJust [1, 0, 4] :: Tensor 'D.Int64 '[3]
-- >>> weight = ones @'D.Float @'[5]
-- >>> dtype &&& shape $ nll_loss @ReduceNone @'D.Float @3 @5 @'[] (logSoftmax @1 input) target weight (-100)
-- (Float,[3])
-- >>> dtype &&& shape $ nll_loss @ReduceMean @'D.Float @3 @5 @'[] (logSoftmax @1 input) target weight (-100)
-- (Float,[])
-- >>> input <- randn @'D.Float @[3, 5, 2]
-- >>> target = fromJust [[1, 1], [0, 1], [4, 0]] :: Tensor 'D.Int64 '[3, 2]
-- >>> weight = ones @'D.Float @'[5]
-- >>> dtype &&& shape $ nll_loss @ReduceNone @'D.Float @3 @5 @'[2] (logSoftmax @1 input) target weight (-100)
-- (Float,[3,2])
-- >>> dtype &&& shape $ nll_loss @ReduceMean @'D.Float @3 @5 @'[2] (logSoftmax @1 input) target weight (-100)
-- (Float,[])
-- >>> input <- randn @'D.Float @[3, 5, 1, 2]
-- >>> target = fromJust [[[1, 1]], [[0, 1]], [[4, 0]]] :: Tensor 'D.Int64 '[3, 1, 2]
-- >>> weight = ones @'D.Float @'[5]
-- >>> dtype &&& shape $ nll_loss @ReduceNone @'D.Float @3 @5 @[1, 2] (logSoftmax @1 input) target weight (-100)
-- (Float,[3,1,2])
-- >>> dtype &&& shape $ nll_loss @ReduceMean @'D.Float @3 @5 @[1, 2] (logSoftmax @1 input) target weight (-100)
-- (Float,[])
-- >>> input <- randn @'D.Float @[3, 5, 2, 1, 2]
-- >>> target = fromJust [[[[1, 1]], [[0, 2]]], [[[0, 1]], [[1, 0]]], [[[4, 0]], [[1, 2]]]] :: Tensor 'D.Int64 '[3, 2, 1, 2]
-- >>> weight = ones @'D.Float @'[5]
-- >>> dtype &&& shape $ nll_loss @ReduceNone @'D.Float @3 @5 @[2, 1, 2] (logSoftmax @1 input) target weight (-100)
-- (Float,[3,2,1,2])
-- >>> dtype &&& shape $ nll_loss @ReduceMean @'D.Float @3 @5 @[2, 1, 2] (logSoftmax @1 input) target weight (-100)
-- (Float,[])
nll_loss
  :: forall reduction dtype n c ds
   . (KnownReduction reduction, KnownNat n, KnownNat c, KnownShape ds)
  => Tensor dtype (n ': c ': ds)
  -> Tensor 'D.Int64 (n ': ds)
  -> Tensor dtype '[c]
  -> Int
  -> Tensor dtype (ConditionalReduction (n ': ds) reduction)
nll_loss input target weight ignoreIndex = case shapeVal @ds of
  [] -> unsafePerformIO $ (cast5 ATen.nll_loss_tttll)
    input
    target
    weight
    (reductionVal @reduction)
    ignoreIndex
  [_h, _w] -> unsafePerformIO $ (cast5 ATen.nll_loss2d_tttll)
    input
    target
    weight
    (reductionVal @reduction)
    ignoreIndex
  h : t -> case reductionVal @reduction of
    0 -> UnsafeMkTensor . D.reshape out $ (natValI @n) : h : t
    _ -> UnsafeMkTensor out
   where
    t'      = [1, foldl (*) h t]
    input'  = D.reshape (toDynamic input) (natValI @n : natValI @c : t')
    target' = D.reshape (toDynamic target) (natValI @n : t')
    out     = unsafePerformIO $ (cast5 ATen.nll_loss2d_tttll)
      input'
      target'
      weight
      (reductionVal @reduction)
      ignoreIndex

-- |
-- >>> dtype &&& shape $ smooth_l1_loss @ReduceNone (ones :: Tensor 'D.Float '[2,2]) (ones :: Tensor 'D.Float '[2,2])
-- (Float,[2,2])
-- >>> dtype &&& shape $ smooth_l1_loss @ReduceSum (ones :: Tensor 'D.Float '[2,2]) (ones :: Tensor 'D.Float '[2,2])
-- (Float,[])
smooth_l1_loss :: forall reduction dtype shape. (KnownReduction reduction) => Tensor dtype shape -> Tensor dtype shape -> Tensor dtype (ConditionalReduction shape reduction)
smooth_l1_loss _input _target = unsafePerformIO $ (cast3 ATen.smooth_l1_loss_ttl) _input _target (reductionVal @reduction)

-- |
-- >>> dtype &&& shape $ soft_margin_loss @ReduceNone (ones :: Tensor 'D.Float '[2,2]) (ones :: Tensor 'D.Float '[2,2])
-- (Float,[2,2])
-- >>> dtype &&& shape $ soft_margin_loss @ReduceSum (ones :: Tensor 'D.Float '[2,2]) (ones :: Tensor 'D.Float '[2,2])
-- (Float,[])
soft_margin_loss :: forall reduction dtype shape. (KnownReduction reduction) => Tensor dtype shape -> Tensor dtype shape -> Tensor dtype (ConditionalReduction shape reduction)
soft_margin_loss _input _target = unsafePerformIO $ (cast3 ATen.soft_margin_loss_ttl) _input _target (reductionVal @reduction)

-- |
-- >>> dtype &&& shape $ elu (ones :: Tensor 'D.Float '[3,2]) 0.1 0.1 0.3
-- (Float,[3,2])
elu :: Tensor dtype shape -> Float -> Float -> Float -> Tensor dtype shape
elu _input _alpha _scale _input_scale = unsafePerformIO $ (cast4 ATen.elu_tsss) _input _alpha _scale _input_scale

-- |
-- -- >>> dtype &&& shape $ glu (ones :: Tensor 'D.Float '[3,2]) 1
-- -- (Float,[3,1])
-- -- >>> dtype &&& shape $ glu (ones :: Tensor 'D.Float '[3,2]) 3
-- -- (Float,[3,2])
-- glu :: Tensor dtype shape -> Int -> Tensor dtype shape
-- glu _input _dim = unsafePerformIO $ (cast2 ATen.glu_tl) _input _dim

-- |
-- >>> dtype &&& shape $ hardtanh (ones :: Tensor 'D.Float '[3,2]) 0 1
-- (Float,[3,2])
hardtanh :: Tensor dtype shape -> Float -> Float -> Tensor dtype shape
hardtanh _input _min_val _max_val = unsafePerformIO $ (cast3 ATen.hardtanh_tss) _input _min_val _max_val

-- leaky_relu :: Tensor dtype shape -> Float -> Tensor dtype shape
-- leaky_relu _input _negative_slope = unsafePerformIO $ (cast2 ATen.leaky_relu_ts) _input _negative_slope

-- |
-- >>> dtype &&& shape $ logSigmoid (ones :: Tensor 'D.Float '[3,2])
-- (Float,[3,2])
logSigmoid :: Tensor dtype shape -> Tensor dtype shape
logSigmoid _input = unsafePerformIO $ (cast1 ATen.log_sigmoid_t) _input

-- |
-- -- >>> dtype &&& shape $ softplus (ones :: Tensor 'D.Float '[3,2])
-- -- (Float,[3,2])
-- softplus :: Tensor dtype shape -> Float -> Float -> Tensor dtype shape
-- softplus _input _beta _threshold = unsafePerformIO $ (cast3 ATen.softplus_tss) _input _beta _threshold

-- |
-- >>> dtype &&& shape $ softshrink (ones :: Tensor 'D.Float '[3,2]) 0.2
-- (Float,[3,2])
softshrink :: Tensor dtype shape -> Float -> Tensor dtype shape
softshrink _input _lambd = unsafePerformIO $ (cast2 ATen.softshrink_ts) _input _lambd

-- |
-- >>> t = adaptiveAvgPool2d @'(8,16) (ones::Tensor 'D.Float '[1,3,16,32])
-- >>> shape t
-- [1,3,8,16]
-- >>> :t t
-- t :: Tensor 'D.Float '[1, 3, 8, 16]
adaptiveAvgPool2d
  :: forall outputSize channelSize inputSize0 inputSize1 batchSize dtype
   . (All KnownNat [channelSize, inputSize0, inputSize1, batchSize, Fst outputSize, Snd outputSize])
  => Tensor dtype '[batchSize, channelSize, inputSize0, inputSize1]
  -> Tensor dtype '[batchSize, channelSize, Fst outputSize, Snd outputSize]
adaptiveAvgPool2d _input = unsafePerformIO $ (cast2 ATen.adaptive_avg_pool2d_tl)
  _input
  ([natValI @(Fst outputSize), natValI @(Snd outputSize)] :: [Int])

-- |
-- >>> t = mkldnnAdaptiveAvgPool2d @'(8,16) (toMKLDNN (ones::Tensor 'D.Float '[1,3,16,32]))
-- >>> shape t
-- [1,3,8,16]
-- >>> :t t
-- t :: Tensor 'D.Float '[1, 3, 8, 16]
mkldnnAdaptiveAvgPool2d
  :: forall outputSize channelSize inputSize0 inputSize1 batchSize dtype
   . (All KnownNat [channelSize, inputSize0, inputSize1, batchSize, Fst outputSize, Snd outputSize])
  => Tensor dtype '[batchSize, channelSize, inputSize0, inputSize1]
  -> Tensor dtype '[batchSize, channelSize, Fst outputSize, Snd outputSize]
mkldnnAdaptiveAvgPool2d _input = unsafePerformIO $ (cast2 ATen.adaptive_avg_pool2d_tl)
  _input
  ([natValI @(Fst outputSize), natValI @(Snd outputSize)] :: [Int])

-- |
-- >>> t = adaptiveAvgPool3d @'(8,16,2) (ones::Tensor 'D.Float '[1,3,16,32,4])
-- >>> shape t
-- [1,3,8,16,2]
-- >>> :t t
-- t :: Tensor 'D.Float '[1, 3, 8, 16, 2]
adaptiveAvgPool3d
  :: forall outputSize channelSize inputSize0 inputSize1 inputSize2 batchSize dtype
   . (All KnownNat [channelSize,
                    inputSize0, inputSize1, inputSize2,
                    batchSize,
                    Fst3 outputSize, Snd3 outputSize, Trd3 outputSize])
  => Tensor dtype '[batchSize, channelSize, inputSize0, inputSize1, inputSize2]
  -> Tensor dtype '[batchSize, channelSize, Fst3 outputSize, Snd3 outputSize, Trd3 outputSize]
adaptiveAvgPool3d _input = unsafePerformIO $ (cast2 ATen.adaptive_avg_pool3d_tl)
  _input
  ([natValI @(Fst3 outputSize), natValI @(Snd3 outputSize), natValI @(Trd3 outputSize)] :: [Int])


-- |
-- >>> t = adaptiveMaxPool2d @'(8,16) (ones::Tensor 'D.Float '[1,3,16,32])
-- >>> shape t
-- [1,3,8,16]
-- >>> :t t
-- t :: Tensor 'D.Float '[1, 3, 8, 16]
adaptiveMaxPool2d
  :: forall outputSize channelSize inputSize0 inputSize1 batchSize dtype
   . (All KnownNat [channelSize, inputSize0, inputSize1, batchSize, Fst outputSize, Snd outputSize])
  => Tensor dtype '[batchSize, channelSize, inputSize0, inputSize1]
  -> Tensor dtype '[batchSize, channelSize, Fst outputSize, Snd outputSize]
adaptiveMaxPool2d _input = fst $ (unsafePerformIO $ (cast2 ATen.adaptive_max_pool2d_tl)
  _input
  ([natValI @(Fst outputSize), natValI @(Snd outputSize)] :: [Int])
  :: (Tensor dtype '[batchSize, channelSize, Fst outputSize, Snd outputSize],
      Tensor 'D.Int64 '[batchSize, channelSize, Fst outputSize, Snd outputSize]))

-- |
-- >>> t = adaptiveMaxPool3d @'(8,16,2) (ones::Tensor 'D.Float '[1,3,16,32,4])
-- >>> shape t
-- [1,3,8,16,2]
-- >>> :t t
-- t :: Tensor 'D.Float '[1, 3, 8, 16, 2]
adaptiveMaxPool3d
  :: forall outputSize channelSize inputSize0 inputSize1 inputSize2 batchSize dtype
   . (All KnownNat [channelSize,
                    inputSize0, inputSize1, inputSize2,
                    batchSize,
                    Fst3 outputSize, Snd3 outputSize, Trd3 outputSize])
  => Tensor dtype '[batchSize, channelSize, inputSize0, inputSize1, inputSize2]
  -> Tensor dtype '[batchSize, channelSize, Fst3 outputSize, Snd3 outputSize, Trd3 outputSize]
adaptiveMaxPool3d _input = fst $ (unsafePerformIO $ (cast2 ATen.adaptive_max_pool3d_tl)
  _input
  ([natValI @(Fst3 outputSize), natValI @(Snd3 outputSize), natValI @(Trd3 outputSize)] :: [Int])
  :: (Tensor dtype '[batchSize, channelSize, Fst3 outputSize, Snd3 outputSize, Trd3 outputSize]
     ,Tensor 'D.Int64 '[batchSize, channelSize, Fst3 outputSize, Snd3 outputSize, Trd3 outputSize]))

-- |
-- >>> t = avgPool2d @'(1,1) @'(1,1) @'(0,0) (ones::Tensor 'D.Float '[1,3,4,5])
-- >>> shape t
-- [1,3,4,5]
-- >>> :t t
-- t :: Tensor 'D.Float '[1, 3, 4, 5]
avgPool2d
  :: forall kernelSize stride padding channelSize inputSize0 inputSize1 batchSize dtype outputSize0 outputSize1.
     (All KnownNat [Fst kernelSize, Snd kernelSize,
                    Fst stride, Snd stride,
                    Fst padding, Snd padding,
                    channelSize,
                    inputSize0, inputSize1,
                    batchSize]
     , ConvSideCheck inputSize0 (Fst kernelSize) (Fst stride) (Fst padding) outputSize0
     , ConvSideCheck inputSize1 (Snd kernelSize) (Snd stride) (Snd padding) outputSize1
     )
  => Tensor dtype '[batchSize, channelSize, inputSize0, inputSize1]
  -> Tensor dtype '[batchSize, channelSize, outputSize0, outputSize1]
avgPool2d _input = unsafePerformIO $ (cast7 ATen.avg_pool2d_tlllbbl)
  _input
  ([natValI @(Fst kernelSize), natValI @(Snd kernelSize)] :: [Int])
  ([natValI @(Fst stride), natValI @(Snd stride)] :: [Int])
  ([natValI @(Fst padding), natValI @(Snd padding)] :: [Int])
  False
  True
  (1 :: Int)

-- |
-- >>> t = avgPool3d @'(1,1,1) @'(1,1,1) @'(0,0,0) (ones::Tensor 'D.Float '[1,3,4,5,6])
-- >>> shape t
-- [1,3,4,5,6]
-- >>> :t t
-- t :: Tensor 'D.Float '[1, 3, 4, 5, 6]
avgPool3d
  :: forall kernelSize stride padding channelSize
            inputSize0 inputSize1 inputSize2
            batchSize dtype
            outputSize0 outputSize1 outputSize2.
     (All KnownNat [Fst3 kernelSize, Snd3 kernelSize, Trd3 kernelSize,
                    Fst3 stride, Snd3 stride, Trd3 stride,
                    Fst3 padding, Snd3 padding, Trd3 padding,
                    channelSize,
                    inputSize0, inputSize1, inputSize2,
                    batchSize]
     , ConvSideCheck inputSize0 (Fst3 kernelSize) (Fst3 stride) (Fst3 padding) outputSize0
     , ConvSideCheck inputSize1 (Snd3 kernelSize) (Snd3 stride) (Snd3 padding) outputSize1
     , ConvSideCheck inputSize2 (Trd3 kernelSize) (Trd3 stride) (Trd3 padding) outputSize2)
  => Tensor dtype '[batchSize, channelSize, inputSize0, inputSize1, inputSize2]
  -> Tensor dtype '[batchSize, channelSize, outputSize0, outputSize1, outputSize2]
avgPool3d _input =
  unsafePerformIO $
  (cast7 ATen.avg_pool3d_tlllbbl)
    _input
    ([natValI @(Fst3 kernelSize), natValI @(Snd3 kernelSize), natValI @(Trd3 kernelSize)] :: [Int])
    ([natValI @(Fst3 stride), natValI @(Snd3 stride), natValI @(Trd3 stride)] :: [Int])
    ([natValI @(Fst3 padding), natValI @(Snd3 padding), natValI @(Trd3 padding)] :: [Int])
    False
    True
    (1::Int)

-- fractional_max_pool2d :: Tensor dtype shape -> (Int,Int) -> (Int,Int) -> Tensor dtype shape -> (Tensor dtype shape,Tensor dtype shape)
-- fractional_max_pool2d _input _kernel_size _output_size _random_samples = unsafePerformIO $ (cast4 ATen.fractional_max_pool2d_tllt) _input _kernel_size _output_size _random_samples

-- fractional_max_pool3d :: Tensor dtype shape -> (Int,Int,Int) -> (Int,Int,Int) -> Tensor dtype shape -> (Tensor dtype shape,Tensor dtype shape)
-- fractional_max_pool3d _input _kernel_size _output_size _random_samples = unsafePerformIO $ (cast4 ATen.fractional_max_pool3d_tllt) _input _kernel_size _output_size _random_samples

-- max_pool2d_with_indices :: Tensor dtype shape -> (Int,Int) -> (Int,Int) -> (Int,Int) -> (Int,Int) -> Bool -> (Tensor dtype shape,Tensor dtype shape)
-- max_pool2d_with_indices _input _kernel_size _stride _padding _dilation _ceil_mode = unsafePerformIO $ (cast6 ATen.max_pool2d_with_indices_tllllb) _input _kernel_size _stride _padding _dilation _ceil_mode

-- max_pool3d_with_indices :: Tensor dtype shape -> (Int,Int,Int) -> (Int,Int,Int) -> (Int,Int,Int) -> (Int,Int,Int) -> Bool -> (Tensor dtype shape,Tensor dtype shape)
-- max_pool3d_with_indices _input _kernel_size _stride _padding _dilation _ceil_mode = unsafePerformIO $ (cast6 ATen.max_pool3d_with_indices_tllllb) _input _kernel_size _stride _padding _dilation _ceil_mode

-- max_unpool2d :: Tensor dtype shape -> Tensor dtype shape -> (Int,Int) -> Tensor dtype shape
-- max_unpool2d _input _indices _output_size = unsafePerformIO $ (cast3 ATen.max_unpool2d_ttl) _input _indices _output_size

-- max_unpool3d :: Tensor dtype shape -> Tensor dtype shape -> (Int,Int,Int) -> (Int,Int,Int) -> (Int,Int,Int) -> Tensor dtype shape
-- max_unpool3d _input _indices _output_size _stride _padding = unsafePerformIO $ (cast5 ATen.max_unpool3d_ttlll) _input _indices _output_size _stride _padding

-- reflection_pad1d :: Tensor dtype shape -> (Int,Int) -> Tensor dtype shape
-- reflection_pad1d _input _padding = unsafePerformIO $ (cast2 ATen.reflection_pad1d_tl) _input _padding

-- reflection_pad2d :: Tensor dtype shape -> (Int,Int,Int,Int) -> Tensor dtype shape
-- reflection_pad2d _input _padding = unsafePerformIO $ (cast2 ATen.reflection_pad2d_tl) _input _padding

-- replication_pad1d :: Tensor dtype shape -> (Int,Int) -> Tensor dtype shape
-- replication_pad1d _input _padding = unsafePerformIO $ (cast2 ATen.replication_pad1d_tl) _input _padding

-- replication_pad2d :: Tensor dtype shape -> (Int,Int,Int,Int) -> Tensor dtype shape
-- replication_pad2d _input _padding = unsafePerformIO $ (cast2 ATen.replication_pad2d_tl) _input _padding

-- replication_pad3d :: Tensor dtype shape -> (Int,Int,Int,Int,Int,Int) -> Tensor dtype shape
-- replication_pad3d _input _padding = unsafePerformIO $ (cast2 ATen.replication_pad3d_tl) _input _padding

-- upsample_linear1d :: Tensor dtype shape -> Int -> Bool -> Tensor dtype shape
-- upsample_linear1d _input _output_size _align_corners = unsafePerformIO $ (cast3 ATen.upsample_linear1d_tlb) _input _output_size _align_corners

-- upsample_bilinear2d :: Tensor dtype shape -> (Int,Int) -> Bool -> Tensor dtype shape
-- upsample_bilinear2d _input _output_size _align_corners = unsafePerformIO $ (cast3 ATen.upsample_bilinear2d_tlb) _input _output_size _align_corners

-- upsample_bicubic2d :: Tensor dtype shape -> (Int,Int) -> Bool -> Tensor dtype shape
-- upsample_bicubic2d _input _output_size _align_corners = unsafePerformIO $ (cast3 ATen.upsample_bicubic2d_tlb) _input _output_size _align_corners

-- upsample_trilinear3d :: Tensor dtype shape -> (Int,Int,Int) -> Bool -> Tensor dtype shape
-- upsample_trilinear3d _input _output_size _align_corners = unsafePerformIO $ (cast3 ATen.upsample_trilinear3d_tlb) _input _output_size _align_corners

-- upsample_nearest1d :: Tensor dtype shape -> Int -> Tensor dtype shape
-- upsample_nearest1d _input _output_size = unsafePerformIO $ (cast2 ATen.upsample_nearest1d_tl) _input _output_size

-- upsample_nearest2d :: Tensor dtype shape -> (Int,Int) -> Tensor dtype shape
-- upsample_nearest2d _input _output_size = unsafePerformIO $ (cast2 ATen.upsample_nearest2d_tl) _input _output_size

-- upsample_nearest3d :: Tensor dtype shape -> (Int,Int,Int) -> Tensor dtype shape
-- upsample_nearest3d _input _output_size = unsafePerformIO $ (cast2 ATen.upsample_nearest3d_tl) _input _output_size

-- conv_dilated2d :: Tensor dtype shape -> Tensor dtype shape -> (Int,Int) -> Tensor dtype shape -> (Int,Int) -> (Int,Int) -> (Int,Int) -> Tensor dtype shape
-- conv_dilated2d _input _weight _kernel_size _bias _stride _padding _dilation = unsafePerformIO $ (cast7 ATen.conv_dilated2d_ttltlll) _input _weight _kernel_size _bias _stride _padding _dilation

-- conv_dilated3d :: Tensor dtype shape -> Tensor dtype shape -> (Int,Int,Int) -> Tensor dtype shape -> (Int,Int,Int) -> (Int,Int,Int) -> (Int,Int,Int) -> Tensor dtype shape
-- conv_dilated3d _input _weight _kernel_size _bias _stride _padding _dilation = unsafePerformIO $ (cast7 ATen.conv_dilated3d_ttltlll) _input _weight _kernel_size _bias _stride _padding _dilation

-- col2im :: Tensor dtype shape -> (Int,Int) -> (Int,Int) -> (Int,Int) -> (Int,Int) -> (Int,Int) -> Tensor dtype shape
-- col2im _input _output_size _kernel_size _dilation _padding _stride = unsafePerformIO $ (cast6 ATen.col2im_tlllll) _input _output_size _kernel_size _dilation _padding _stride

-- im2col :: Tensor dtype shape -> (Int,Int) -> (Int,Int) -> (Int,Int) -> (Int,Int) -> Tensor dtype shape
-- im2col _input _kernel_size _dilation _padding _stride = unsafePerformIO $ (cast5 ATen.im2col_tllll) _input _kernel_size _dilation _padding _stride
