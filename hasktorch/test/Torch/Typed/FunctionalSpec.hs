{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE UndecidableSuperClasses #-}
{-# LANGUAGE NoStarIsType #-}
{-# OPTIONS_GHC -Wno-partial-type-signatures #-}
{-# OPTIONS_GHC -freduction-depth=0 #-}

module Torch.Typed.FunctionalSpec
  ( Torch.Typed.FunctionalSpec.spec,
  )
where

import Control.Exception.Safe
import Data.Proxy
import Data.Reflection
import Foreign.Storable
import GHC.TypeLits
import Test.Hspec
import Test.QuickCheck
import qualified Torch.DType as D
import qualified Torch.Device as D
import qualified Torch.Functional as D
import Torch.HList
import qualified Torch.Internal.Managed.Type.Context as LibTorch
import qualified Torch.Scalar as D
import qualified Torch.Tensor as D
import qualified Torch.TensorFactories as D
import qualified Torch.TensorOptions as D
import Torch.Typed.Aux
import Torch.Typed.AuxSpec
import Torch.Typed.Factories
import Torch.Typed.Functional
import Torch.Typed.Tensor
import Prelude hiding
  ( abs,
    acos,
    acosh,
    all,
    any,
    asin,
    asinh,
    atan,
    atanh,
    cos,
    cosh,
    exp,
    floor,
    log,
    max,
    min,
    round,
    sin,
    sinh,
    sqrt,
    tan,
    tanh,
  )

data UnaryAllDTypesSpec
  = SignSpec
  | OnesLikeSpec
  | ZerosLikeSpec

instance
  ( TensorOptions shape dtype device
  ) =>
  Apply' UnaryAllDTypesSpec ((Proxy device, (Proxy dtype, Proxy shape)), IO ()) (IO ())
  where
  apply' SignSpec (_, agg) = agg >> do
    let t = sign (ones @shape @dtype @device)
    checkDynamicTensorAttributes t
  apply' OnesLikeSpec (_, agg) = agg >> do
    let t = onesLike (ones @shape @dtype @device)
    checkDynamicTensorAttributes t
  apply' ZerosLikeSpec (_, agg) = agg >> do
    let t = zerosLike (ones @shape @dtype @device)
    checkDynamicTensorAttributes t

data UnaryStandardDTypesSpec
  = AbsSpec

instance
  ( TensorOptions shape dtype device,
    DTypeIsNotHalf device dtype,
    DTypeIsNotBool device dtype
  ) =>
  Apply' UnaryStandardDTypesSpec ((Proxy device, (Proxy dtype, Proxy shape)), IO ()) (IO ())
  where
  apply' AbsSpec (_, agg) = agg >> do
    let t = abs (ones @shape @dtype @device)
    checkDynamicTensorAttributes t

data UnaryStandardFloatingPointDTypesSpec
  = FracSpec
  | CeilSpec
  | FloorSpec
  | RoundSpec
  | TruncSpec
  | ErfSpec
  | ErfcSpec
  | ErfinvSpec
  | ExpSpec
  | Expm1Spec
  | LogSpec
  | Log1pSpec
  | Log2Spec
  | Log10Spec
  | LgammaSpec
  | DigammaSpec
  | ReluSpec
  | SeluSpec
  | SigmoidSpec
  | LogSigmoidSpec
  | SinSpec
  | SinhSpec
  | AsinSpec
  | CosSpec
  | CoshSpec
  | AcosSpec
  | TanSpec
  | TanhSpec
  | AtanSpec
  | SqrtSpec
  | RsqrtSpec
  | RandLikeSpec
  | RandnLikeSpec

instance
  ( TensorOptions shape dtype device,
    StandardFloatingPointDTypeValidation device dtype
  ) =>
  Apply' UnaryStandardFloatingPointDTypesSpec ((Proxy device, (Proxy dtype, Proxy shape)), IO ()) (IO ())
  where
  apply' FracSpec (_, agg) = agg >> do
    let t = frac (ones @shape @dtype @device)
    checkDynamicTensorAttributes t
  apply' CeilSpec (_, agg) = agg >> do
    let t = ceil (ones @shape @dtype @device)
    checkDynamicTensorAttributes t
  apply' FloorSpec (_, agg) = agg >> do
    let t = floor (ones @shape @dtype @device)
    checkDynamicTensorAttributes t
  apply' TruncSpec (_, agg) = agg >> do
    let t = trunc (ones @shape @dtype @device)
    checkDynamicTensorAttributes t
  apply' ErfSpec (_, agg) = agg >> do
    let t = erf (ones @shape @dtype @device)
    checkDynamicTensorAttributes t
  apply' ErfcSpec (_, agg) = agg >> do
    let t = erfc (ones @shape @dtype @device)
    checkDynamicTensorAttributes t
  apply' ErfinvSpec (_, agg) = agg >> do
    let t = erfinv (ones @shape @dtype @device)
    checkDynamicTensorAttributes t
  apply' ExpSpec (_, agg) = agg >> do
    let t = exp (ones @shape @dtype @device)
    checkDynamicTensorAttributes t
  apply' Expm1Spec (_, agg) = agg >> do
    let t = expm1 (ones @shape @dtype @device)
    checkDynamicTensorAttributes t
  apply' LogSpec (_, agg) = agg >> do
    let t = log (ones @shape @dtype @device)
    checkDynamicTensorAttributes t
  apply' Log1pSpec (_, agg) = agg >> do
    let t = log1p (ones @shape @dtype @device)
    checkDynamicTensorAttributes t
  apply' Log2Spec (_, agg) = agg >> do
    let t = log2 (ones @shape @dtype @device)
    checkDynamicTensorAttributes t
  apply' Log10Spec (_, agg) = agg >> do
    let t = log10 (ones @shape @dtype @device)
    checkDynamicTensorAttributes t
  apply' LgammaSpec (_, agg) = agg >> do
    let t = lgamma (ones @shape @dtype @device)
    checkDynamicTensorAttributes t
  apply' DigammaSpec (_, agg) = agg >> do
    let t = digamma (ones @shape @dtype @device)
    checkDynamicTensorAttributes t
  apply' ReluSpec (_, agg) = agg >> do
    let t = relu (ones @shape @dtype @device)
    checkDynamicTensorAttributes t
  apply' SeluSpec (_, agg) = agg >> do
    let t = selu (ones @shape @dtype @device)
    checkDynamicTensorAttributes t
  apply' SigmoidSpec (_, agg) = agg >> do
    let t = sigmoid (ones @shape @dtype @device)
    checkDynamicTensorAttributes t
  apply' LogSigmoidSpec (_, agg) = agg >> do
    let t = logSigmoid (ones @shape @dtype @device)
    checkDynamicTensorAttributes t
  apply' SinSpec (_, agg) = agg >> do
    let t = sin (ones @shape @dtype @device)
    checkDynamicTensorAttributes t
  apply' SinhSpec (_, agg) = agg >> do
    let t = sinh (ones @shape @dtype @device)
    checkDynamicTensorAttributes t
  apply' AsinSpec (_, agg) = agg >> do
    let t = asin (ones @shape @dtype @device)
    checkDynamicTensorAttributes t
  apply' CosSpec (_, agg) = agg >> do
    let t = cos (ones @shape @dtype @device)
    checkDynamicTensorAttributes t
  apply' CoshSpec (_, agg) = agg >> do
    let t = cosh (ones @shape @dtype @device)
    checkDynamicTensorAttributes t
  apply' AcosSpec (_, agg) = agg >> do
    let t = acos (ones @shape @dtype @device)
    checkDynamicTensorAttributes t
  apply' TanSpec (_, agg) = agg >> do
    let t = tan (ones @shape @dtype @device)
    checkDynamicTensorAttributes t
  apply' TanhSpec (_, agg) = agg >> do
    let t = tanh (ones @shape @dtype @device)
    checkDynamicTensorAttributes t
  apply' AtanSpec (_, agg) = agg >> do
    let t = atan (ones @shape @dtype @device)
    checkDynamicTensorAttributes t
  apply' SqrtSpec (_, agg) = agg >> do
    let t = sqrt (ones @shape @dtype @device)
    checkDynamicTensorAttributes t
  apply' RsqrtSpec (_, agg) = agg >> do
    let t = rsqrt (ones @shape @dtype @device)
    checkDynamicTensorAttributes t
  apply' RandLikeSpec (_, agg) = agg >> do
    t <- randLike (ones @shape @dtype @device)
    checkDynamicTensorAttributes t
  apply' RandnLikeSpec (_, agg) = agg >> do
    t <- randnLike (ones @shape @dtype @device)
    checkDynamicTensorAttributes t

data MishSpec = MishSpec

instance
  ( TensorOptions shape dtype device,
    StandardFloatingPointDTypeValidation device dtype,
    BasicArithmeticDTypeIsValid device dtype,
    shape ~ Broadcast shape shape
  ) =>
  Apply' MishSpec ((Proxy device, (Proxy dtype, Proxy shape)), IO ()) (IO ())
  where
  apply' MishSpec (_, agg) = agg >> do
    let t = mish (ones @shape @dtype @device)
    checkDynamicTensorAttributes t

data GeluSpec = GeluSpec

instance
  ( TensorOptions shape dtype device,
    GeluDTypeIsValid device dtype
  ) =>
  Apply' GeluSpec ((Proxy device, (Proxy dtype, Proxy shape)), IO ()) (IO ())
  where
  apply' GeluSpec (_, agg) = agg >> do
    let t = gelu (ones @shape @dtype @device)
    checkDynamicTensorAttributes t

data LeakyReluSpec = LeakyReluSpec

instance
  ( TensorOptions shape dtype device,
    D.Scalar a,
    StandardFloatingPointDTypeValidation device dtype
  ) =>
  Apply' LeakyReluSpec ((Proxy device, (a, (Proxy dtype, Proxy shape))), IO ()) (IO ())
  where
  apply' LeakyReluSpec ((_, (negativeSlope, _)), agg) = agg >> do
    let t = leakyRelu negativeSlope (ones @shape @dtype @device)
    checkDynamicTensorAttributes t

data ELUSpec = ELUSpec

instance
  ( TensorOptions shape dtype device,
    D.Scalar a,
    StandardFloatingPointDTypeValidation device dtype
  ) =>
  Apply' ELUSpec ((Proxy device, ((a, (a, a)), (Proxy dtype, Proxy shape))), IO ()) (IO ())
  where
  apply' ELUSpec ((_, ((alpha, (scale, inputScale)), _)), agg) = agg >> do
    let t = elu alpha scale inputScale (ones @shape @dtype @device)
    checkDynamicTensorAttributes t

data ToDTypeSpec = ToDTypeSpec

instance
  ( TensorOptions shape dtype device,
    TensorOptions shape dtype' device,
    KnownDType dtype'
  ) =>
  Apply' ToDTypeSpec ((Proxy device, ((Proxy dtype, Proxy dtype'), Proxy shape)), IO ()) (IO ())
  where
  apply' ToDTypeSpec (_, agg) = agg >> do
    let t = ones @shape @dtype @device
        t' = Torch.Typed.Tensor.toDType @dtype' t
    checkDynamicTensorAttributes t'

data SumAllSpec = SumAllSpec

instance
  ( TensorOptions shape dtype device,
    SumDTypeIsValid device dtype,
    KnownDType (SumDType dtype),
    KnownDevice device
  ) =>
  Apply' SumAllSpec ((Proxy device, (Proxy dtype, Proxy shape)), IO ()) (IO ())
  where
  apply' SumAllSpec (_, agg) = agg >> do
    let t = ones @shape @dtype @device
        t' = sumAll t
    checkDynamicTensorAttributes t'

data SumDimSpec = SumDimSpec

instance
  ( TensorOptions shape dtype device,
    TensorOptions shape' dtype' device,
    KnownNat d,
    shape' ~ DropValue shape d,
    dtype' ~ SumDType dtype,
    SumDTypeIsValid device dtype
  ) =>
  Apply' SumDimSpec ((Proxy d, (Proxy device, (Proxy dtype, Proxy shape))), IO ()) (IO ())
  where
  apply' SumDimSpec (_, agg) = agg >> do
    let t = ones @shape @dtype @device
        t' = sumDim @d t
    checkDynamicTensorAttributes t'

data AggregationSpec
  = MinSpec
  | MaxSpec
  | MedianSpec
  | MeanSpec

instance
  ( TensorOptions shape dtype device,
    KnownDType dtype,
    KnownDevice device,
    AggregationDTypeIsValid device dtype,
    AllDimsPositive shape
  ) =>
  Apply' AggregationSpec ((Proxy device, (Proxy dtype, Proxy shape)), IO ()) (IO ())
  where
  apply' MinSpec (_, agg) = agg >> do
    let t = ones @shape @dtype @device
        t' = min t
    checkDynamicTensorAttributes t'
  apply' MaxSpec (_, agg) = agg >> do
    let t = ones @shape @dtype @device
        t' = min t
    checkDynamicTensorAttributes t'
  apply' MedianSpec (_, agg) = agg >> do
    let t = ones @shape @dtype @device
        t' = min t
    checkDynamicTensorAttributes t'
  apply' MeanSpec (_, agg) = agg >> do
    let t = ones @shape @dtype @device
        t' = min t
    checkDynamicTensorAttributes t'

data SqueezeAllSpec = SqueezeAllSpec

instance
  ( TensorOptions shape dtype device,
    TensorOptions shape' dtype device,
    shape' ~ SqueezeAll shape
  ) =>
  Apply' SqueezeAllSpec ((Proxy device, (Proxy dtype, Proxy shape)), IO ()) (IO ())
  where
  apply' SqueezeAllSpec (_, agg) = agg >> do
    let t = ones @shape @dtype @device
        t' = squeezeAll t
    checkDynamicTensorAttributes t'

data LossSpec
  = BinaryCrossEntropySpec
  | MSELossSpec

instance
  ( TensorOptions shape dtype device,
    TensorOptions shape' dtype device,
    KnownReduction reduction,
    shape' ~ ConditionalReduction shape reduction,
    StandardFloatingPointDTypeValidation device dtype
  ) =>
  Apply' LossSpec ((Proxy reduction, (Proxy device, (Proxy dtype, Proxy shape))), IO ()) (IO ())
  where
  apply' BinaryCrossEntropySpec (_, agg) = agg >> do
    let weight = ones @shape @dtype @device
        prediction = ones @shape @dtype @device
        target = ones @shape @dtype @device
        t = binaryCrossEntropy @reduction weight prediction target
    checkDynamicTensorAttributes t
  apply' MSELossSpec (_, agg) = agg >> do
    let prediction = ones @shape @dtype @device
        target = ones @shape @dtype @device
        t = mseLoss @reduction prediction target
    checkDynamicTensorAttributes t

data SoftmaxSpec
  = SoftmaxSpec
  | LogSoftmaxSpec

instance
  ( TensorOptions shape dtype device,
    KnownNat dim,
    DimOutOfBoundCheck shape dim,
    KnownDType dtype,
    StandardFloatingPointDTypeValidation device dtype
  ) =>
  Apply' SoftmaxSpec ((Proxy dim, (Proxy device, (Proxy dtype, Proxy shape))), IO ()) (IO ())
  where
  apply' SoftmaxSpec (_, agg) = agg >> do
    let t = ones @shape @dtype @device
        t' = softmax @dim t
    checkDynamicTensorAttributes t'
  apply' LogSoftmaxSpec (_, agg) = agg >> do
    let t = ones @shape @dtype @device
        t' = logSoftmax @dim t
    checkDynamicTensorAttributes t'

data DotSpec = DotSpec

instance
  ( TensorOptions '[size] dtype device,
    DotDTypeIsValid device dtype,
    KnownDType dtype,
    KnownDevice device
  ) =>
  Apply' DotSpec ((Proxy device, (Proxy dtype, Proxy size)), IO ()) (IO ())
  where
  apply' DotSpec (_, agg) = agg >> do
    let a = ones @'[size] @dtype @device
        b = ones @'[size] @dtype @device
        t = dot a b
    checkDynamicTensorAttributes t

data InverseSpec = InverseSpec

instance
  ( TensorOptions shape dtype device,
    TensorOptions shape' dtype device,
    shape' ~ Square shape,
    InverseShapeIsValid device shape,
    InverseDTypeIsValid device dtype,
    RandDTypeIsValid device dtype
  ) =>
  Apply' InverseSpec ((Proxy device, (Proxy dtype, Proxy shape)), IO ()) (IO ())
  where
  apply' InverseSpec (_, agg) = agg >> do
    t <- rand @shape @dtype @device
    let t' = inverse t
    checkDynamicTensorAttributes t'

data SymeigSpec = SymeigSpec | SymeigvaluesSpec

instance
  ( TensorOptions shape dtype device,
    TensorOptions shape' dtype device,
    TensorOptions shape'' dtype device,
    shape' ~ VectorOfSquare shape,
    shape'' ~ Square shape,
    SymeigDTypeIsValid device dtype,
    RandDTypeIsValid device dtype
  ) =>
  Apply' SymeigSpec ((Proxy device, (Proxy dtype, Proxy shape)), IO ()) (IO ())
  where
  apply' SymeigSpec (_, agg) = agg >> do
    t <- rand @shape @dtype @device
    foldMap
      ( \upper -> do
          let (t', t'') = symeig upper t
          checkDynamicTensorAttributes t'
          checkDynamicTensorAttributes t''
      )
      [D.Upper, D.Lower]
  apply' SymeigvaluesSpec (_, agg) = agg >> do
    t <- rand @shape @dtype @device
    foldMap
      ( \upper -> do
          let t' = symeigvalues upper t
          checkDynamicTensorAttributes t'
      )
      [D.Upper, D.Lower]

data EigSpec = EigSpec

instance
  ( TensorOptions shape dtype device,
    TensorOptions shape' dtype device,
    shape ~ '[n, n],
    shape' ~ ConditionalEigenVectors eigenvectors n,
    KnownNat n,
    KnownEigenVectors eigenvectors,
    KnownDType dtype,
    KnownDevice device,
    EigDTypeIsValid device dtype,
    RandDTypeIsValid device dtype
  ) =>
  Apply' EigSpec ((Proxy eigenvectors, (Proxy device, (Proxy dtype, Proxy n))), IO ()) (IO ())
  where
  apply' EigSpec (_, agg) = agg >> do
    t <- rand @shape @dtype @device
    let (t', t'') = eig @eigenvectors @n @shape' @dtype @device t
    checkDynamicTensorAttributes t'
    checkDynamicTensorAttributes t''

data SVDSpec = SVDSpec

instance
  ( TensorOptions shape dtype device,
    TensorOptions shapeU dtype device,
    TensorOptions shapeS dtype device,
    TensorOptions shapeV dtype device,
    KnownReducedSVD reduced,
    '(shapeU, shapeS, shapeV) ~ SVDShapes shape reduced,
    RandDTypeIsValid device dtype,
    SVDDTypeIsValid device dtype
  ) =>
  Apply' SVDSpec ((Proxy reduced, (Proxy device, (Proxy dtype, Proxy shape))), IO ()) (IO ())
  where
  apply' SVDSpec (_, agg) = agg >> do
    a <- randn @shape @dtype @device
    let (u, s, v) = svd @reduced a
    checkDynamicTensorAttributes u
    checkDynamicTensorAttributes s
    checkDynamicTensorAttributes v

data CholeskySpec = CholeskySpec

instance
  ( TensorOptions shape dtype device,
    TensorOptions shape' dtype device,
    TensorOptions shape'' dtype device,
    shape' ~ Square shape,
    shape'' ~ Square (MatMul shape (Transpose shape (LastDim shape) (LastDim shape - 1))),
    1 <= LastDim shape,
    KnownNat (LastDim shape),
    KnownNat (LastDim shape - 1),
    MatMulDTypeIsValid device dtype,
    CholeskyDTypeIsValid device dtype,
    RandDTypeIsValid device dtype
  ) =>
  Apply' CholeskySpec ((Proxy device, (Proxy dtype, Proxy shape)), IO ()) (IO ())
  where
  apply' CholeskySpec (_, agg) = agg >> do
    t <- rand @shape @dtype @device
    let t' = t `matmul` transpose @(Backwards shape 0) @(Backwards shape 1) t
    foldMap
      ( \tri -> do
          let t'' = cholesky tri t'
          checkDynamicTensorAttributes t''
      )
      [D.Upper, D.Lower]

data CholeskyInverseSpec = CholeskyInverseSpec

instance
  ( TensorOptions shape dtype device,
    shape ~ '[n, n],
    1 <= n,
    RandDTypeIsValid device dtype,
    MatMulDTypeIsValid device dtype,
    CholeskyDTypeIsValid device dtype
  ) =>
  Apply' CholeskyInverseSpec ((Proxy device, (Proxy dtype, Proxy shape)), IO ()) (IO ())
  where
  apply' CholeskyInverseSpec (_, agg) = agg >> do
    t <- rand @shape @dtype @device
    let t' = t `matmul` transpose @0 @1 t
    foldMap
      ( \tri -> do
          let t'' = cholesky tri t'
          let t''' = choleskyInverse tri t''
          checkDynamicTensorAttributes t'''
      )
      [D.Upper, D.Lower]

data CholeskySolveSpec = CholeskySolveSpec

instance
  ( TensorOptions m_k dtype device,
    TensorOptions m_m dtype device,
    Square m_m ~ m_m,
    MatMul m_m (Transpose m_m (LastDim m_m) (LastDim m_m - 1)) ~ m_m,
    FstSquareDim m_m ~ FstSquareDim m_k,
    1 <= FstSquareDim m_m,
    1 <= LastDim m_m,
    KnownNat (LastDim m_m),
    KnownNat (LastDim m_m - 1),
    MatMulDTypeIsValid device dtype,
    CholeskyDTypeIsValid device dtype,
    RandDTypeIsValid device dtype
  ) =>
  Apply' CholeskySolveSpec ((Proxy device, (Proxy dtype, (Proxy m_k, Proxy m_m))), IO ()) (IO ())
  where
  apply' CholeskySolveSpec (_, agg) = agg >> do
    t <- rand @m_m @dtype @device
    let a = t `matmul` transpose @(Backwards m_m 0) @(Backwards m_m 1) t
    b <- rand @m_k
    foldMap
      ( \tri -> do
          let u = cholesky tri a
          checkDynamicTensorAttributes u
          let c = choleskySolve tri b u
          checkDynamicTensorAttributes c
      )
      [D.Upper, D.Lower]

data SolveSpec = SolveSpec

instance
  ( TensorOptions m_k dtype device,
    TensorOptions m_m dtype device,
    Square m_m ~ m_m,
    FstSquareDim m_m ~ FstSquareDim m_k,
    1 <= FstSquareDim m_m,
    SolveDTypeIsValid device dtype,
    RandDTypeIsValid device dtype
  ) =>
  Apply' SolveSpec ((Proxy device, (Proxy dtype, (Proxy m_k, Proxy m_m))), IO ()) (IO ())
  where
  apply' SolveSpec (_, agg) = agg >> do
    b <- rand @m_k @dtype @device
    a <- rand @m_m
    let (c, lu) = solve b a
    checkDynamicTensorAttributes c
    checkDynamicTensorAttributes lu

data TransposeSpec = TransposeSpec

instance
  ( TensorOptions shape dtype device,
    TensorOptions shape' dtype device,
    shape' ~ Transpose shape n m,
    KnownNat n,
    KnownNat m
  ) =>
  Apply' TransposeSpec (((Proxy n, Proxy m), (Proxy device, (Proxy dtype, Proxy shape))), IO ()) (IO ())
  where
  apply' TransposeSpec (_, agg) = agg >> do
    let t = ones @shape @dtype @device
        t' = transpose @n @m t
    checkDynamicTensorAttributes t'

data Transpose2DSpec = Transpose2DSpec

instance
  ( TensorOptions '[i, j] dtype device,
    TensorOptions '[j, i] dtype device
  ) =>
  Apply' Transpose2DSpec ((Proxy device, (Proxy dtype, Proxy '[i, j])), IO ()) (IO ())
  where
  apply' Transpose2DSpec (_, agg) = agg >> do
    let t = ones @'[i, j] @dtype @device
        t' = transpose2D t
    checkDynamicTensorAttributes t'

data AnyAllSpec = AnySpec | AllSpec

instance
  ( TensorOptions shape 'D.Bool device,
    KnownDevice device
  ) =>
  Apply' AnyAllSpec ((Proxy device, Proxy shape), IO ()) (IO ())
  where
  apply' AnySpec (_, agg) = agg >> do
    let t = ones @shape @'D.Bool @device
        t' = any t
    checkDynamicTensorAttributes t'
  apply' AllSpec (_, agg) = agg >> do
    let t = ones @shape @'D.Bool @device
        t' = all t
    checkDynamicTensorAttributes t'

data AnyPrimeAllPrimeSpec = AnyPrimeSpec | AllPrimeSpec

instance
  ( TensorOptions shape 'D.Bool device,
    TensorOptions shape' 'D.Bool device,
    KnownNat dim,
    KnownKeepOrDropDim keepOrDropDim,
    shape' ~ ConditionalDropDimension shape dim keepOrDropDim
  ) =>
  Apply' AnyPrimeAllPrimeSpec (((Proxy dim, Proxy keepOrDropDim), (Proxy device, Proxy shape)), IO ()) (IO ())
  where
  apply' AnyPrimeSpec (_, agg) = agg >> do
    let t = ones @shape @'D.Bool @device
        t' = anyDim @dim @keepOrDropDim t
    checkDynamicTensorAttributes t'
  apply' AllPrimeSpec (_, agg) = agg >> do
    let t = ones @shape @'D.Bool @device
        t' = allDim @dim @keepOrDropDim t
    checkDynamicTensorAttributes t'

data LstmCellSpec = LstmCellSpec

instance
  ( TensorOptions '[4 * hiddenSize, inputSize] dtype device,
    TensorOptions '[4 * hiddenSize, hiddenSize] dtype device,
    TensorOptions '[4 * hiddenSize] dtype device,
    TensorOptions '[batchSize, hiddenSize] dtype device,
    TensorOptions '[batchSize, inputSize] dtype device,
    KnownNat inputSize,
    KnownNat hiddenSize,
    KnownNat batchSize
  ) =>
  Apply' LstmCellSpec ((Proxy device, (Proxy dtype, (Proxy hiddenSize, Proxy inputSize, Proxy batchSize))), IO ()) (IO ())
  where
  apply' LstmCellSpec (_, agg) = agg >> do
    let wi = ones @'[4 * hiddenSize, inputSize] @dtype @device
        wh = ones @'[4 * hiddenSize, hiddenSize] @dtype @device
        bi = ones @'[4 * hiddenSize] @dtype @device
        bh = ones @'[4 * hiddenSize] @dtype @device
        cc = ones @'[batchSize, hiddenSize] @dtype @device
        hc = ones @'[batchSize, hiddenSize] @dtype @device
        input = ones @'[batchSize, inputSize] @dtype @device
        (ncc, nhc) = lstmCell wi wh bi bh (cc, hc) input
    checkDynamicTensorAttributes ncc
    checkDynamicTensorAttributes nhc

data GruCellSpec = GruCellSpec

instance
  ( TensorOptions '[3 * hiddenSize, inputSize] dtype device,
    TensorOptions '[3 * hiddenSize, hiddenSize] dtype device,
    TensorOptions '[3 * hiddenSize] dtype device,
    TensorOptions '[batchSize, hiddenSize] dtype device,
    TensorOptions '[batchSize, inputSize] dtype device,
    KnownNat inputSize,
    KnownNat hiddenSize,
    KnownNat batchSize
  ) =>
  Apply' GruCellSpec ((Proxy device, (Proxy dtype, (Proxy hiddenSize, Proxy inputSize, Proxy batchSize))), IO ()) (IO ())
  where
  apply' GruCellSpec (_, agg) = agg >> do
    let wi = ones @'[3 * hiddenSize, inputSize] @dtype @device
        wh = ones @'[3 * hiddenSize, hiddenSize] @dtype @device
        bi = ones @'[3 * hiddenSize] @dtype @device
        bh = ones @'[3 * hiddenSize] @dtype @device
        hx = ones @'[batchSize, hiddenSize] @dtype @device
        input = ones @'[batchSize, inputSize] @dtype @device
        nhx = gruCell wi wh bi bh hx input
    checkDynamicTensorAttributes nhx

spec = before_ printSeed $ do
  foldMap spec' availableDevices
  where
    printSeed = do
      putStr "      seed:"
      LibTorch.get_manual_seed >>= print

spec' :: D.Device -> Spec
spec' device =
  describe ("for " <> show device) $ do
    let standardShapes = Proxy @'[2, 3] :. HNil -- (Proxy :: Proxy ('[] :: [Nat])) :. Proxy @'[0]  :. Proxy @'[0, 1] :. Proxy @'[1, 0] :. Proxy @'[2, 3] :. HNil
        squareShapes = Proxy @'[0, 0] :. Proxy @'[1, 1] :. Proxy @'[2, 2] :. Proxy @'[0, 0, 0] :. Proxy @'[0, 1, 1] :. Proxy @'[1, 0, 0] :. Proxy @'[3, 2, 2] :. HNil
        reductions = Proxy @D.ReduceNone :. Proxy @D.ReduceMean :. Proxy @D.ReduceSum :. HNil
    describe "unary ops" $ do
      let dispatch unaryAllDTypesSpec = case device of
            D.Device {D.deviceType = D.CPU, D.deviceIndex = 0} ->
              hfoldrM @IO unaryAllDTypesSpec () (hattach cpu (hproduct allDTypes standardShapes))
            D.Device {D.deviceType = D.CUDA, D.deviceIndex = 0} ->
              hfoldrM @IO unaryAllDTypesSpec () (hattach cuda0 (hproduct allDTypes standardShapes))
      it "abs" $ case device of
        D.Device {D.deviceType = D.CPU, D.deviceIndex = 0} ->
          hfoldrM @IO AbsSpec () (hattach cpu (hproduct standardDTypes standardShapes))
        D.Device {D.deviceType = D.CUDA, D.deviceIndex = 0} ->
          hfoldrM @IO AbsSpec () (hattach cuda0 (hproduct standardDTypes standardShapes))
      it "sign" $ dispatch SignSpec
      it "onesLike" $ dispatch OnesLikeSpec
      it "zerosLike" $ dispatch ZerosLikeSpec
    describe "unary floating-point ops" $ do
      let scalarParams = (0.01 :: Double) :. (0.01 :: Float) :. (1 :: Int) :. HNil
          dispatch unaryStandardFloatingPointDTypesSpec = case device of
            D.Device {D.deviceType = D.CPU, D.deviceIndex = 0} ->
              hfoldrM @IO unaryStandardFloatingPointDTypesSpec () (hattach cpu (hproduct standardFloatingPointDTypes standardShapes))
            D.Device {D.deviceType = D.CUDA, D.deviceIndex = 0} ->
              hfoldrM @IO unaryStandardFloatingPointDTypesSpec () (hattach cuda0 (hproduct allFloatingPointDTypes standardShapes))
      it "frac" $ dispatch FracSpec
      it "ceil" $ dispatch CeilSpec
      it "floor" $ dispatch FloorSpec
      it "trunc" $ dispatch TruncSpec
      it "erf" $ dispatch ErfSpec
      it "erfc" $ dispatch ErfcSpec
      it "erfinv" $ dispatch ErfinvSpec
      it "exp" $ dispatch ExpSpec
      it "expm1" $ dispatch Expm1Spec
      it "log" $ dispatch LogSpec
      it "log1p" $ dispatch Log1pSpec
      it "log2" $ dispatch Log2Spec
      it "log10" $ dispatch Log10Spec
      it "lgamma" $ dispatch LgammaSpec
      it "digamma" $ dispatch DigammaSpec
      it "relu" $ dispatch ReluSpec
      it "selu" $ dispatch SeluSpec
      it "mish" $ case device of
        D.Device {D.deviceType = D.CPU, D.deviceIndex = 0} ->
          hfoldrM @IO MishSpec () (hattach cpu (hproduct standardFloatingPointDTypes standardShapes))
        D.Device {D.deviceType = D.CUDA, D.deviceIndex = 0} ->
          hfoldrM @IO MishSpec () (hattach cuda0 (hproduct allFloatingPointDTypes standardShapes))
      it "gelu" $ case device of
        D.Device {D.deviceType = D.CPU, D.deviceIndex = 0} ->
          hfoldrM @IO GeluSpec () (hattach cpu (hproduct standardFloatingPointDTypes standardShapes))
        D.Device {D.deviceType = D.CUDA, D.deviceIndex = 0} ->
          hfoldrM @IO GeluSpec () (hattach cuda0 (hproduct standardFloatingPointDTypes standardShapes))
      it "leakyRelu" $ case device of
        D.Device {D.deviceType = D.CPU, D.deviceIndex = 0} ->
          hfoldrM @IO
            LeakyReluSpec
            ()
            (hattach cpu (hproduct scalarParams (hproduct standardFloatingPointDTypes standardShapes)))
        D.Device {D.deviceType = D.CUDA, D.deviceIndex = 0} ->
          hfoldrM @IO
            LeakyReluSpec
            ()
            (hattach cuda0 (hproduct scalarParams (hproduct standardFloatingPointDTypes standardShapes)))
      it "elu" $ case device of
        D.Device {D.deviceType = D.CPU, D.deviceIndex = 0} ->
          hfoldrM @IO
            ELUSpec
            ()
            (hattach cpu (hproduct (hzip scalarParams (hzip scalarParams scalarParams)) (hproduct standardFloatingPointDTypes standardShapes)))
        D.Device {D.deviceType = D.CUDA, D.deviceIndex = 0} ->
          hfoldrM @IO
            ELUSpec
            ()
            (hattach cuda0 (hproduct (hzip scalarParams (hzip scalarParams scalarParams)) (hproduct standardFloatingPointDTypes standardShapes)))
      it "sigmoid" $ dispatch SigmoidSpec
      it "logSigmoid" $ dispatch LogSigmoidSpec
      it "sin" $ dispatch SinSpec
      it "sinh" $ dispatch SinhSpec
      it "asin" $ dispatch AsinSpec
      it "cos" $ dispatch CosSpec
      it "cosh" $ dispatch CoshSpec
      it "acos" $ dispatch AcosSpec
      it "tan" $ dispatch TanSpec
      it "tanh" $ dispatch TanhSpec
      it "atan" $ dispatch AtanSpec
      it "sqrt" $ dispatch SqrtSpec
      it "rsqrt" $ dispatch RsqrtSpec
      it "randLike" $ dispatch RandLikeSpec
      it "randnLike" $ dispatch RandnLikeSpec
      it "toDType" $ case device of
        D.Device {D.deviceType = D.CPU, D.deviceIndex = 0} ->
          hfoldrM @IO ToDTypeSpec () (hattach cpu (hproduct (hproduct allDTypes allDTypes) standardShapes))
        D.Device {D.deviceType = D.CUDA, D.deviceIndex = 0} ->
          hfoldrM @IO ToDTypeSpec () (hattach cuda0 (hproduct (hproduct allDTypes allDTypes) standardShapes))
    describe "aggregation" $ do
      it "sumAll" $ case device of
        D.Device {D.deviceType = D.CPU, D.deviceIndex = 0} ->
          hfoldrM @IO SumAllSpec () (hattach cpu (hproduct almostAllDTypes standardShapes))
        D.Device {D.deviceType = D.CUDA, D.deviceIndex = 0} ->
          hfoldrM @IO SumAllSpec () (hattach cuda0 (hproduct allDTypes standardShapes))
      it "sumDim" $ do
        let sumDimDims = Proxy @0 :. Proxy @1 :. HNil
            sumDimShapes = Proxy @'[1, 0] :. Proxy @'[2, 3] :. HNil
        case device of
          D.Device {D.deviceType = D.CPU, D.deviceIndex = 0} ->
            hfoldrM @IO SumDimSpec () (hproduct sumDimDims (hattach cpu (hproduct almostAllDTypes sumDimShapes)))
          D.Device {D.deviceType = D.CUDA, D.deviceIndex = 0} ->
            hfoldrM @IO SumDimSpec () (hproduct sumDimDims (hattach cuda0 (hproduct allDTypes sumDimShapes)))
      let aggregationShapes = (Proxy :: Proxy ('[] :: [Nat])) :. Proxy @'[1] :. Proxy @'[2, 3] :. HNil
          dispatch aggregationSpec = case device of
            D.Device {D.deviceType = D.CPU, D.deviceIndex = 0} ->
              hfoldrM @IO aggregationSpec () (hattach cpu (hproduct almostAllDTypes standardShapes))
            D.Device {D.deviceType = D.CUDA, D.deviceIndex = 0} ->
              hfoldrM @IO aggregationSpec () (hattach cuda0 (hproduct allDTypes standardShapes))
      it "min" $ dispatch MinSpec
      it "max" $ dispatch MaxSpec
      it "median" $ dispatch MedianSpec
      it "mean" $ dispatch MeanSpec
    describe "shape ops" $ do
      it "squeezeAll" $ case device of
        D.Device {D.deviceType = D.CPU, D.deviceIndex = 0} ->
          hfoldrM @IO SqueezeAllSpec () (hattach cpu (hproduct allDTypes standardShapes))
        D.Device {D.deviceType = D.CUDA, D.deviceIndex = 0} ->
          hfoldrM @IO SqueezeAllSpec () (hattach cuda0 (hproduct allDTypes standardShapes))
      it "transpose" $ do
        let transposeDims =
              hzip
                (Proxy @0 :. Proxy @0 :. Proxy @1 :. HNil)
                (Proxy @0 :. Proxy @1 :. Proxy @0 :. HNil)
            transposeShapes = Proxy @'[0, 0] :. Proxy @'[0, 1] :. Proxy @'[1, 0] :. Proxy @'[2, 3] :. Proxy @'[0, 1, 1] :. Proxy @'[1, 0, 1] :. HNil
        case device of
          D.Device {D.deviceType = D.CPU, D.deviceIndex = 0} ->
            hfoldrM @IO TransposeSpec () (hproduct transposeDims (hattach cpu (hproduct allDTypes transposeShapes)))
          D.Device {D.deviceType = D.CUDA, D.deviceIndex = 0} ->
            hfoldrM @IO TransposeSpec () (hproduct transposeDims (hattach cuda0 (hproduct allDTypes transposeShapes)))
      it "transpose2d" $ case device of
        D.Device {D.deviceType = D.CPU, D.deviceIndex = 0} ->
          hfoldrM @IO Transpose2DSpec () (hattach cpu (hproduct allDTypes (Proxy @'[2, 3] :. HNil)))
        D.Device {D.deviceType = D.CUDA, D.deviceIndex = 0} ->
          hfoldrM @IO Transpose2DSpec () (hattach cuda0 (hproduct allDTypes (Proxy @'[2, 3] :. HNil)))
    describe "loss functions" $ do
      let dispatch lossSpec = case device of
            D.Device {D.deviceType = D.CPU, D.deviceIndex = 0} ->
              hfoldrM @IO lossSpec () (hproduct reductions (hattach cpu (hproduct standardFloatingPointDTypes standardShapes)))
            D.Device {D.deviceType = D.CUDA, D.deviceIndex = 0} ->
              hfoldrM @IO lossSpec () (hproduct reductions (hattach cuda0 (hproduct allFloatingPointDTypes standardShapes)))
      it "binaryCrossEntropy" $ dispatch BinaryCrossEntropySpec
      it "mseLoss" $ dispatch MSELossSpec
    describe "softmax" $ do
      let softmaxDims = Proxy @0 :. Proxy @1 :. HNil
          softmaxShapes = Proxy @'[1, 0] :. Proxy @'[2, 3] :. HNil
          dispatch softmaxSpec = case device of
            D.Device {D.deviceType = D.CPU, D.deviceIndex = 0} ->
              hfoldrM @IO softmaxSpec () (hproduct softmaxDims (hattach cpu (hproduct standardFloatingPointDTypes standardShapes)))
            D.Device {D.deviceType = D.CUDA, D.deviceIndex = 0} ->
              hfoldrM @IO softmaxSpec () (hproduct softmaxDims (hattach cuda0 (hproduct allFloatingPointDTypes standardShapes)))
      it "softmax" $ dispatch SoftmaxSpec
      it "logSoftmax" $ dispatch LogSoftmaxSpec
    describe "linear algrebra" $ do
      it "dot" $ case device of
        D.Device {D.deviceType = D.CPU, D.deviceIndex = 0} ->
          hfoldrM @IO DotSpec () (hattach cpu (hproduct standardDTypes (Proxy @0 :. Proxy @1 :. Proxy @2 :. HNil)))
        D.Device {D.deviceType = D.CUDA, D.deviceIndex = 0} ->
          hfoldrM @IO DotSpec () (hattach cuda0 (hproduct allFloatingPointDTypes (Proxy @0 :. Proxy @1 :. Proxy @2 :. HNil)))
      it "inverse" $ case device of
        D.Device {D.deviceType = D.CPU, D.deviceIndex = 0} ->
          hfoldrM @IO InverseSpec () (hattach cpu (hproduct standardFloatingPointDTypes squareShapes))
        D.Device {D.deviceType = D.CUDA, D.deviceIndex = 0} ->
          hfoldrM @IO InverseSpec () (hattach cuda0 (hproduct standardFloatingPointDTypes (Proxy @'[1, 1] :. Proxy @'[2, 2] :. Proxy @'[1, 1, 1] :. Proxy @'[2, 2, 2] :. HNil)))
      let dispatchSymeigSpec symeigSpec = case device of
            D.Device {D.deviceType = D.CPU, D.deviceIndex = 0} ->
              hfoldrM @IO symeigSpec () (hattach cpu (hproduct standardFloatingPointDTypes squareShapes))
            D.Device {D.deviceType = D.CUDA, D.deviceIndex = 0} ->
              hfoldrM @IO symeigSpec () (hattach cuda0 (hproduct standardFloatingPointDTypes squareShapes))
      it "symeig" $ do
        dispatchSymeigSpec SymeigSpec
      it "symeigvalues" $ do
        dispatchSymeigSpec SymeigvaluesSpec
      it "eig" $ do
        let eigenVectors = Proxy @'EnableEigenVectors :. Proxy @'DisableEigenVectors :. HNil
            ns = Proxy @0 :. Proxy @2 :. Proxy @10 :. HNil
        case device of
          D.Device {D.deviceType = D.CPU, D.deviceIndex = 0} ->
            hfoldrM @IO EigSpec () (hproduct eigenVectors (hattach cpu (hproduct standardFloatingPointDTypes ns)))
          D.Device {D.deviceType = D.CUDA, D.deviceIndex = 0} ->
            hfoldrM @IO EigSpec () (hproduct eigenVectors (hattach cuda0 (hproduct standardFloatingPointDTypes ns)))
      it "svd" $ do
        let svdShapes = Proxy @'[0, 0] :. Proxy @'[0, 1] :. Proxy @'[1, 0] :. Proxy @'[1, 1] :. Proxy @'[1, 2] :. Proxy @'[2, 1] :. Proxy @'[0, 0, 0] :. Proxy @'[0, 0, 1] :. Proxy @'[0, 1, 0] :. Proxy @'[0, 1, 1] :. Proxy @'[1, 0, 0] :. Proxy @'[1, 0, 1] :. Proxy @'[1, 1, 0] :. Proxy @'[1, 1, 1] :. Proxy @'[3, 2, 3] :. Proxy @'[3, 3, 2] :. HNil
            reducedSVD = Proxy @'ThinSVD :. Proxy @'FullSVD :. HNil
        case device of
          D.Device {D.deviceType = D.CPU, D.deviceIndex = 0} ->
            hfoldrM @IO SVDSpec () (hproduct reducedSVD (hattach cpu (hproduct standardFloatingPointDTypes svdShapes)))
          D.Device {D.deviceType = D.CUDA, D.deviceIndex = 0} ->
            hfoldrM @IO SVDSpec () (hproduct reducedSVD (hattach cuda0 (hproduct standardFloatingPointDTypes svdShapes)))
      it "cholesky" $ case device of
        D.Device {D.deviceType = D.CPU, D.deviceIndex = 0} ->
          hfoldrM @IO CholeskySpec () (hattach cpu (hproduct standardFloatingPointDTypes squareShapes))
        D.Device {D.deviceType = D.CUDA, D.deviceIndex = 0} ->
          hfoldrM @IO CholeskySpec () (hattach cuda0 (hproduct standardFloatingPointDTypes squareShapes))
      it "choleskyInverse" $ do
        let choleskyInverseShapes = Proxy @'[1, 1] :. Proxy @'[2, 2] :. HNil
        case device of
          D.Device {D.deviceType = D.CPU, D.deviceIndex = 0} ->
            hfoldrM @IO CholeskyInverseSpec () (hattach cpu (hproduct standardFloatingPointDTypes choleskyInverseShapes))
          D.Device {D.deviceType = D.CUDA, D.deviceIndex = 0} ->
            hfoldrM @IO CholeskyInverseSpec () (hattach cuda0 (hproduct standardFloatingPointDTypes choleskyInverseShapes))
      it "choleskySolve" $ do
        let choleskySolveShapes =
              hzip
                (Proxy @'[1, 0] :. Proxy @'[1, 2] :. Proxy @'[2, 1] :. Proxy @'[3, 1, 2] :. HNil)
                (Proxy @'[1, 1] :. Proxy @'[1, 1] :. Proxy @'[2, 2] :. Proxy @'[3, 1, 1] :. HNil)
        case device of
          D.Device {D.deviceType = D.CPU, D.deviceIndex = 0} ->
            hfoldrM @IO CholeskySolveSpec () (hattach cpu (hproduct standardFloatingPointDTypes choleskySolveShapes))
          D.Device {D.deviceType = D.CUDA, D.deviceIndex = 0} ->
            hfoldrM @IO CholeskySolveSpec () (hattach cuda0 (hproduct standardFloatingPointDTypes choleskySolveShapes))
      it "solve" $ do
        let solveShapes =
              hzip
                (Proxy @'[1, 0] :. Proxy @'[1, 2] :. Proxy @'[2, 1] :. Proxy @'[3, 1, 2] :. HNil)
                (Proxy @'[1, 1] :. Proxy @'[1, 1] :. Proxy @'[2, 2] :. Proxy @'[3, 1, 1] :. HNil)
        case device of
          D.Device {D.deviceType = D.CPU, D.deviceIndex = 0} ->
            hfoldrM @IO SolveSpec () (hattach cpu (hproduct standardFloatingPointDTypes solveShapes))
          D.Device {D.deviceType = D.CUDA, D.deviceIndex = 0} ->
            hfoldrM @IO SolveSpec () (hattach cuda0 (hproduct standardFloatingPointDTypes solveShapes))
    describe "boolean algebra" $ do
      do
        let dispatch anyAllSpec = case device of
              D.Device {D.deviceType = D.CPU, D.deviceIndex = 0} ->
                hfoldrM @IO anyAllSpec () (hattach cpu standardShapes)
              D.Device {D.deviceType = D.CUDA, D.deviceIndex = 0} ->
                hfoldrM @IO anyAllSpec () (hattach cuda0 standardShapes)
        it "all" $ dispatch AllSpec
        it "any" $ dispatch AnySpec
      do
        let anyPrimeAllPrimeDims = Proxy @0 :. Proxy @1 :. HNil
            keepOrDropDims = Proxy @KeepDim :. Proxy @DropDim :. HNil
            anyPrimeAllPrimeShapes = Proxy @'[0, 0] :. Proxy @'[0, 1] :. Proxy @'[1, 0] :. Proxy @'[2, 3] :. Proxy @'[0, 1, 1] :. Proxy @'[1, 0, 1] :. HNil
            dispatch anyPrimeAllPrimeSpec = case device of
              D.Device {D.deviceType = D.CPU, D.deviceIndex = 0} ->
                hfoldrM @IO
                  anyPrimeAllPrimeSpec
                  ()
                  ( hproduct
                      (hproduct anyPrimeAllPrimeDims keepOrDropDims)
                      (hattach cpu anyPrimeAllPrimeShapes)
                  )
              D.Device {D.deviceType = D.CUDA, D.deviceIndex = 0} ->
                hfoldrM @IO
                  anyPrimeAllPrimeSpec
                  ()
                  ( hproduct
                      (hproduct anyPrimeAllPrimeDims keepOrDropDims)
                      (hattach cuda0 anyPrimeAllPrimeShapes)
                  )
        it "allDim" $ dispatch AllPrimeSpec
        it "anyDim" $ dispatch AnyPrimeSpec
    describe "pooling"
      $ it "maxPool2d"
      $ do
        let c = maxPool2d @'(1, 1) @'(1, 1) @'(0, 0) (ones :: CPUTensor 'D.Float '[1, 3, 4, 5])
        checkDynamicTensorAttributes c
    describe "sorting"
      $ it "topk"
      $ do
        let (c, c') = topk @3 @1 True True (ones :: CPUTensor 'D.Float '[2, 3])
        checkDynamicTensorAttributes c
        checkDynamicTensorAttributes c'
    describe "upsampling" $ do
      it "upsample_nearest2d" $ do
        let c = upsample_nearest2d @5 @3 (ones :: CPUTensor 'D.Float '[2, 3, 2, 2])
        checkDynamicTensorAttributes c
      it "upsample_bicubic2d" $ do
        let c = upsample_bicubic2d @5 @3 False (ones :: CPUTensor 'D.Float '[2, 3, 2, 2])
        checkDynamicTensorAttributes c
      it "upsample_bilinear2d" $ do
        let c = upsample_bilinear2d @5 @3 False (ones :: CPUTensor 'D.Float '[2, 3, 2, 2])
        checkDynamicTensorAttributes c
    describe "binary native ops" $ return ()
    describe "RNNCells op" $ do
      it "lstmCell op" $ do
        let sizes =
              hzip3
                (Proxy @2 :. Proxy @4 :. Proxy @6 :. Proxy @7 :. HNil)
                (Proxy @7 :. Proxy @6 :. Proxy @5 :. Proxy @4 :. HNil)
                (Proxy @5 :. Proxy @10 :. Proxy @15 :. Proxy @20 :. HNil)
        case device of
          D.Device {D.deviceType = D.CPU, D.deviceIndex = 0} ->
            hfoldrM @IO LstmCellSpec () (hattach cpu (hproduct standardFloatingPointDTypes sizes))
          D.Device {D.deviceType = D.CUDA, D.deviceIndex = 0} ->
            hfoldrM @IO LstmCellSpec () (hattach cuda0 (hproduct standardFloatingPointDTypes sizes))
      it "gruCell op" $ do
        let sizes =
              hzip3
                (Proxy @2 :. Proxy @4 :. Proxy @6 :. Proxy @7 :. HNil)
                (Proxy @7 :. Proxy @6 :. Proxy @5 :. Proxy @4 :. HNil)
                (Proxy @5 :. Proxy @10 :. Proxy @15 :. Proxy @20 :. HNil)
        case device of
          D.Device {D.deviceType = D.CPU, D.deviceIndex = 0} ->
            hfoldrM @IO GruCellSpec () (hattach cpu (hproduct standardFloatingPointDTypes sizes))
          D.Device {D.deviceType = D.CUDA, D.deviceIndex = 0} ->
            hfoldrM @IO GruCellSpec () (hattach cuda0 (hproduct standardFloatingPointDTypes sizes))
