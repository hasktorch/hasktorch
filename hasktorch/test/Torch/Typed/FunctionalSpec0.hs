{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE NoStarIsType #-}
{-# LANGUAGE FlexibleContexts #-}
{-# OPTIONS_GHC -freduction-depth=0 #-}

module Torch.Typed.FunctionalSpec0
  ( Torch.Typed.FunctionalSpec0.spec,
  )
where

import Data.Proxy
import GHC.TypeLits
import Test.Hspec (Spec, before_, describe, it)
import Test.QuickCheck ()
import Torch.Internal.Managed.Type.Context (get_manual_seed)
import Torch.Typed
import Torch.Typed.AuxiliarySpec
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
  apply' SignSpec (_, agg) =
    agg >> do
      let t = sign (ones @shape @dtype @device)
      checkDynamicTensorAttributes t
  apply' OnesLikeSpec (_, agg) =
    agg >> do
      let t = onesLike (ones @shape @dtype @device)
      checkDynamicTensorAttributes t
  apply' ZerosLikeSpec (_, agg) =
    agg >> do
      let t = zerosLike (ones @shape @dtype @device)
      checkDynamicTensorAttributes t

data UnaryStandardDTypesSpec
  = AbsSpec

instance
  ( TensorOptions shape dtype device,
    StandardDTypeValidation device dtype
  ) =>
  Apply' UnaryStandardDTypesSpec ((Proxy device, (Proxy dtype, Proxy shape)), IO ()) (IO ())
  where
  apply' AbsSpec (_, agg) =
    agg >> do
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
  apply' FracSpec (_, agg) =
    agg >> do
      let t = frac (ones @shape @dtype @device)
      checkDynamicTensorAttributes t
  apply' CeilSpec (_, agg) =
    agg >> do
      let t = ceil (ones @shape @dtype @device)
      checkDynamicTensorAttributes t
  apply' FloorSpec (_, agg) =
    agg >> do
      let t = floor (ones @shape @dtype @device)
      checkDynamicTensorAttributes t
  apply' TruncSpec (_, agg) =
    agg >> do
      let t = trunc (ones @shape @dtype @device)
      checkDynamicTensorAttributes t
  apply' ErfSpec (_, agg) =
    agg >> do
      let t = erf (ones @shape @dtype @device)
      checkDynamicTensorAttributes t
  apply' ErfcSpec (_, agg) =
    agg >> do
      let t = erfc (ones @shape @dtype @device)
      checkDynamicTensorAttributes t
  apply' ErfinvSpec (_, agg) =
    agg >> do
      let t = erfinv (ones @shape @dtype @device)
      checkDynamicTensorAttributes t
  apply' ExpSpec (_, agg) =
    agg >> do
      let t = exp (ones @shape @dtype @device)
      checkDynamicTensorAttributes t
  apply' Expm1Spec (_, agg) =
    agg >> do
      let t = expm1 (ones @shape @dtype @device)
      checkDynamicTensorAttributes t
  apply' LogSpec (_, agg) =
    agg >> do
      let t = log (ones @shape @dtype @device)
      checkDynamicTensorAttributes t
  apply' Log1pSpec (_, agg) =
    agg >> do
      let t = log1p (ones @shape @dtype @device)
      checkDynamicTensorAttributes t
  apply' Log2Spec (_, agg) =
    agg >> do
      let t = log2 (ones @shape @dtype @device)
      checkDynamicTensorAttributes t
  apply' Log10Spec (_, agg) =
    agg >> do
      let t = log10 (ones @shape @dtype @device)
      checkDynamicTensorAttributes t
  apply' LgammaSpec (_, agg) =
    agg >> do
      let t = lgamma (ones @shape @dtype @device)
      checkDynamicTensorAttributes t
  apply' DigammaSpec (_, agg) =
    agg >> do
      let t = digamma (ones @shape @dtype @device)
      checkDynamicTensorAttributes t
  apply' ReluSpec (_, agg) =
    agg >> do
      let t = relu (ones @shape @dtype @device)
      checkDynamicTensorAttributes t
  apply' SeluSpec (_, agg) =
    agg >> do
      let t = selu (ones @shape @dtype @device)
      checkDynamicTensorAttributes t
  apply' SigmoidSpec (_, agg) =
    agg >> do
      let t = sigmoid (ones @shape @dtype @device)
      checkDynamicTensorAttributes t
  apply' LogSigmoidSpec (_, agg) =
    agg >> do
      let t = logSigmoid (ones @shape @dtype @device)
      checkDynamicTensorAttributes t
  apply' SinSpec (_, agg) =
    agg >> do
      let t = sin (ones @shape @dtype @device)
      checkDynamicTensorAttributes t
  apply' SinhSpec (_, agg) =
    agg >> do
      let t = sinh (ones @shape @dtype @device)
      checkDynamicTensorAttributes t
  apply' AsinSpec (_, agg) =
    agg >> do
      let t = asin (ones @shape @dtype @device)
      checkDynamicTensorAttributes t
  apply' CosSpec (_, agg) =
    agg >> do
      let t = cos (ones @shape @dtype @device)
      checkDynamicTensorAttributes t
  apply' CoshSpec (_, agg) =
    agg >> do
      let t = cosh (ones @shape @dtype @device)
      checkDynamicTensorAttributes t
  apply' AcosSpec (_, agg) =
    agg >> do
      let t = acos (ones @shape @dtype @device)
      checkDynamicTensorAttributes t
  apply' TanSpec (_, agg) =
    agg >> do
      let t = tan (ones @shape @dtype @device)
      checkDynamicTensorAttributes t
  apply' TanhSpec (_, agg) =
    agg >> do
      let t = tanh (ones @shape @dtype @device)
      checkDynamicTensorAttributes t
  apply' AtanSpec (_, agg) =
    agg >> do
      let t = atan (ones @shape @dtype @device)
      checkDynamicTensorAttributes t
  apply' SqrtSpec (_, agg) =
    agg >> do
      let t = sqrt (ones @shape @dtype @device)
      checkDynamicTensorAttributes t
  apply' RsqrtSpec (_, agg) =
    agg >> do
      let t = rsqrt (ones @shape @dtype @device)
      checkDynamicTensorAttributes t
  apply' RandLikeSpec (_, agg) =
    agg >> do
      t <- randLike (ones @shape @dtype @device)
      checkDynamicTensorAttributes t
  apply' RandnLikeSpec (_, agg) =
    agg >> do
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
  apply' MishSpec (_, agg) =
    agg >> do
      let t = mish (ones @shape @dtype @device)
      checkDynamicTensorAttributes t

data GeluSpec = GeluSpec

instance
  ( TensorOptions shape dtype device,
    GeluDTypeIsValid device dtype
  ) =>
  Apply' GeluSpec ((Proxy device, (Proxy dtype, Proxy shape)), IO ()) (IO ())
  where
  apply' GeluSpec (_, agg) =
    agg >> do
      let t = gelu (ones @shape @dtype @device)
      checkDynamicTensorAttributes t

data LeakyReluSpec = LeakyReluSpec

instance
  ( TensorOptions shape dtype device,
    Scalar a,
    StandardFloatingPointDTypeValidation device dtype
  ) =>
  Apply' LeakyReluSpec ((Proxy device, (a, (Proxy dtype, Proxy shape))), IO ()) (IO ())
  where
  apply' LeakyReluSpec ((_, (negativeSlope, _)), agg) =
    agg >> do
      let t = leakyRelu negativeSlope (ones @shape @dtype @device)
      checkDynamicTensorAttributes t

data ELUSpec = ELUSpec

instance
  ( TensorOptions shape dtype device,
    Scalar a,
    StandardFloatingPointDTypeValidation device dtype
  ) =>
  Apply' ELUSpec ((Proxy device, ((a, (a, a)), (Proxy dtype, Proxy shape))), IO ()) (IO ())
  where
  apply' ELUSpec ((_, ((alpha, (scale, inputScale)), _)), agg) =
    agg >> do
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
  apply' ToDTypeSpec (_, agg) =
    agg >> do
      let t = ones @shape @dtype @device
          t' = toDType @dtype' @dtype t
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
  apply' SumAllSpec (_, agg) =
    agg >> do
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
  apply' SumDimSpec (_, agg) =
    agg >> do
      let t = ones @shape @dtype @device
          t' = sumDim @d t
      checkDynamicTensorAttributes t'

data MinMaxSpec
  = MinSpec
  | MaxSpec

instance
  ( TensorOptions shape dtype device,
    KnownDType dtype,
    KnownDevice device,
    MinMaxDTypeIsValid device dtype,
    AllDimsPositive shape
  ) =>
  Apply' MinMaxSpec ((Proxy device, (Proxy dtype, Proxy shape)), IO ()) (IO ())
  where
  apply' MinSpec (_, agg) =
    agg >> do
      let t = ones @shape @dtype @device
          t' = min t
      checkDynamicTensorAttributes t'
  apply' MaxSpec (_, agg) =
    agg >> do
      let t = ones @shape @dtype @device
          t' = max t
      checkDynamicTensorAttributes t'

data MeanAllSpec = MeanAllSpec

instance
  ( TensorOptions shape dtype device,
    KnownDType dtype,
    KnownDevice device,
    MeanDTypeValidation device dtype,
    AllDimsPositive shape
  ) =>
  Apply' MeanAllSpec ((Proxy device, (Proxy dtype, Proxy shape)), IO ()) (IO ())
  where
  apply' MeanAllSpec (_, agg) =
    agg >> do
      let t = ones @shape @dtype @device
          t' = meanAll t
      checkDynamicTensorAttributes t'

data MeanDimSpec = MeanDimSpec

instance
  ( TensorOptions shape dtype device,
    TensorOptions shape' dtype device,
    KnownNat dim,
    shape' ~ DropValue shape dim,
    MeanDTypeValidation device dtype,
    AllDimsPositive shape
  ) =>
  Apply' MeanDimSpec ((Proxy dim, (Proxy device, (Proxy dtype, Proxy shape))), IO ()) (IO ())
  where
  apply' MeanDimSpec (_, agg) =
    agg >> do
      let t = ones @shape @dtype @device
          t' = meanDim @dim t
      checkDynamicTensorAttributes t'

data MeanSpec = MeanSpec

instance
  ( TensorOptions shape dtype device,
    TensorOptions shape' dtype device,
    KnownNat dim,
    KnownKeepOrDropDim keepOrDropDim,
    shape' ~ ConditionalDropDimension shape dim keepOrDropDim,
    MeanDTypeValidation device dtype,
    AllDimsPositive shape
  ) =>
  Apply' MeanSpec (((Proxy dim, Proxy keepOrDropDim), (Proxy device, (Proxy dtype, Proxy shape))), IO ()) (IO ())
  where
  apply' MeanSpec (_, agg) =
    agg >> do
      let t = ones @shape @dtype @device
          t' = mean @dim @keepOrDropDim t
      checkDynamicTensorAttributes t'

data MedianAllSpec = MedianAllSpec

instance
  ( TensorOptions shape dtype device,
    KnownDType dtype,
    KnownDevice device,
    StandardDTypeValidation device dtype,
    AllDimsPositive shape
  ) =>
  Apply' MedianAllSpec ((Proxy device, (Proxy dtype, Proxy shape)), IO ()) (IO ())
  where
  apply' MedianAllSpec (_, agg) =
    agg >> do
      let t = ones @shape @dtype @device
          t' = medianAll t
      checkDynamicTensorAttributes t'

data MedianDimSpec = MedianDimSpec

instance
  ( TensorOptions shape dtype device,
    TensorOptions shape' dtype device,
    TensorOptions shape' 'Int64 device,
    KnownNat dim,
    shape' ~ DropValue shape dim,
    StandardDTypeValidation device dtype,
    AllDimsPositive shape
  ) =>
  Apply' MedianDimSpec ((Proxy dim, (Proxy device, (Proxy dtype, Proxy shape))), IO ()) (IO ())
  where
  apply' MedianDimSpec (_, agg) =
    agg >> do
      let t = ones @shape @dtype @device
          (t', t'') = medianDim @dim t
      checkDynamicTensorAttributes t'
      checkDynamicTensorAttributes t''

data MedianSpec = MedianSpec

instance
  ( TensorOptions shape dtype device,
    TensorOptions shape' dtype device,
    TensorOptions shape' 'Int64 device,
    KnownNat dim,
    KnownKeepOrDropDim keepOrDropDim,
    shape' ~ ConditionalDropDimension shape dim keepOrDropDim,
    StandardDTypeValidation device dtype,
    AllDimsPositive shape
  ) =>
  Apply' MedianSpec (((Proxy dim, Proxy keepOrDropDim), (Proxy device, (Proxy dtype, Proxy shape))), IO ()) (IO ())
  where
  apply' MedianSpec (_, agg) =
    agg >> do
      let t = ones @shape @dtype @device
          (t', t'') = median @dim @keepOrDropDim t
      checkDynamicTensorAttributes t'
      checkDynamicTensorAttributes t''

data ModeSpec = ModeSpec

instance
  ( TensorOptions shape dtype device,
    TensorOptions shape' dtype device,
    TensorOptions shape' 'Int64 device,
    KnownNat dim,
    KnownKeepOrDropDim keepOrDropDim,
    shape' ~ ConditionalDropDimension shape dim keepOrDropDim,
    StandardDTypeValidation device dtype,
    AllDimsPositive shape
  ) =>
  Apply' ModeSpec (((Proxy dim, Proxy keepOrDropDim), (Proxy device, (Proxy dtype, Proxy shape))), IO ()) (IO ())
  where
  apply' ModeSpec (_, agg) =
    agg >> do
      let t = ones @shape @dtype @device
          (t', t'') = mode @dim @keepOrDropDim t
      checkDynamicTensorAttributes t'
      checkDynamicTensorAttributes t''

spec :: Spec
spec = before_ printSeed $ do
  foldMap spec' availableDevices
  where
    printSeed = do
      putStr "      seed:"
      get_manual_seed >>= print

spec' :: Device -> Spec
spec' device =
  describe ("for " <> show device) $ do
    let standardShapes = Proxy @'[2, 3] :. HNil -- (Proxy :: Proxy ('[] :: [Nat])) :. Proxy @'[0]  :. Proxy @'[0, 1] :. Proxy @'[1, 0] :. Proxy @'[2, 3] :. HNil
        squareShapes = Proxy @'[0, 0] :. Proxy @'[1, 1] :. Proxy @'[2, 2] :. Proxy @'[0, 0, 0] :. Proxy @'[0, 1, 1] :. Proxy @'[1, 0, 0] :. Proxy @'[3, 2, 2] :. HNil
        reductions = Proxy @ReduceNone :. Proxy @ReduceMean :. Proxy @ReduceSum :. HNil

    describe "unary ops" $ do
      let dispatch unaryAllDTypesSpec = case device of
            Device {deviceType = CPU, deviceIndex = 0} ->
              hfoldrM @IO unaryAllDTypesSpec () (hattach cpu (hproduct allDTypes standardShapes))
            Device {deviceType = CUDA, deviceIndex = 0} ->
              hfoldrM @IO unaryAllDTypesSpec () (hattach cuda0 (hproduct allDTypes standardShapes))

      it "abs" $ case device of
        Device {deviceType = CPU, deviceIndex = 0} ->
          hfoldrM @IO AbsSpec () (hattach cpu (hproduct standardDTypes standardShapes))
        Device {deviceType = CUDA, deviceIndex = 0} ->
          hfoldrM @IO AbsSpec () (hattach cuda0 (hproduct standardDTypes standardShapes))
      it "sign" $ dispatch SignSpec
      it "onesLike" $ dispatch OnesLikeSpec
      it "zerosLike" $ dispatch ZerosLikeSpec

    describe "unary floating-point ops" $ do
      let scalarParams = (0.01 :: Double) :. (0.01 :: Float) :. (1 :: Int) :. HNil
          dispatch unaryStandardFloatingPointDTypesSpec = case device of
            Device {deviceType = CPU, deviceIndex = 0} ->
              hfoldrM @IO unaryStandardFloatingPointDTypesSpec () (hattach cpu (hproduct standardFloatingPointDTypes standardShapes))
            Device {deviceType = CUDA, deviceIndex = 0} ->
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
        Device {deviceType = CPU, deviceIndex = 0} ->
          hfoldrM @IO MishSpec () (hattach cpu (hproduct standardFloatingPointDTypes standardShapes))
        Device {deviceType = CUDA, deviceIndex = 0} ->
          hfoldrM @IO MishSpec () (hattach cuda0 (hproduct allFloatingPointDTypes standardShapes))
      it "gelu" $ case device of
        Device {deviceType = CPU, deviceIndex = 0} ->
          hfoldrM @IO GeluSpec () (hattach cpu (hproduct standardFloatingPointDTypes standardShapes))
        Device {deviceType = CUDA, deviceIndex = 0} ->
          hfoldrM @IO GeluSpec () (hattach cuda0 (hproduct standardFloatingPointDTypes standardShapes))
      it "leakyRelu" $ case device of
        Device {deviceType = CPU, deviceIndex = 0} ->
          hfoldrM @IO
            LeakyReluSpec
            ()
            (hattach cpu (hproduct scalarParams (hproduct standardFloatingPointDTypes standardShapes)))
        Device {deviceType = CUDA, deviceIndex = 0} ->
          hfoldrM @IO
            LeakyReluSpec
            ()
            (hattach cuda0 (hproduct scalarParams (hproduct standardFloatingPointDTypes standardShapes)))
      it "elu" $ case device of
        Device {deviceType = CPU, deviceIndex = 0} ->
          hfoldrM @IO
            ELUSpec
            ()
            (hattach cpu (hproduct (hzip scalarParams (hzip scalarParams scalarParams)) (hproduct standardFloatingPointDTypes standardShapes)))
        Device {deviceType = CUDA, deviceIndex = 0} ->
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
        Device {deviceType = CPU, deviceIndex = 0} ->
          hfoldrM @IO ToDTypeSpec () (hattach cpu (hproduct (hproduct allDTypes allDTypes) standardShapes))
        Device {deviceType = CUDA, deviceIndex = 0} ->
          hfoldrM @IO ToDTypeSpec () (hattach cuda0 (hproduct (hproduct allDTypes allDTypes) standardShapes))

    describe "aggregation" $ do
      it "sumAll" $ case device of
        Device {deviceType = CPU, deviceIndex = 0} ->
          hfoldrM @IO SumAllSpec () (hattach cpu (hproduct almostAllDTypes standardShapes))
        Device {deviceType = CUDA, deviceIndex = 0} ->
          hfoldrM @IO SumAllSpec () (hattach cuda0 (hproduct allDTypes standardShapes))
      it "sumDim" $ do
        let sumDimDims = Proxy @0 :. Proxy @1 :. HNil
            sumDimShapes = Proxy @'[1, 0] :. Proxy @'[2, 3] :. HNil
        case device of
          Device {deviceType = CPU, deviceIndex = 0} ->
            hfoldrM @IO SumDimSpec () (hproduct sumDimDims (hattach cpu (hproduct almostAllDTypes sumDimShapes)))
          Device {deviceType = CUDA, deviceIndex = 0} ->
            hfoldrM @IO SumDimSpec () (hproduct sumDimDims (hattach cuda0 (hproduct allDTypes sumDimShapes)))
      do
        let shapes = (Proxy :: Proxy ('[] :: [Nat])) :. Proxy @'[1] :. Proxy @'[2, 3] :. HNil
            dispatch spec = case device of
              Device {deviceType = CPU, deviceIndex = 0} ->
                hfoldrM @IO spec () (hattach cpu (hproduct almostAllDTypes shapes))
              Device {deviceType = CUDA, deviceIndex = 0} ->
                hfoldrM @IO spec () (hattach cuda0 (hproduct allDTypes shapes))
        it "min" $ dispatch MinSpec
        it "max" $ dispatch MaxSpec
      it "meanAll" $ do
        let shapes = (Proxy :: Proxy ('[] :: [Nat])) :. Proxy @'[1] :. Proxy @'[2, 3] :. HNil
        case device of
          Device {deviceType = CPU, deviceIndex = 0} ->
            hfoldrM @IO MeanAllSpec () (hattach cpu (hproduct standardFloatingPointDTypes shapes))
          Device {deviceType = CUDA, deviceIndex = 0} ->
            hfoldrM @IO MeanAllSpec () (hattach cuda0 (hproduct standardFloatingPointDTypes shapes))
      it "meanDim" $ do
        let dims = Proxy @0 :. Proxy @1 :. HNil
            shapes = Proxy @'[1, 3] :. Proxy @'[2, 3] :. HNil
        case device of
          Device {deviceType = CPU, deviceIndex = 0} ->
            hfoldrM @IO MeanDimSpec () (hproduct dims (hattach cpu (hproduct standardFloatingPointDTypes shapes)))
          Device {deviceType = CUDA, deviceIndex = 0} ->
            hfoldrM @IO MeanDimSpec () (hproduct dims (hattach cuda0 (hproduct standardFloatingPointDTypes shapes)))
      it "mean" $ do
        let dims = Proxy @0 :. Proxy @1 :. HNil
            keepOrDropDims = Proxy @KeepDim :. Proxy @DropDim :. HNil
            shapes = Proxy @'[1, 12] :. Proxy @'[2, 3] :. HNil
        case device of
          Device {deviceType = CPU, deviceIndex = 0} ->
            hfoldrM @IO MeanSpec () (hproduct (hproduct dims keepOrDropDims) (hattach cpu (hproduct standardFloatingPointDTypes shapes)))
          Device {deviceType = CUDA, deviceIndex = 0} ->
            hfoldrM @IO MeanSpec () (hproduct (hproduct dims keepOrDropDims) (hattach cpu (hproduct standardFloatingPointDTypes shapes)))
      it "medianAll" $ do
        let shapes = (Proxy :: Proxy ('[] :: [Nat])) :. Proxy @'[1] :. Proxy @'[2, 3] :. HNil
        case device of
          Device {deviceType = CPU, deviceIndex = 0} ->
            hfoldrM @IO MedianAllSpec () (hattach cpu (hproduct standardDTypes shapes))
          Device {deviceType = CUDA, deviceIndex = 0} ->
            hfoldrM @IO MedianAllSpec () (hattach cuda0 (hproduct (withHalf standardDTypes) shapes))
      it "medianDim" $ do
        let dims = Proxy @0 :. Proxy @1 :. HNil
            shapes = Proxy @'[1, 17, 1] :. Proxy @'[2, 3] :. HNil
        case device of
          Device {deviceType = CPU, deviceIndex = 0} ->
            hfoldrM @IO MedianDimSpec () (hproduct dims (hattach cpu (hproduct standardDTypes shapes)))
          Device {deviceType = CUDA, deviceIndex = 0} ->
            hfoldrM @IO MedianDimSpec () (hproduct dims (hattach cuda0 (hproduct (withHalf standardDTypes) shapes)))
      it "median" $ do
        let dims = Proxy @0 :. Proxy @1 :. HNil
            keepOrDropDims = Proxy @KeepDim :. Proxy @DropDim :. HNil
            shapes = Proxy @'[2, 13] :. Proxy @'[2, 3] :. Proxy @'[1, 3, 7] :. HNil
        case device of
          Device {deviceType = CPU, deviceIndex = 0} ->
            hfoldrM @IO MedianSpec () (hproduct (hproduct dims keepOrDropDims) (hattach cpu (hproduct standardDTypes shapes)))
          Device {deviceType = CUDA, deviceIndex = 0} ->
            hfoldrM @IO MedianSpec () (hproduct (hproduct dims keepOrDropDims) (hattach cuda0 (hproduct (withHalf standardDTypes) shapes)))
      it "mode" $ do
        let dims = Proxy @0 :. Proxy @1 :. HNil
            keepOrDropDims = Proxy @KeepDim :. Proxy @DropDim :. HNil
            shapes = Proxy @'[2, 13] :. Proxy @'[2, 3] :. Proxy @'[1, 3, 7] :. HNil
        case device of
          Device {deviceType = CPU, deviceIndex = 0} ->
            hfoldrM @IO ModeSpec () (hproduct (hproduct dims keepOrDropDims) (hattach cpu (hproduct standardDTypes shapes)))
          Device {deviceType = CUDA, deviceIndex = 0} ->
            hfoldrM @IO ModeSpec () (hproduct (hproduct dims keepOrDropDims) (hattach cuda0 (hproduct (withHalf standardDTypes) shapes)))

