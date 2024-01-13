{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}

module Torch.Typed.OptimSpec
  ( Torch.Typed.OptimSpec.spec,
  )
where

import Control.Monad (foldM)
import Data.Kind
import Data.Maybe
import Data.Proxy
import GHC.Exts (toList)
import GHC.Generics
import GHC.TypeLits
import Test.Hspec (Spec, describe, it, shouldBe)
import Test.QuickCheck ()
import Torch (ATenTensor)
import Torch.Internal.Class (Castable)
import Torch.Internal.Managed.Type.Context (manual_seed_L)
import Torch.Typed
import Torch.Typed.AuxiliarySpec
import Prelude hiding
  ( cos,
    exp,
    sqrt,
  )

data
  ConvQuadSpec
    (features :: Nat)
    (dtype :: DType)
    (device :: (DeviceType, Nat))
  = ConvQuadSpec
  deriving (Show, Eq)

data
  ConvQuad
    (features :: Nat)
    (dtype :: DType)
    (device :: (DeviceType, Nat))
  where
  ConvQuad ::
    forall features dtype device.
    {w :: Parameter device dtype '[features]} ->
    ConvQuad features dtype device
  deriving (Show, Generic, Parameterized)

instance
  ( RandDTypeIsValid device dtype,
    KnownNat features,
    KnownDType dtype,
    KnownDevice device
  ) =>
  Randomizable
    (ConvQuadSpec features dtype device)
    (ConvQuad features dtype device)
  where
  sample _ = ConvQuad <$> (makeIndependent =<< randn)

convQuad ::
  forall features dtype device.
  (KnownDevice device, DotDTypeIsValid device dtype) =>
  ConvQuad features dtype device ->
  Tensor device dtype '[features, features] ->
  Tensor device dtype '[features] ->
  Tensor device dtype '[]
convQuad ConvQuad {..} a b =
  let w' = toDependent w in mulScalar (0.5 :: Float) (dot w' (mv a w')) - dot b w'

data
  RosenbrockSpec
    (dtype :: DType)
    (device :: (DeviceType, Nat))
  = RosenbrockSpec
  deriving (Show, Eq)

data
  Rosenbrock
    (dtype :: DType)
    (device :: (DeviceType, Nat))
  where
  Rosenbrock ::
    forall dtype device.
    { x :: Parameter device dtype '[1],
      y :: Parameter device dtype '[1]
    } ->
    Rosenbrock dtype device
  deriving (Show, Generic, Parameterized)

instance
  ( RandDTypeIsValid device dtype,
    KnownDType dtype,
    KnownDevice device
  ) =>
  Randomizable
    (RosenbrockSpec dtype device)
    (Rosenbrock dtype device)
  where
  sample _ = Rosenbrock <$> (makeIndependent =<< randn) <*> (makeIndependent =<< randn)

rosenbrock ::
  forall a dtype device.
  (KnownDevice device, Scalar a) =>
  Rosenbrock dtype device ->
  a ->
  a ->
  Tensor device dtype '[]
rosenbrock Rosenbrock {..} a b =
  let x' = toDependent x
      y' = toDependent y
      square c = powScalar (2 :: Int) c
   in reshape $ square (subScalar a x') + mulScalar b (square (y' - square x'))

data
  AckleySpec
    (features :: Nat)
    (dtype :: DType)
    (device :: (DeviceType, Nat))
  = AckleySpec
  deriving (Show, Eq)

data
  Ackley
    (features :: Nat)
    (dtype :: DType)
    (device :: (DeviceType, Nat))
  where
  Ackley ::
    forall features dtype device.
    {pos :: Parameter device dtype '[features]} ->
    Ackley features dtype device
  deriving (Show, Generic, Parameterized)

instance
  ( RandDTypeIsValid device dtype,
    KnownNat features,
    KnownDType dtype,
    KnownDevice device
  ) =>
  Randomizable
    (AckleySpec features dtype device)
    (Ackley features dtype device)
  where
  sample _ =
    Ackley <$> (makeIndependent =<< randn)

ackley ::
  forall features a dtype device.
  ( KnownNat features,
    Scalar a,
    Num a,
    dtype ~ SumDType dtype,
    KnownDType dtype,
    StandardFloatingPointDTypeValidation device dtype,
    SumDTypeIsValid device dtype,
    KnownDevice device
  ) =>
  Ackley features dtype device ->
  a ->
  a ->
  a ->
  Tensor device dtype '[]
ackley Ackley {..} a b c =
  mulScalar (- a) (exp . mulScalar (- b) . sqrt . divScalar d . sumAll $ pos' * pos')
    + addScalar a (exp ones)
    - exp (divScalar d . sumAll . cos . mulScalar c $ pos')
  where
    d = product . shape $ pos'
    pos' = toDependent pos

foldLoop ::
  forall a b m. (Num a, Enum a, Monad m) => b -> a -> (b -> a -> m b) -> m b
foldLoop x count block = foldM block x ([1 .. count] :: [a])

optimize ::
  forall model optim parameters tensors gradients dtype device.
  ( -- gradients ~ GradR parameters
    HasGrad (HList parameters) (HList gradients),
    tensors ~ gradients,
    HMap' ToDependent parameters tensors,
    Castable (HList gradients) [ATenTensor],
    parameters ~ Parameters model,
    Parameterized model,
    Optimizer optim gradients tensors dtype device,
    HMapM' IO MakeIndependent tensors parameters,
    Show model
  ) =>
  model ->
  optim ->
  (model -> Loss device dtype) ->
  LearningRate device dtype ->
  Int ->
  IO (model, optim)
optimize initModel initOptim loss learningRate numIters =
  foldLoop (initModel, initOptim) numIters $
    \(model, optim) _ -> runStep model optim (loss model) learningRate

-- optimize initModel initOptim loss learningRate numIters = do
--   print $ "initial model: " <> show initModel
--   print $ "initial loss:" <> show (loss initModel)
--   (finalModel, finalOptim) <- foldLoop (initModel, initOptim) numIters
--     $ \(model, optim) _ -> runStep model optim (loss model) learningRate
--   print $ "final model: " <> show finalModel
--   print $ "final loss:" <> show (loss finalModel)
--   pure (finalModel, finalOptim)

data OptimConvQuadSpec = GDConvQuadSpec | GDMConvQuadSpec | AdamConvQuadSpec

instance
  ( KnownNat features,
    KnownDType dtype,
    KnownDevice device,
    DotDTypeIsValid device dtype,
    BasicArithmeticDTypeIsValid device dtype,
    StandardFloatingPointDTypeValidation device dtype
  ) =>
  Apply' OptimConvQuadSpec ((Proxy device, (Proxy dtype, Proxy features)), IO ()) (IO ())
  where
  apply' GDConvQuadSpec (_, agg) =
    agg >> do
      manual_seed_L 123
      initModel <- sample (ConvQuadSpec @features @'Float @'( 'CPU, 0))
      let initModel' = toDevice @device @'( 'CPU, 0) . toDType @dtype @'Float $ initModel
          initOptim = mkGD
          a = eyeSquare @features @dtype @device
          b = zeros @'[features] @dtype @device
          loss model = convQuad model a b
          learningRate = 0.1
          numIter = 1000
      (model, _optim) <- optimize initModel' initOptim loss learningRate numIter
      isNonZero (isclose 1e-03 1e-04 False (loss model) zeros) `shouldBe` True
  apply' GDMConvQuadSpec (_, agg) =
    agg >> do
      manual_seed_L 123
      initModel <- sample (ConvQuadSpec @features @'Float @'( 'CPU, 0))
      let initModel' = toDevice @device @'( 'CPU, 0) . toDType @dtype @'Float $ initModel
          initOptim = mkGDM 0.9 (flattenParameters initModel')
          a = eyeSquare @features @dtype @device
          b = zeros @'[features] @dtype @device
          loss model = convQuad model a b
          learningRate = 0.1
          numIter = 1000
      (model, _optim) <- optimize initModel' initOptim loss learningRate numIter
      isNonZero (isclose 1e-03 1e-04 False (loss model) zeros) `shouldBe` True
  apply' AdamConvQuadSpec (_, agg) =
    agg >> do
      manual_seed_L 123
      initModel <- sample (ConvQuadSpec @features @'Float @'( 'CPU, 0))
      let initModel' = toDevice @device @'( 'CPU, 0) . toDType @dtype @'Float $ initModel
          initOptim = mkAdam 0 0.9 0.999 (flattenParameters initModel')
          a = eyeSquare @features @dtype @device
          b = zeros @'[features] @dtype @device
          loss model = convQuad model a b
          learningRate = 0.1
          numIter = 1000
      (model, _optim) <- optimize initModel' initOptim loss learningRate numIter
      isNonZero (isclose 1e-03 1e-04 False (loss model) zeros) `shouldBe` True

data OptimRosenbrockSpec = GDRosenbrockSpec | GDMRosenbrockSpec | AdamRosenbrockSpec

instance
  ( KnownDType dtype,
    KnownDevice device,
    BasicArithmeticDTypeIsValid device dtype,
    StandardFloatingPointDTypeValidation device dtype
  ) =>
  Apply' OptimRosenbrockSpec ((Proxy device, Proxy dtype), IO ()) (IO ())
  where
  apply' GDRosenbrockSpec (_, agg) =
    agg >> do
      manual_seed_L 123
      initModel <- sample (RosenbrockSpec @'Float @'( 'CPU, 0))
      let initModel' = toDevice @device @'( 'CPU, 0) . toDType @dtype @'Float $ initModel
          initOptim = mkGD
          a :: Float = 1.0
          b :: Float = 100.0
          loss model = rosenbrock model a b
          learningRate = 0.002
          numIter = 15000
      (model, _optim) <- optimize initModel' initOptim loss learningRate numIter
      let close = isclose 1e-01 1e-01 False (cat @0 . hmap' ToDependent . flattenParameters $ model) ones
      (toList . Just) close `shouldBe` [True, True]
      isNonZero (isclose 1e-04 1e-04 False (loss model) zeros) `shouldBe` True
  apply' GDMRosenbrockSpec (_, agg) =
    agg >> do
      manual_seed_L 123
      initModel <- sample (RosenbrockSpec @'Float @'( 'CPU, 0))
      let initModel' = toDevice @device @'( 'CPU, 0) . toDType @dtype @'Float $ initModel
          initOptim = mkGDM 0.9 (flattenParameters initModel')
          a :: Float = 1.0
          b :: Float = 100.0
          loss model = rosenbrock model a b
          learningRate = 0.001
          numIter = 10000
      (model, _optim) <- optimize initModel' initOptim loss learningRate numIter
      let close = (isclose 1e-01 1e-01 False (cat @0 . hmap' ToDependent . flattenParameters $ model) ones)
      (toList . Just) close `shouldBe` [True, True]
      isNonZero (isclose 1e-04 1e-04 False (loss model) zeros) `shouldBe` True
  apply' AdamRosenbrockSpec (_, agg) =
    agg >> do
      manual_seed_L 123
      initModel <- sample (RosenbrockSpec @'Float @'( 'CPU, 0))
      let initModel' = toDevice @device @'( 'CPU, 0) . toDType @dtype @'Float $ initModel
          initOptim = mkAdam 0 0.9 0.999 (flattenParameters initModel')
          a :: Float = 1.0
          b :: Float = 100.0
          loss model = rosenbrock model a b
          learningRate = 0.005
          numIter = 5000
      (model, _optim) <- optimize initModel' initOptim loss learningRate numIter
      let close = (isclose 1e-02 1e-02 False (cat @0 . hmap' ToDependent . flattenParameters $ model) ones)
      (toList . Just) close `shouldBe` [True, True]
      isNonZero (isclose 1e-04 1e-04 False (loss model) zeros) `shouldBe` True

data OptimAckleySpec = GDAckleySpec | GDMAckleySpec | AdamAckleySpec

instance
  ( KnownDType dtype,
    KnownDevice device,
    BasicArithmeticDTypeIsValid device dtype,
    StandardFloatingPointDTypeValidation device dtype,
    SumDTypeIsValid device dtype,
    dtype ~ SumDType dtype
  ) =>
  Apply' OptimAckleySpec ((Proxy device, Proxy dtype), IO ()) (IO ())
  where
  apply' GDAckleySpec (_, agg) =
    agg >> do
      manual_seed_L 123
      initModel <- sample (AckleySpec @2 @'Float @'( 'CPU, 0))
      let initModel' = toDevice @device @'( 'CPU, 0) . toDType @dtype @'Float $ initModel
          initOptim = mkGD
          a :: Float = 20.0
          b :: Float = 0.2
          c :: Float = 2 * pi
          loss model = ackley model a b c
          learningRate = 0.00001
          numIter = 5000
      (model, _optim) <- optimize initModel' initOptim loss learningRate numIter
      let finalLoss = loss model
      let close = isclose 1e-03 1e-03 False (cat @0 . hmap' ToDependent . flattenParameters $ model) zeros
      (toList . Just) close `shouldBe` [True, True]
      isNonZero (isclose 1e-04 1e-04 False (loss model) zeros) `shouldBe` True
  apply' GDMAckleySpec (_, agg) =
    agg >> do
      manual_seed_L 123
      initModel <- sample (AckleySpec @2 @'Float @'( 'CPU, 0))
      let initModel' = toDevice @device @'( 'CPU, 0) . toDType @dtype @'Float $ initModel
          initOptim = mkGDM 0.9 (flattenParameters initModel')
          a :: Float = 20.0
          b :: Float = 0.2
          c :: Float = 2 * pi
          loss model = ackley model a b c
          learningRate = 0.000005
          numIter = 5000
      (model, _optim) <- optimize initModel' initOptim loss learningRate numIter
      let finalLoss = loss model
      let close = isclose 1e-03 1e-03 False (cat @0 . hmap' ToDependent . flattenParameters $ model) zeros
      (toList . Just) close `shouldBe` [True, True]
      isNonZero (isclose 1e-04 1e-04 False (loss model) zeros) `shouldBe` True
  apply' AdamAckleySpec (_, agg) =
    agg >> do
      manual_seed_L 123
      initModel <- sample (AckleySpec @2 @'Float @'( 'CPU, 0))
      let initModel' = toDevice @device @'( 'CPU, 0) . toDType @dtype @'Float $ initModel
          initOptim = mkAdam 0 0.9 0.999 (flattenParameters initModel')
          a :: Float = 20.0
          b :: Float = 0.2
          c :: Float = 2 * pi
          loss model = ackley model a b c
          learningRate = 0.0001
          numIter = 2000
      (model, _optim) <- optimize initModel' initOptim loss learningRate numIter
      let finalLoss = loss model
      let close = isclose 1e-03 1e-03 False (cat @0 . hmap' ToDependent . flattenParameters $ model) zeros
      (toList . Just) close `shouldBe` [True, True]
      isNonZero (isclose 1e-04 1e-04 False (loss model) zeros) `shouldBe` True

spec = foldMap spec' availableDevices

spec' :: Device -> Spec
spec' device = describe ("for " <> show device) $ do
  describe "GD" $ do
    it "convex quadratic" $ case device of
      Device {deviceType = CPU, deviceIndex = 0} ->
        hfoldrM @IO GDConvQuadSpec () (hattach cpu (hproduct standardFloatingPointDTypes (Proxy @0 :. Proxy @1 :. Proxy @2 :. HNil)))
      Device {deviceType = CUDA, deviceIndex = 0} ->
        hfoldrM @IO GDConvQuadSpec () (hattach cuda0 (hproduct allFloatingPointDTypes (Proxy @0 :. Proxy @1 :. Proxy @2 :. HNil)))
    it "Rosenbrock" $ case device of
      Device {deviceType = CPU, deviceIndex = 0} ->
        hfoldrM @IO GDRosenbrockSpec () (hattach cpu standardFloatingPointDTypes)
      Device {deviceType = CUDA, deviceIndex = 0} ->
        hfoldrM @IO GDRosenbrockSpec () (hattach cuda0 standardFloatingPointDTypes)
    it "Ackley" $ case device of
      Device {deviceType = CPU, deviceIndex = 0} ->
        hfoldrM @IO GDAckleySpec () (hattach cpu standardFloatingPointDTypes)
      Device {deviceType = CUDA, deviceIndex = 0} ->
        hfoldrM @IO GDAckleySpec () (hattach cuda0 standardFloatingPointDTypes)
  describe "GDM" $ do
    it "convex quadratic" $ case device of
      Device {deviceType = CPU, deviceIndex = 0} ->
        hfoldrM @IO GDMConvQuadSpec () (hattach cpu (hproduct standardFloatingPointDTypes (Proxy @0 :. Proxy @1 :. Proxy @2 :. HNil)))
      Device {deviceType = CUDA, deviceIndex = 0} ->
        hfoldrM @IO GDMConvQuadSpec () (hattach cuda0 (hproduct allFloatingPointDTypes (Proxy @0 :. Proxy @1 :. Proxy @2 :. HNil)))
    it "Rosenbrock" $ case device of
      Device {deviceType = CPU, deviceIndex = 0} ->
        hfoldrM @IO GDMRosenbrockSpec () (hattach cpu standardFloatingPointDTypes)
      Device {deviceType = CUDA, deviceIndex = 0} ->
        hfoldrM @IO GDMRosenbrockSpec () (hattach cuda0 standardFloatingPointDTypes)
    it "Ackley" $ case device of
      Device {deviceType = CPU, deviceIndex = 0} ->
        hfoldrM @IO GDMAckleySpec () (hattach cpu standardFloatingPointDTypes)
      Device {deviceType = CUDA, deviceIndex = 0} ->
        hfoldrM @IO GDMAckleySpec () (hattach cuda0 standardFloatingPointDTypes)
  describe "Adam" $ do
    it "convex quadratic" $ case device of
      Device {deviceType = CPU, deviceIndex = 0} ->
        hfoldrM @IO AdamConvQuadSpec () (hattach cpu (hproduct standardFloatingPointDTypes (Proxy @0 :. Proxy @1 :. Proxy @2 :. HNil)))
      Device {deviceType = CUDA, deviceIndex = 0} ->
        hfoldrM @IO AdamConvQuadSpec () (hattach cuda0 (hproduct allFloatingPointDTypes (Proxy @0 :. Proxy @1 :. Proxy @2 :. HNil)))
    it "Rosenbrock" $ case device of
      Device {deviceType = CPU, deviceIndex = 0} ->
        hfoldrM @IO AdamRosenbrockSpec () (hattach cpu standardFloatingPointDTypes)
      Device {deviceType = CUDA, deviceIndex = 0} ->
        hfoldrM @IO AdamRosenbrockSpec () (hattach cuda0 standardFloatingPointDTypes)
    it "Ackley" $ case device of
      Device {deviceType = CPU, deviceIndex = 0} ->
        hfoldrM @IO AdamAckleySpec () (hattach cpu standardFloatingPointDTypes)
      Device {deviceType = CUDA, deviceIndex = 0} ->
        hfoldrM @IO AdamAckleySpec () (hattach cuda0 standardFloatingPointDTypes)
