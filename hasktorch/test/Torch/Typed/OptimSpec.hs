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
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE OverloadedLists #-}

module Torch.Typed.OptimSpec
  ( Torch.Typed.OptimSpec.spec
  )
where

import           Prelude                 hiding ( exp
                                                , cos
                                                , sqrt
                                                )
import           Control.Monad                  ( foldM )
import           Control.Exception.Safe
import           Foreign.Storable
import           Torch.HList
import           Data.Kind
import           Data.Proxy
import           Data.Maybe
import           Data.Reflection
import           GHC.Generics
import           GHC.TypeLits
import           GHC.Exts

import           Test.Hspec
import           Test.QuickCheck

import qualified Torch.Internal.Cast                     as ATen
import qualified Torch.Internal.Class                    as ATen
import qualified Torch.Internal.Type                     as ATen
import qualified Torch.Internal.Managed.Type.Tensor      as ATen
import qualified Torch.Internal.Managed.Type.Context     as ATen
import qualified Torch.Device                  as D
import qualified Torch.DType                   as D
import qualified Torch.Functional              as D
import qualified Torch.Tensor                  as D
import qualified Torch.TensorFactories         as D
import qualified Torch.TensorOptions           as D
import qualified Torch.Scalar                  as D
import qualified Torch.NN                      as A
import           Torch.Typed.Aux
import           Torch.Typed.Tensor
import           Torch.Typed.Parameter
import           Torch.Typed.Functional
import           Torch.Typed.Factories
import           Torch.Typed.NN
import           Torch.Typed.Autograd
import           Torch.Typed.Optim
import           Torch.Typed.Serialize
import           Torch.Typed.AuxSpec
import           Torch.Typed.Device
import           Torch.Typed.DType

data ConvQuadSpec (features :: Nat)
                  (dtype :: D.DType)
                  (device :: (D.DeviceType, Nat))
  = ConvQuadSpec deriving (Show, Eq)

data ConvQuad (features :: Nat)
              (dtype :: D.DType)
              (device :: (D.DeviceType, Nat))
 where
  ConvQuad
    :: forall features dtype device
     . { w :: Parameter device dtype '[features] }
    -> ConvQuad features dtype device
 deriving (Show, Generic)

instance
  ( RandDTypeIsValid device dtype
  , KnownNat features
  , KnownDType dtype
  , KnownDevice device
  ) => A.Randomizable (ConvQuadSpec features dtype device)
                      (ConvQuad     features dtype device)
 where
  sample _ = ConvQuad <$> (makeIndependent =<< randn)

convQuad
  :: forall features dtype device
   . DotDTypeIsValid device dtype
  => ConvQuad features dtype device
  -> Tensor device dtype '[features, features]
  -> Tensor device dtype '[features]
  -> Tensor device dtype '[]
convQuad ConvQuad {..} a b =
  let w' = Torch.Typed.Parameter.toDependent w in cmul (0.5 :: Float) (dot w' (mv a w')) - dot b w'

data RosenbrockSpec (dtype :: D.DType)
                    (device :: (D.DeviceType, Nat))
  = RosenbrockSpec deriving (Show, Eq)

data Rosenbrock (dtype :: D.DType)
                (device :: (D.DeviceType, Nat))
 where
  Rosenbrock
    :: forall dtype device
     . { x :: Parameter device dtype '[1]
       , y :: Parameter device dtype '[1]
       }
    -> Rosenbrock dtype device
 deriving (Show, Generic)

instance
  ( RandDTypeIsValid device dtype
  , KnownDType dtype
  , KnownDevice device
  ) => A.Randomizable (RosenbrockSpec dtype device)
                      (Rosenbrock     dtype device)
 where
  sample _ = Rosenbrock <$> (makeIndependent =<< randn) <*> (makeIndependent =<< randn)

rosenbrock
  :: forall a dtype device
   . D.Scalar a
  => Rosenbrock dtype device
  -> a
  -> a
  -> Tensor device dtype '[]
rosenbrock Rosenbrock {..} a b =
  let x' = Torch.Typed.Parameter.toDependent x
      y' = Torch.Typed.Parameter.toDependent y
      square c = pow (2 :: Int) c
  in  reshape $ square (csub a x') + cmul b (square (y' - square x'))

data AckleySpec (features :: Nat)
                (dtype :: D.DType)
                (device :: (D.DeviceType, Nat))
  = AckleySpec deriving (Show, Eq)

data Ackley (features :: Nat)
            (dtype :: D.DType)
            (device :: (D.DeviceType, Nat))
 where
  Ackley
    :: forall features dtype device
     . { pos :: Parameter device dtype '[features] }
    -> Ackley features dtype device
 deriving (Show, Generic)

instance
  ( RandDTypeIsValid device dtype
  , KnownNat features
  , KnownDType dtype
  , KnownDevice device
  ) => A.Randomizable (AckleySpec features dtype device)
                      (Ackley     features dtype device)
 where
  sample _ =
    Ackley <$> (makeIndependent =<< randn)

ackley
  :: forall features a dtype device
   . ( KnownNat features
     , D.Scalar a
     , Num a
     , dtype ~ SumDType dtype
     , KnownDType dtype
     , StandardFloatingPointDTypeValidation device dtype
     , SumDTypeIsValid device dtype
     , KnownDevice device
     )
  => Ackley features dtype device
  -> a
  -> a
  -> a
  -> Tensor device dtype '[]
ackley Ackley {..} a b c =
  cmul (-a) (exp . cmul (-b) . sqrt . cdiv d . sumAll $ pos' * pos')
    + cadd a (exp ones)
    - exp (cdiv d . sumAll . cos . cmul c $ pos')
 where
  d    = product . shape $ pos'
  pos' = Torch.Typed.Parameter.toDependent pos

foldLoop
  :: forall a b m . (Num a, Enum a, Monad m) => b -> a -> (b -> a -> m b) -> m b
foldLoop x count block = foldM block x ([1 .. count] :: [a])

optimize
  :: forall model optim parameters tensors gradients dtype device
   . ( -- gradients ~ GradR parameters
       HasGrad (HList parameters) (HList gradients)
     , tensors ~ gradients
     , HMap' ToDependent parameters tensors
     , ATen.Castable (HList gradients) [D.ATenTensor]
     , Parameterized model parameters
     , Optimizer optim gradients tensors dtype device
     , HMapM' IO MakeIndependent tensors parameters
     , Show model
     )
  => model
  -> optim
  -> (model -> Loss device dtype)
  -> LearningRate device dtype
  -> Int
  -> IO (model, optim)
optimize initModel initOptim loss learningRate numIters =
  foldLoop (initModel, initOptim) numIters
    $ \(model, optim) _ -> runStep model optim (loss model) learningRate
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
  ( KnownNat features
  , KnownDType dtype
  , KnownDevice device
  , DotDTypeIsValid device dtype
  , BasicArithmeticDTypeIsValid device dtype
  , StandardFloatingPointDTypeValidation device dtype
  ) => Apply OptimConvQuadSpec
             (Proxy device, (Proxy dtype, Proxy features))
             (() -> IO ())
 where
  apply GDConvQuadSpec _ _ = do
    ATen.manual_seed_L 123
    initModel <- A.sample (ConvQuadSpec @features @'D.Float @'( 'D.CPU, 0))
    let initModel'   = Torch.Typed.Device.toDevice @device @'( 'D.CPU, 0) . Torch.Typed.DType.toDType @dtype @'D.Float $ initModel
        initOptim    = mkGD
        a            = eyeSquare @features @dtype @device
        b            = zeros @'[features] @dtype @device
        loss model   = convQuad model a b
        learningRate = 0.1
        numIter      = 1000
    (model, _optim) <- optimize initModel' initOptim loss learningRate numIter
    isNonZero (isclose 1e-03 1e-04 False (loss model) zeros) `shouldBe` True
  apply GDMConvQuadSpec _ _ = do
    ATen.manual_seed_L 123
    initModel <- A.sample (ConvQuadSpec @features @'D.Float @'( 'D.CPU, 0))
    let initModel'   = Torch.Typed.Device.toDevice @device @'( 'D.CPU, 0) . Torch.Typed.DType.toDType @dtype @'D.Float $ initModel
        initOptim    = mkGDM 0.9 (flattenParameters initModel')
        a            = eyeSquare @features @dtype @device
        b            = zeros @'[features] @dtype @device
        loss model   = convQuad model a b
        learningRate = 0.1
        numIter      = 1000
    (model, _optim) <- optimize initModel' initOptim loss learningRate numIter
    isNonZero (isclose 1e-03 1e-04 False (loss model) zeros) `shouldBe` True
  apply AdamConvQuadSpec _ _ = do
    ATen.manual_seed_L 123
    initModel <- A.sample (ConvQuadSpec @features @'D.Float @'( 'D.CPU, 0))
    let initModel'   = Torch.Typed.Device.toDevice @device @'( 'D.CPU, 0) . Torch.Typed.DType.toDType @dtype @'D.Float $ initModel
        initOptim    = mkAdam 0 0.9 0.999 (flattenParameters initModel')
        a            = eyeSquare @features @dtype @device
        b            = zeros @'[features] @dtype @device
        loss model   = convQuad model a b
        learningRate = 0.1
        numIter      = 1000
    (model, _optim) <- optimize initModel' initOptim loss learningRate numIter
    isNonZero (isclose 1e-03 1e-04 False (loss model) zeros) `shouldBe` True

data OptimRosenbrockSpec = GDRosenbrockSpec | GDMRosenbrockSpec | AdamRosenbrockSpec

instance
  ( KnownDType dtype
  , KnownDevice device
  , BasicArithmeticDTypeIsValid device dtype
  , StandardFloatingPointDTypeValidation device dtype
  ) => Apply OptimRosenbrockSpec
             (Proxy device, Proxy dtype)
             (() -> IO ())
 where
  apply GDRosenbrockSpec _ _ = do
    ATen.manual_seed_L 123
    initModel <- A.sample (RosenbrockSpec @'D.Float @'( 'D.CPU, 0))
    let initModel'   = Torch.Typed.Device.toDevice @device @'( 'D.CPU, 0) . Torch.Typed.DType.toDType @dtype @'D.Float $ initModel
        initOptim    = mkGD
        a :: Float   = 1.0
        b :: Float   = 100.0
        loss model   = rosenbrock model a b
        learningRate = 0.002
        numIter      = 15000
    (model, _optim) <- optimize initModel' initOptim loss learningRate numIter
    (toList . Just . Torch.Typed.Tensor.toDevice @'( 'D.CPU, 0)) (isclose 1e-04 1e-04 False (cat @0 . hmap' ToDependent . flattenParameters $ model) ones) `shouldBe` [True, True]
    isNonZero (isclose 1e-04 1e-04 False (loss model) zeros) `shouldBe` True
  apply GDMRosenbrockSpec _ _ = do
    ATen.manual_seed_L 123
    initModel <- A.sample (RosenbrockSpec @'D.Float @'( 'D.CPU, 0))
    let initModel'   = Torch.Typed.Device.toDevice @device @'( 'D.CPU, 0) . Torch.Typed.DType.toDType @dtype @'D.Float $ initModel
        initOptim    = mkGDM 0.9 (flattenParameters initModel')
        a :: Float   = 1.0
        b :: Float   = 100.0
        loss model   = rosenbrock model a b
        learningRate = 0.001
        numIter      = 10000
    (model, _optim) <- optimize initModel' initOptim loss learningRate numIter
    (toList . Just . Torch.Typed.Tensor.toDevice @'( 'D.CPU, 0)) (isclose 1e-04 1e-04 False (cat @0 . hmap' ToDependent . flattenParameters $ model) ones) `shouldBe` [True, True]
    isNonZero (isclose 1e-04 1e-04 False (loss model) zeros) `shouldBe` True
  apply AdamRosenbrockSpec _ _ = do
    ATen.manual_seed_L 123
    initModel <- A.sample (RosenbrockSpec @'D.Float @'( 'D.CPU, 0))
    let initModel'   = Torch.Typed.Device.toDevice @device @'( 'D.CPU, 0) . Torch.Typed.DType.toDType @dtype @'D.Float $ initModel
        initOptim    = mkAdam 0 0.9 0.999 (flattenParameters initModel')
        a :: Float   = 1.0
        b :: Float   = 100.0
        loss model   = rosenbrock model a b
        learningRate = 0.005
        numIter      = 5000
    (model, _optim) <- optimize initModel' initOptim loss learningRate numIter
    (toList . Just . Torch.Typed.Tensor.toDevice @'( 'D.CPU, 0)) (isclose 1e-04 1e-04 False (cat @0 . hmap' ToDependent . flattenParameters $ model) ones) `shouldBe` [True, True]
    isNonZero (isclose 1e-04 1e-04 False (loss model) zeros) `shouldBe` True

data OptimAckleySpec = GDAckleySpec | GDMAckleySpec | AdamAckleySpec

instance
  ( KnownDType dtype
  , KnownDevice device
  , BasicArithmeticDTypeIsValid device dtype
  , StandardFloatingPointDTypeValidation device dtype
  , SumDTypeIsValid device dtype
  , dtype ~ SumDType dtype
  ) => Apply OptimAckleySpec
             (Proxy device, Proxy dtype)
             (() -> IO ())
 where
  apply GDAckleySpec _ _ = do
    ATen.manual_seed_L 123
    initModel <- A.sample (AckleySpec @2 @'D.Float @'( 'D.CPU, 0))
    let initModel'   = Torch.Typed.Device.toDevice @device @'( 'D.CPU, 0) . Torch.Typed.DType.toDType @dtype @'D.Float $ initModel
        initOptim    = mkGD
        a :: Float   = 20.0
        b :: Float   = 0.2
        c :: Float   = 2 * pi
        loss model   = ackley model a b c
        learningRate = 0.00001
        numIter      = 5000
    (model, _optim) <- optimize initModel' initOptim loss learningRate numIter
    let finalLoss = loss model
    (toList . Just . Torch.Typed.Tensor.toDevice @'( 'D.CPU, 0)) (isclose 1e-04 1e-04 False (cat @0 . hmap' ToDependent . flattenParameters $ model) zeros) `shouldBe` [True, True]
    isNonZero (isclose 1e-04 1e-04 False (loss model) zeros) `shouldBe` True
  apply GDMAckleySpec _ _ = do
    ATen.manual_seed_L 123
    initModel <- A.sample (AckleySpec @2 @'D.Float @'( 'D.CPU, 0))
    let initModel'   = Torch.Typed.Device.toDevice @device @'( 'D.CPU, 0) . Torch.Typed.DType.toDType @dtype @'D.Float $ initModel
        initOptim    = mkGDM 0.9 (flattenParameters initModel')
        a :: Float   = 20.0
        b :: Float   = 0.2
        c :: Float   = 2 * pi
        loss model   = ackley model a b c
        learningRate = 0.000005
        numIter      = 5000
    (model, _optim) <- optimize initModel' initOptim loss learningRate numIter
    let finalLoss = loss model
    (toList . Just . Torch.Typed.Tensor.toDevice @'( 'D.CPU, 0)) (isclose 1e-04 1e-04 False (cat @0 . hmap' ToDependent . flattenParameters $ model) zeros) `shouldBe` [True, True]
    isNonZero (isclose 1e-04 1e-04 False (loss model) zeros) `shouldBe` True
  apply AdamAckleySpec _ _ = do
    ATen.manual_seed_L 123
    initModel <- A.sample (AckleySpec @2 @'D.Float @'( 'D.CPU, 0))
    let initModel'   = Torch.Typed.Device.toDevice @device @'( 'D.CPU, 0) . Torch.Typed.DType.toDType @dtype @'D.Float $ initModel
        initOptim    = mkAdam 0 0.9 0.999 (flattenParameters initModel')
        a :: Float   = 20.0
        b :: Float   = 0.2
        c :: Float   = 2 * pi
        loss model   = ackley model a b c
        learningRate = 0.0001
        numIter      = 2000
    (model, _optim) <- optimize initModel' initOptim loss learningRate numIter
    let finalLoss = loss model
    (toList . Just . Torch.Typed.Tensor.toDevice @'( 'D.CPU, 0)) (isclose 1e-04 1e-04 False (cat @0 . hmap' ToDependent . flattenParameters $ model) zeros) `shouldBe` [True, True]
    isNonZero (isclose 1e-04 1e-04 False (loss model) zeros) `shouldBe` True

spec = foldMap spec' availableDevices

spec' :: D.Device -> Spec
spec' device = describe ("for " <> show device) $ do
  describe "GD" $ do
    it "convex quadratic" $ case device of
      D.Device { D.deviceType = D.CPU, D.deviceIndex = 0 } ->
        hfoldrM @IO GDConvQuadSpec () (hattach cpu   (hproduct standardFloatingPointDTypes (Proxy @0 :. Proxy @1 :. Proxy @2 :. HNil)))
      D.Device { D.deviceType = D.CUDA, D.deviceIndex = 0 } ->
        hfoldrM @IO GDConvQuadSpec () (hattach cuda0 (hproduct allFloatingPointDTypes      (Proxy @0 :. Proxy @1 :. Proxy @2 :. HNil)))
    it "Rosenbrock" $ case device of
      D.Device { D.deviceType = D.CPU, D.deviceIndex = 0 } ->
        hfoldrM @IO GDRosenbrockSpec () (hattach cpu   standardFloatingPointDTypes)
      D.Device { D.deviceType = D.CUDA, D.deviceIndex = 0 } ->
        hfoldrM @IO GDRosenbrockSpec () (hattach cuda0 standardFloatingPointDTypes)
    it "Ackley" $ case device of
      D.Device { D.deviceType = D.CPU, D.deviceIndex = 0 } ->
        hfoldrM @IO GDAckleySpec () (hattach cpu   standardFloatingPointDTypes)
      D.Device { D.deviceType = D.CUDA, D.deviceIndex = 0 } ->
        hfoldrM @IO GDAckleySpec () (hattach cuda0 standardFloatingPointDTypes)
  describe "GDM" $ do
    it "convex quadratic" $ case device of
      D.Device { D.deviceType = D.CPU, D.deviceIndex = 0 } ->
        hfoldrM @IO GDMConvQuadSpec () (hattach cpu   (hproduct standardFloatingPointDTypes (Proxy @0 :. Proxy @1 :. Proxy @2 :. HNil)))
      D.Device { D.deviceType = D.CUDA, D.deviceIndex = 0 } ->
        hfoldrM @IO GDMConvQuadSpec () (hattach cuda0 (hproduct allFloatingPointDTypes      (Proxy @0 :. Proxy @1 :. Proxy @2 :. HNil)))
    it "Rosenbrock" $ case device of
      D.Device { D.deviceType = D.CPU, D.deviceIndex = 0 } ->
        hfoldrM @IO GDMRosenbrockSpec () (hattach cpu   standardFloatingPointDTypes)
      D.Device { D.deviceType = D.CUDA, D.deviceIndex = 0 } ->
        hfoldrM @IO GDMRosenbrockSpec () (hattach cuda0 standardFloatingPointDTypes)
    it "Ackley" $ case device of
      D.Device { D.deviceType = D.CPU, D.deviceIndex = 0 } ->
        hfoldrM @IO GDMAckleySpec () (hattach cpu   standardFloatingPointDTypes)
      D.Device { D.deviceType = D.CUDA, D.deviceIndex = 0 } ->
        hfoldrM @IO GDMAckleySpec () (hattach cuda0 standardFloatingPointDTypes)
  describe "Adam" $ do
    it "convex quadratic" $ case device of
      D.Device { D.deviceType = D.CPU, D.deviceIndex = 0 } ->
        hfoldrM @IO AdamConvQuadSpec () (hattach cpu   (hproduct standardFloatingPointDTypes (Proxy @0 :. Proxy @1 :. Proxy @2 :. HNil)))
      D.Device { D.deviceType = D.CUDA, D.deviceIndex = 0 } ->
        hfoldrM @IO AdamConvQuadSpec () (hattach cuda0 (hproduct allFloatingPointDTypes      (Proxy @0 :. Proxy @1 :. Proxy @2 :. HNil)))
    it "Rosenbrock" $ case device of
      D.Device { D.deviceType = D.CPU, D.deviceIndex = 0 } ->
        hfoldrM @IO AdamRosenbrockSpec () (hattach cpu   standardFloatingPointDTypes)
      D.Device { D.deviceType = D.CUDA, D.deviceIndex = 0 } ->
        hfoldrM @IO AdamRosenbrockSpec () (hattach cuda0 standardFloatingPointDTypes)
    it "Ackley" $ case device of
      D.Device { D.deviceType = D.CPU, D.deviceIndex = 0 } ->
        hfoldrM @IO AdamAckleySpec () (hattach cpu   standardFloatingPointDTypes)
      D.Device { D.deviceType = D.CUDA, D.deviceIndex = 0 } ->
        hfoldrM @IO AdamAckleySpec () (hattach cuda0 standardFloatingPointDTypes)
