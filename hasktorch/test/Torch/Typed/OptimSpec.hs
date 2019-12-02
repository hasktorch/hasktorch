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

module Torch.Typed.OptimSpec
  ( Torch.Typed.OptimSpec.spec
  )
where

import           Prelude
import           Control.Monad                  ( foldM )
import           Control.Exception.Safe
import           Foreign.Storable
import           Data.HList
import           Data.Kind
import           Data.Proxy
import           Data.Reflection
import           GHC.Generics
import           GHC.TypeLits

import           Test.Hspec
import           Test.QuickCheck

import qualified ATen.Cast                     as ATen
import qualified ATen.Class                    as ATen
import qualified ATen.Type                     as ATen
import qualified ATen.Managed.Type.Tensor      as ATen
import qualified ATen.Managed.Type.Context     as ATen
import qualified Torch.Device                  as D
import qualified Torch.DType                   as D
import qualified Torch.Functions               as D
import qualified Torch.Tensor                  as D
import qualified Torch.TensorFactories         as D
import qualified Torch.TensorOptions           as D
import qualified Torch.NN                      as A
import           Torch.Typed.Aux
import           Torch.Typed.Tensor
import           Torch.Typed.Parameter
import           Torch.Typed.Native
import           Torch.Typed.Factories
import           Torch.Typed.NN
import           Torch.Typed.Autograd
import           Torch.Typed.Optim
import           Torch.Typed.Serialize
import           Torch.Typed.AuxSpec
 
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

convQuad
  :: forall features dtype device
   . DotDTypeIsValid device dtype
  => ConvQuad features dtype device
  -> Tensor device dtype '[features, features]
  -> Tensor device dtype '[features]
  -> Tensor device dtype '[]
convQuad ConvQuad {..} a b =
  let w' = toDependent w
  in  cmul (0.5 :: Float) (dot w' (mv a w')) - dot b w'

instance
  ( RandDTypeIsValid device dtype
  , KnownNat features
  , KnownDType dtype
  , KnownDevice device
  ) => A.Randomizable (ConvQuadSpec features dtype device)
                      (ConvQuad     features dtype device)
 where
  sample _ = ConvQuad <$> (makeIndependent =<< randn)

foldLoop
  :: forall a b m . (Num a, Enum a, Monad m) => b -> a -> (b -> a -> m b) -> m b
foldLoop x count block = foldM block x ([1 .. count] :: [a])

optimize
  :: forall model optim parameters tensors gradients dtype device
   . ( gradients ~ GradR parameters dtype device
     , tensors ~ gradients
     , HMap' ToDependent parameters tensors
     , ATen.Castable (HList gradients) [D.ATenTensor]
     , Parameterized model parameters
     , Optimizer optim gradients tensors dtype device
     , HMapM' IO MakeIndependent tensors parameters
     )
  => model
  -> optim
  -> (model -> Loss device dtype)
  -> LearningRate device dtype
  -> Int
  -> IO (model, optim)
optimize initModel initOptim loss learningRate numIters =
  foldLoop (initModel, initOptim) numIters $ \(model, optim) _ ->
    runStep model optim (loss model) learningRate

data OptimSpec = OptimSpec

instance
  ( KnownNat features
  , KnownDType dtype
  , KnownDevice device
  , RandDTypeIsValid device dtype
  , DotDTypeIsValid device dtype
  , BasicArithmeticDTypeIsValid device dtype
  , StandardFloatingPointDTypeValidation device dtype
  ) => Apply OptimSpec
             (Proxy device, (Proxy dtype, Proxy features))
             (() -> IO ())
 where
  apply OptimSpec _ _ = do
    ATen.manual_seed_L 123
    initModel <- A.sample (ConvQuadSpec @features @dtype @device)
    let initOptim = mkAdam 0 0.9 0.999 (flattenParameters initModel)
        a = eyeSquare @features @dtype @device
        b = zeros @'[features] @dtype @device
        loss model = convQuad model a b
        learningRate = 0.1
        numIter = 10000
    (model, _optim) <- optimize initModel initOptim loss learningRate numIter
    let finalLoss = loss model
    print finalLoss
    isNonZero (isclose 1e-03 1e-04 False finalLoss zeros) `shouldBe` True

spec = foldMap spec' availableDevices

spec' :: D.Device -> Spec
spec' device = 
  describe ("for " <> show device) $ do
    describe "Adam" $ do
      return ()
      it "convex quadratic" $ case device of
        D.Device { D.deviceType = D.CPU,  D.deviceIndex = 0 } ->
          hfoldrM @IO OptimSpec () (hattach cpu   (hCartesianProduct standardFloatingPointDTypes (Proxy @0 :. Proxy @1 :. Proxy @2 :. HNil)))
        D.Device { D.deviceType = D.CUDA, D.deviceIndex = 0 } ->
          hfoldrM @IO OptimSpec () (hattach cuda0 (hCartesianProduct allFloatingPointDTypes      (Proxy @0 :. Proxy @1 :. Proxy @2 :. HNil)))
