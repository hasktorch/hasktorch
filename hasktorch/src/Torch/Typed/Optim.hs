{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE NoStarIsType #-}

module Torch.Typed.Optim where

import Control.Monad.State
import Data.Kind
import System.Mem (performGC)
import qualified Torch.DType as D
import qualified Torch.Device as D
import Torch.HList
import qualified Torch.Internal.Cast as ATen
import qualified Torch.Internal.Class as ATen
import Torch.Internal.GC (mallocTrim)
import qualified Torch.Tensor as D
import Torch.Typed.Autograd
import Torch.Typed.Auxiliary
import Torch.Typed.Factories
import Torch.Typed.Functional
import Torch.Typed.Parameter
import Torch.Typed.Tensor
import Prelude hiding (div, sqrt)

type LearningRate device dtype = Tensor device dtype '[]

type Loss device dtype = Tensor device dtype '[]

data ZerosLike = ZerosLike

instance
  ( parameter ~ Parameter device dtype shape,
    momentum ~ Tensor device dtype shape,
    TensorOptions shape dtype device
  ) =>
  Apply' ZerosLike parameter momentum
  where
  apply' _ _ = zeros

class Optimizer optim gradients tensors dtype device where
  step ::
    LearningRate device dtype ->
    HList gradients ->
    HList tensors ->
    optim ->
    (HList tensors, optim)

runStep ::
  forall model optim parameters gradients tensors dtype device.
  ( Parameterized model,
    parameters ~ Parameters model,
    HasGrad (HList parameters) (HList gradients),
    tensors ~ gradients,
    HMap' ToDependent parameters tensors,
    ATen.Castable (HList gradients) [D.ATenTensor],
    Optimizer optim gradients tensors dtype device,
    HMapM' IO MakeIndependent tensors parameters
  ) =>
  model ->
  optim ->
  Loss device dtype ->
  LearningRate device dtype ->
  IO (model, optim)
runStep model optim loss learningRate = do
  performGC
  mallocTrim 0
  let parameters = flattenParameters model
      gradients = grad loss parameters
      tensors = hmap' ToDependent parameters
      (tensors', optim') = step learningRate gradients tensors optim
  parameters' <- hmapM' MakeIndependent tensors'
  let model' = replaceParameters model parameters'
  return (model', optim')

runStep' ::
  forall model optim parameters gradients tensors dtype device.
  ( Parameterized model,
    parameters ~ Parameters model,
    tensors ~ gradients,
    HMap' ToDependent parameters tensors,
    Optimizer optim gradients tensors dtype device,
    HMapM' IO MakeIndependent tensors parameters
  ) =>
  model ->
  optim ->
  LearningRate device dtype ->
  HList gradients ->
  IO (model, optim)
runStep' model optim learningRate gradients = do
  performGC
  mallocTrim 0
  let parameters = flattenParameters model
      tensors = hmap' ToDependent parameters
      (tensors', optim') = step learningRate gradients tensors optim
  parameters' <- hmapM' MakeIndependent tensors'
  let model' = replaceParameters model parameters'
  return (model', optim')

--
-- Gradient Descent (GD)
--

-- | Dummy state representation for GD Optimizer
data GD = GD

mkGD :: GD
mkGD = GD

newtype GDStep device dtype = GDStep (LearningRate device dtype)

instance
  ( parameter ~ Tensor device dtype shape,
    gradient ~ Tensor device dtype shape,
    shape ~ Broadcast '[] shape,
    BasicArithmeticDTypeIsValid device dtype,
    KnownDevice device
  ) =>
  Apply' (GDStep device dtype) (parameter, gradient) parameter
  where
  apply' (GDStep learningRate) (parameter, gradient) =
    parameter - mul learningRate gradient

-- | Gradient descent step with a dummy state variable
gd ::
  forall gradients tensors dtype device.
  HZipWith (GDStep device dtype) tensors gradients tensors =>
  LearningRate device dtype ->
  HList gradients ->
  HList tensors ->
  GD ->
  (HList tensors, GD)
gd learningRate gradients parameters gd =
  let step = hzipWith (GDStep learningRate) parameters gradients in (step, gd)

instance
  ( HZipWith (GDStep device dtype) tensors gradients tensors
  ) =>
  Optimizer GD gradients tensors dtype device
  where
  step = gd

instance Parameterized GD where
  type Parameters GD = '[]
  flattenParameters _ = HNil
  replaceParameters = const

--
-- Gradient Descent with Momentum (GDM)
--

-- | State representation for GDM Optimizer
data GDM (momenta :: [Type]) = GDM
  { beta :: Float, -- moment forgetting factor
    momenta :: HList momenta -- momenta
  }

mkGDM ::
  forall parameters momenta.
  (HMap' ZerosLike parameters momenta) =>
  Float ->
  HList parameters ->
  GDM momenta
mkGDM beta parameters = GDM beta (hmap' ZerosLike parameters)

data GDMStep device dtype = GDMStep Float (LearningRate device dtype)

instance
  ( parameter ~ Tensor device dtype shape,
    gradient ~ Tensor device dtype shape,
    momentum ~ Tensor device dtype shape,
    shape ~ Broadcast '[] shape,
    KnownDevice device,
    BasicArithmeticDTypeIsValid device dtype
  ) =>
  Apply' (GDMStep device dtype) (parameter, gradient, momentum) (parameter, momentum)
  where
  apply' (GDMStep beta learningRate) (parameter, gradient, momentum) =
    let momentum' = mulScalar beta momentum + gradient
        parameter' = parameter - mul learningRate momentum'
     in (parameter', momentum')

-- | gradient descent with momentum step
gdm ::
  forall gradients tensors momenta gdmStep dtype device.
  ( HZipWith3 (GDMStep device dtype) tensors gradients momenta gdmStep,
    HMap' AFst gdmStep tensors,
    HMap' ASnd gdmStep momenta
  ) =>
  -- | learning rate
  LearningRate device dtype ->
  -- | model parameter gradient tensors
  HList gradients ->
  -- | model parameter tensors
  HList tensors ->
  -- | beta and model parameter momentum tensors
  GDM momenta ->
  -- | returns updated parameters and momenta
  (HList tensors, GDM momenta)
gdm learningRate gradients parameters (GDM beta momenta) =
  let step = hzipWith3 (GDMStep beta learningRate) parameters gradients momenta
   in (hmap' AFst step, GDM beta (hmap' ASnd step))

instance
  ( HZipWith3 (GDMStep device dtype) tensors gradients momenta gdmStep,
    HMap' AFst gdmStep tensors,
    HMap' ASnd gdmStep momenta
  ) =>
  Optimizer (GDM momenta) gradients tensors dtype device
  where
  step = gdm

instance Parameterized (GDM momenta) where
  type Parameters (GDM momenta) = momenta
  flattenParameters GDM {..} = momenta
  replaceParameters gdm momenta = gdm {momenta = momenta}

--
-- Adam
-- https://arxiv.org/pdf/1412.6980.pdf
--

type AdamIter = Tensor '( 'D.CPU, 0) 'D.Int64 '[]

-- | State representation for Adam Optimizer
data Adam (momenta :: [Type]) = Adam
  { iter :: AdamIter, -- iteration
    beta1 :: Float, -- 1st moment forgetting factor
    beta2 :: Float, -- 2nd moment forgetting factor
    momenta1 :: HList momenta, -- 1st momenta
    momenta2 :: HList momenta -- 2nd momenta
  }

mkAdam ::
  forall parameters momenta.
  (HMap' ZerosLike parameters momenta) =>
  AdamIter ->
  Float ->
  Float ->
  HList parameters ->
  Adam momenta
mkAdam iter beta1 beta2 parameters =
  Adam
    iter
    beta1
    beta2
    (hmap' ZerosLike parameters)
    (hmap' ZerosLike parameters)

newtype AdamMomentum1Update = AdamMomentum1Update Float

-- | decaying average of the first momenta
instance
  ( gradient ~ Tensor device dtype shape,
    momentum1 ~ Tensor device dtype shape,
    KnownDevice device
  ) =>
  Apply' AdamMomentum1Update (momentum1, gradient) momentum1
  where
  apply' (AdamMomentum1Update beta1) (momentum1, gradient) =
    mulScalar beta1 momentum1 + mulScalar (1 - beta1) gradient

newtype AdamMomentum2Update = AdamMomentum2Update Float

-- | decaying average of the second momenta
instance
  ( gradient ~ Tensor device dtype shape,
    momentum2 ~ Tensor device dtype shape,
    shape ~ Broadcast shape shape,
    KnownDevice device,
    BasicArithmeticDTypeIsValid device dtype
  ) =>
  Apply' AdamMomentum2Update (momentum2, gradient) momentum2
  where
  apply' (AdamMomentum2Update beta2) (momentum2, gradient) =
    mulScalar beta2 momentum2 + mulScalar (1 - beta2) (mul gradient gradient)

data AdamBiasAdjustment = AdamBiasAdjustment AdamIter Float

-- | bias adjustment
instance
  ( momentum ~ Tensor device dtype shape,
    KnownDevice device,
    KnownDType dtype,
    shape ~ Reverse (Reverse shape),
    BasicArithmeticDTypeIsValid device dtype
  ) =>
  Apply' AdamBiasAdjustment momentum momentum
  where
  apply' (AdamBiasAdjustment iter beta) momentum =
    let iter' = toDevice @device @'( 'D.CPU, 0) . toDType @dtype @'D.Int64 $ iter + 1
        beta' = full @'[] @dtype @device beta
     in momentum `div` (1 - pow iter' beta')

data AdamParameterUpdate device dtype = AdamParameterUpdate Float (LearningRate device dtype)

-- | parameter update
instance
  ( parameter ~ Tensor device dtype shape,
    momentum ~ Tensor device dtype shape,
    shape ~ Broadcast '[] shape,
    KnownDevice device,
    BasicArithmeticDTypeIsValid device dtype,
    StandardFloatingPointDTypeValidation device dtype
  ) =>
  Apply'
    (AdamParameterUpdate device dtype)
    (parameter, momentum, momentum)
    parameter
  where
  apply'
    (AdamParameterUpdate eps learningRate)
    (parameter, biasAdjustedMomentum1, biasAdjustedMomentum2) =
      parameter - mul learningRate biasAdjustedMomentum1
        / addScalar eps (sqrt biasAdjustedMomentum2)

-- | Adam step
adam ::
  forall gradients tensors momenta adamStep dtype device.
  ( HZipWith AdamMomentum1Update momenta gradients momenta,
    HZipWith AdamMomentum2Update momenta gradients momenta,
    HMap' AdamBiasAdjustment momenta momenta,
    HZipWith3 (AdamParameterUpdate device dtype) tensors momenta momenta tensors
  ) =>
  -- | learning rate
  LearningRate device dtype ->
  -- | model parameter gradient tensors
  HList gradients ->
  -- | model parameter tensors
  HList tensors ->
  -- | adam parameters - beta1, beta2, momenta1, momenta2, iteration
  Adam momenta ->
  -- | returns new parameters + updated adam parameters
  (HList tensors, Adam momenta)
adam learningRate gradients parameters Adam {..} =
  (parameters', Adam (iter + 1) beta1 beta2 momenta1' momenta2')
  where
    momenta1' = hzipWith (AdamMomentum1Update beta1) momenta1 gradients
    momenta2' = hzipWith (AdamMomentum2Update beta2) momenta2 gradients
    biasAdjustedMomenta1 = hmap' (AdamBiasAdjustment iter beta1) momenta1'
    biasAdjustedMomenta2 = hmap' (AdamBiasAdjustment iter beta2) momenta2'
    parameters' =
      hzipWith3
        (AdamParameterUpdate 1e-37 learningRate)
        parameters
        biasAdjustedMomenta1
        biasAdjustedMomenta2

instance
  ( HZipWith AdamMomentum1Update momenta gradients momenta,
    HZipWith AdamMomentum2Update momenta gradients momenta,
    HMap' AdamBiasAdjustment momenta momenta,
    HZipWith3 (AdamParameterUpdate device dtype) tensors momenta momenta tensors
  ) =>
  Optimizer (Adam momenta) gradients tensors dtype device
  where
  step = adam

instance
  HAppendFD momenta momenta (momenta ++ momenta) =>
  Parameterized (Adam momenta)
  where
  type Parameters (Adam momenta) = AdamIter ': (momenta ++ momenta)
  flattenParameters Adam {..} = iter :. (momenta1 `happendFD` momenta2)
  replaceParameters adam (iter :. momenta) =
    let (momenta1, momenta2) = hunappendFD momenta
     in adam {iter = iter, momenta1 = momenta1, momenta2 = momenta2}
