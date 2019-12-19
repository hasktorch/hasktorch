{-# LANGUAGE DataKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}

module Torch.Typed.Optim where

import           Prelude                 hiding ( sqrt )
import           Control.Monad.State
import           Torch.HList

import qualified Torch.Internal.Cast                     as ATen
import qualified Torch.Internal.Class                    as ATen
import qualified Torch.Tensor                  as D
import           Torch.Typed.Aux
import           Torch.Typed.Tensor
import           Torch.Typed.Autograd
import           Torch.Typed.Parameter
import           Torch.Typed.Functional
import           Torch.Typed.Factories

type LearningRate device dtype = Tensor device dtype '[]
type Loss device dtype = Tensor device dtype '[]

data ZerosLike = ZerosLike

instance
  ( parameter ~ Parameter device dtype shape
  , momentum ~ Tensor device dtype shape
  , TensorOptions shape dtype device
  ) => Apply' ZerosLike parameter momentum where
  apply' _ _ = zeros

class Optimizer optim gradients tensors dtype device where
  step :: LearningRate device dtype -> HList gradients -> HList tensors -> optim -> (HList tensors, optim)

runStep
  :: forall model optim parameters gradients tensors dtype device
   . ( Parameterized model parameters
     , HasGrad (HList parameters) (HList gradients)
     , tensors ~ gradients
     , HMap' ToDependent parameters tensors
     , ATen.Castable (HList gradients) [D.ATenTensor]
     , Optimizer optim gradients tensors dtype device
     , HMapM' IO MakeIndependent tensors parameters
     )
  => model
  -> optim
  -> Loss device dtype
  -> LearningRate device dtype
  -> IO (model, optim)
runStep modelState optimState loss learningRate = do
  let parameters              = flattenParameters modelState
      gradients               = grad loss parameters
      tensors                 = hmap' ToDependent parameters
      (tensors', optimState') = step learningRate gradients tensors optimState
  parameters' <- hmapM' MakeIndependent tensors'
  let modelState' = replaceParameters modelState parameters'
  return (modelState', optimState')

--
-- Gradient Descent (GD)
--

-- | Dummy state representation for GD Optimizer
data GD = GD

mkGD :: GD
mkGD = GD

newtype GDStep device dtype = GDStep (LearningRate device dtype)

instance
  ( parameter ~ Tensor device dtype shape
  , gradient ~ Tensor device dtype shape
  , shape ~ Broadcast '[] shape
  , BasicArithmeticDTypeIsValid device dtype
  ) => Apply' (GDStep device dtype) (parameter, gradient) parameter where
  apply' (GDStep learningRate) (parameter, gradient) =
    parameter - mul learningRate gradient

-- | Gradient descent step with a dummy state variable
gd
  :: forall gradients tensors dtype device
   . HZipWith (GDStep device dtype) tensors gradients tensors
  => LearningRate device dtype
  -> HList gradients
  -> HList tensors
  -> GD
  -> (HList tensors, GD)
gd learningRate gradients parameters gd =
  let step = hZipWith (GDStep learningRate) parameters gradients in (step, gd)

instance
  ( HZipWith (GDStep device dtype) tensors gradients tensors
  ) => Optimizer GD gradients tensors dtype device where
  step = gd

--
-- Gradient Descent with Momentum (GDM)
--

-- | State representation for GDM Optimizer
data GDM momenta = GDM
  { beta :: Float -- moment forgetting factor
  , momenta :: HList momenta -- momenta
  }

mkGDM
  :: forall parameters momenta
   . (HMap' ZerosLike parameters momenta)
  => Float
  -> HList parameters
  -> GDM momenta
mkGDM beta parameters = GDM beta (hmap' ZerosLike parameters)

data GDMStep device dtype = GDMStep Float (LearningRate device dtype)

instance
  ( parameter ~ Tensor device dtype shape
  , gradient ~ Tensor device dtype shape
  , momentum ~ Tensor device dtype shape
  , shape ~ Broadcast '[] shape
  , BasicArithmeticDTypeIsValid device dtype
  ) => Apply' (GDMStep device dtype) (parameter, gradient, momentum) (parameter, momentum) where
  apply' (GDMStep beta learningRate) (parameter, gradient, momentum) =
    let momentum'  = cmul beta momentum + gradient
        parameter' = parameter - mul learningRate momentum'
    in  (parameter', momentum')

-- | gradient descent with momentum step
gdm
  :: forall gradients tensors momenta gdmStep dtype device
   . ( HZipWith3 (GDMStep device dtype) tensors gradients momenta gdmStep
     , HMap' Torch.HList.Fst gdmStep tensors
     , HMap' Torch.HList.Snd gdmStep momenta
     )
  => LearningRate device dtype -- ^ learning rate
  -> HList gradients -- ^ model parameter gradient tensors
  -> HList tensors -- ^ model parameter tensors
  -> GDM momenta -- ^ beta and model parameter momentum tensors
  -> (HList tensors, GDM momenta) -- ^ returns updated parameters and momenta
gdm learningRate gradients parameters (GDM beta momenta) =
  let step = hZipWith3 (GDMStep beta learningRate) parameters gradients momenta
  in  (hmap' Fst step, GDM beta (hmap' Snd step))

instance
  ( HZipWith3 (GDMStep device dtype) tensors gradients momenta gdmStep
  , HMap' Torch.HList.Fst gdmStep tensors
  , HMap' Torch.HList.Snd gdmStep momenta
  ) => Optimizer (GDM momenta) gradients tensors dtype device where
  step = gdm

--
-- Adam
-- https://arxiv.org/pdf/1412.6980.pdf
--

-- | State representation for Adam Optimizer
data Adam momenta = Adam
  { iter :: Int -- iteration
  , beta1 :: Float -- 1st moment forgetting factor
  , beta2 :: Float -- 2nd moment forgetting factor
  , momenta1 :: HList momenta -- 1st momenta
  , momenta2 :: HList momenta -- 2nd momenta
  }

mkAdam
  :: forall parameters momenta
   . (HMap' ZerosLike parameters momenta)
  => Int
  -> Float
  -> Float
  -> HList parameters
  -> Adam momenta
mkAdam iter beta1 beta2 parameters = Adam iter
                                          beta1
                                          beta2
                                          (hmap' ZerosLike parameters)
                                          (hmap' ZerosLike parameters)

newtype AdamMomentum1Update = AdamMomentum1Update Float

-- | decaying average of the first momenta
instance
  ( gradient ~ Tensor device dtype shape
  , momentum1 ~ Tensor device dtype shape
  ) => Apply' AdamMomentum1Update (momentum1, gradient) momentum1 where
    apply' (AdamMomentum1Update beta1) (momentum1, gradient) =
      cmul beta1 momentum1 + cmul (1 - beta1) gradient

newtype AdamMomentum2Update = AdamMomentum2Update Float

-- | decaying average of the second momenta
instance
  ( gradient ~ Tensor device dtype shape
  , momentum2 ~ Tensor device dtype shape
  , shape ~ Broadcast shape shape
  , BasicArithmeticDTypeIsValid device dtype
  ) => Apply' AdamMomentum2Update (momentum2, gradient) momentum2 where
    apply' (AdamMomentum2Update beta2) (momentum2, gradient) =
      cmul beta2 momentum2 + cmul (1 - beta2) (mul gradient gradient)

data AdamBiasAdjustment = AdamBiasAdjustment Int Float

-- | bias adjustment
instance
  ( momentum ~ Tensor device dtype shape
  ) => Apply' AdamBiasAdjustment momentum momentum where
    apply' (AdamBiasAdjustment iter beta) momentum =
      cdiv (1 - beta ^ (iter + 1)) momentum

data AdamParameterUpdate device dtype = AdamParameterUpdate Float (LearningRate device dtype)

-- | parameter update
instance
  ( parameter ~ Tensor device dtype shape
  , momentum ~ Tensor device dtype shape
  , shape ~ Broadcast '[] shape
  , BasicArithmeticDTypeIsValid device dtype
  , StandardFloatingPointDTypeValidation device dtype
  ) => Apply' (AdamParameterUpdate device dtype) (parameter, momentum, momentum) parameter where
  apply' (AdamParameterUpdate eps learningRate) (parameter, biasAdjustedMomentum1, biasAdjustedMomentum2) =
    parameter - mul learningRate biasAdjustedMomentum1 / cadd eps (sqrt biasAdjustedMomentum2)

-- | Adam step
adam
  :: forall gradients tensors momenta adamStep dtype device
   . ( HZipWith AdamMomentum1Update momenta gradients momenta
     , HZipWith AdamMomentum2Update momenta gradients momenta
     , HMap' AdamBiasAdjustment momenta momenta
     , HZipWith3 (AdamParameterUpdate device dtype) tensors momenta momenta tensors
     )
  => LearningRate device dtype  -- ^ learning rate
  -> HList gradients -- ^ model parameter gradient tensors
  -> HList tensors -- ^ model parameter tensors
  -> Adam momenta -- ^ adam parameters - beta1, beta2, momenta1, momenta2, iteration
  -> (HList tensors, Adam momenta) -- ^ returns new parameters + updated adam parameters
adam learningRate gradients parameters Adam {..} =
  (parameters', Adam (iter + 1) beta1 beta2 momenta1' momenta2')
 where
  momenta1' = hZipWith (AdamMomentum1Update beta1) momenta1 gradients
  momenta2' = hZipWith (AdamMomentum2Update beta2) momenta2 gradients
  biasAdjustedMomenta1 = hmap' (AdamBiasAdjustment iter beta1) momenta1'
  biasAdjustedMomenta2 = hmap' (AdamBiasAdjustment iter beta2) momenta2'
  parameters' = hZipWith3 (AdamParameterUpdate 1e-37 learningRate) parameters biasAdjustedMomenta1 biasAdjustedMomenta2

instance
  ( HZipWith AdamMomentum1Update momenta gradients momenta
  , HZipWith AdamMomentum2Update momenta gradients momenta
  , HMap' AdamBiasAdjustment momenta momenta
  , HZipWith3 (AdamParameterUpdate device dtype) tensors momenta momenta tensors
  ) => Optimizer (Adam momenta) gradients tensors dtype device where
  step = adam
