{-# LANGUAGE RecordWildCards #-}

module Optimizers where

import Control.Monad.State
import Prelude hiding (sqrt)

import Torch.Tensor
import Torch.Functional
import Torch.Autograd
import Torch.NN

type LearningRate = Tensor
newtype Gradients = Gradients [Tensor] deriving Show

grad' t p = Gradients (grad t p)

class Optimizer o where
    step :: LearningRate -> Gradients -> [Tensor] -> o -> ([Tensor], o)

--
-- Gradient Descent
--

data GD = GD deriving Show

-- | Stateless gradient descent step
gd :: LearningRate -> Gradients -> [Tensor] -> [Tensor]
gd lr (Gradients gradients) parameters = zipWith step parameters gradients
  where
    step p dp = p - (lr * dp)

-- | Gradient descent step with a dummy state variable
gd' :: LearningRate -> Gradients -> [Tensor] -> GD -> ([Tensor], GD) 
gd' lr gradients depParameters dummy = (gd lr gradients depParameters, dummy)

instance Optimizer GD where
    step = gd'

--
-- Gradient Descent with Momentum
--

data GDM = GDM { beta :: Float, momentum :: [Tensor] } deriving Show

-- gradient descent with momentum step
gdm 
    :: LearningRate -- ^ learning rate
    -> Gradients -- ^ model parameter gradients
    -> [Tensor] -- ^ model parameters
    -> GDM -- ^ beta & momentum
    -> ([Tensor], GDM) -- ^ returns new parameters + updated momentum
gdm lr (Gradients gradients) parameters (GDM beta momentum) = 
    (fmap fst runStep, GDM beta (fmap snd runStep))
    where
        step p dp z = let z' = mulScalar z beta + dp in (p - lr * z', z')
        runStep = (zipWith3 step) parameters gradients momentum

instance Optimizer GDM where
    step = gdm

--
-- Adam
--

-- | State representation for Adam Optimizer
data Adam = Adam { 
    beta1 :: Float, -- 1st moment forgetting factor
    beta2 :: Float, -- 2nd moment forgetting factor
    m1 :: [Tensor], -- 1st moment
    m2 :: [Tensor], -- 2nd moment
    iter :: Int -- iteration
    } deriving Show

-- | Adam step
adam 
    :: LearningRate  -- ^ learning rate
    -> Gradients -- ^ model parameter gradients
    -> [Tensor] -- ^ model parameters
    -> Adam -- ^ adam parameters - beta1, beta2, moments, iteration
    -> ([Tensor], Adam) -- ^ returns new parameters + updated adam parameters
adam lr (Gradients gradients) parameters Adam{..} = (parameters', Adam beta1 beta2 m1' m2' (iter+1))
    where
        -- decaying averages of 1st & 2nd moments
        f1 m1 dp = mulScalar m1 beta1 + mulScalar dp (1 - beta1)
        f2 m2 dp = mulScalar m2 beta2 + mulScalar (dp * dp) (1 - beta2)
        m1' = zipWith f1 m1 gradients
        m2' = zipWith f2 m2 gradients
        -- bias adjustment
        a beta m = divScalar m (1 - beta^(iter + 1))
        a1 = fmap (a beta1) m1'
        a2 = fmap (a beta2) m2'
        -- parameter update
        eps = 1e-37
        update prevParam a1' a2' = prevParam  - lr * a1' / (sqrt a2' + eps)
        parameters' = zipWith3 update parameters a1 a2

instance Optimizer Adam where
    step = adam
