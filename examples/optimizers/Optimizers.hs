{-# LANGUAGE RecordWildCards #-}

module Optimizers where

import Prelude hiding (sqrt)

import Torch.Tensor
import Torch.Functions
import Torch.Autograd
import Torch.NN


-- type aliases for readability
type LearningRate = Tensor
type Gradient = Tensor

class Optimizer o where
    step :: LearningRate -> o -> [Parameter] -> [Gradient] -> ([Tensor], o)

--
-- Gradient Descent
--

data GD = GD deriving Show

-- | Gradient descent step
gd :: LearningRate -> [Parameter] -> [Gradient] -> [Tensor]
gd lr parameters gradients = zipWith step depParameters gradients
  where
    step p dp = p - (lr * dp)
    depParameters = fmap toDependent parameters

instance Optimizer GD where
    step lr dummy parameters gradients = (gd lr parameters gradients, dummy) 

--
-- Gradient Descent with Momentum
--

data GDM = GDM { beta :: Float, momentum :: [Tensor] } deriving Show

-- gradient descent with momentum step
gdm 
    :: LearningRate -- ^ learning rate
    -> GDM -- ^ beta & momentum
    -> [Parameter] -- ^ model parameters
    -> [Gradient] -- ^ model parameter gradients
    -> ([Tensor], GDM) -- ^ returns new parameters + updated momentum
gdm lr GDM{..} parameters gradients = (fmap fst runStep, GDM beta (fmap snd runStep))
  where
    step p dp z = let z' = mulScalar z beta + dp in (p - lr * z', z')
    runStep = (zipWith3 step) (fmap toDependent parameters) gradients momentum

instance Optimizer GDM where
    step lr state parameters gradients = gdm lr state parameters gradients

--
-- Adam
--

-- | State representation for Adam Optimizer
data Adam = Adam { 
    beta1 :: Float,
    beta2 :: Float,
    m1 :: [Tensor], -- 1st moment
    m2 :: [Tensor], -- 2nd moment
    iter :: Int -- iteration
    } deriving Show

-- | Adap step
adam 
    :: LearningRate  -- ^ learning rate
    -> Adam -- ^ adam parameters - beta1, beta2, moments, iteration
    -> [Parameter] -- ^ model parameters
    -> [Gradient] -- ^ model parameter gradients
    -> ([Tensor], Adam) -- ^ returns new parameters + updated adam parameters
adam lr Adam{..} parameters gradients = (parameters', Adam beta1 beta2 m1' m2' (iter+1))
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
        depParameters  = fmap toDependent parameters
        parameters' = zipWith3 update depParameters a1 a2

instance Optimizer Adam where
    step lr state parameters gradients = adam lr state parameters gradients
