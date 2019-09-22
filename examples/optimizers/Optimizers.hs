{-# LANGUAGE RecordWildCards #-}

module Optimizers where

import Prelude hiding (sqrt)

import Torch.Tensor
import Torch.Functions
import Torch.Autograd
import Torch.NN

type LearningRate = Tensor
type Gradient = [Tensor]

class Optimizer a where
    step :: LearningRate -> a -> [Parameter] -> Gradient -> ([Tensor], a)

--
-- Gradient Descent
--

data GD = GD deriving Show

-- | Gradient descent step
gd :: LearningRate -> [Parameter] -> Gradient -> [Tensor]
gd lr parameters gradients = zipWith step depParameters gradients
  where
    step p dp = p - (lr * dp)
    depParameters = fmap toDependent parameters

instance Optimizer GD where
    step lr dummy parameters gradients = (gd lr parameters gradients, dummy) 

--
-- Gradient Descent with Momentum
--

data GDM = GDM { beta :: Float, memory :: [Tensor] } deriving Show

-- gradient descent with momentum step
gdm 
    :: LearningRate -- ^ learning rate
    -> GDM -- ^ beta & memory
    -> [Parameter] -- ^ parameters
    -> Gradient --gradients
    -> ([Tensor], GDM)
gdm lr GDM{..} parameters gradients = (fmap fst runStep, GDM beta (fmap snd runStep))
  where
    z' dp z = mulScalar z beta + dp
    step p dp z = let newZ = z' dp z in (p - lr * newZ, newZ)
    depParameters = fmap toDependent parameters
    runStep = (zipWith3 step) depParameters gradients memory

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
adam :: LearningRate -> Adam -> [Parameter] -> Gradient -> ([Tensor], Adam)
adam lr Adam{..} parameters gradients = (w, Adam beta1 beta2 m1' m2' (iter+1))
    where
        -- 1st & 2nd moments
        f1 m1 dp = mulScalar m1 beta1 + mulScalar dp (1 - beta1)
        f2 m2 dp = mulScalar m2 beta2 + mulScalar (dp * dp) (1 - beta2)
        m1' = zipWith f1 m1 gradients
        m2' = zipWith f2 m2 gradients
        -- averages of moments
        a beta m = divScalar m (1 - beta^(iter + 1))
        a1 = fmap (a beta1) m1'
        a2 = fmap (a beta2) m2'
        -- parameter update
        eps = 1e-15
        fw wprev avg1 avg2 = wprev - lr * avg1 / (sqrt avg2 + eps)
        parameters' = fmap toDependent parameters
        w = zipWith3 fw parameters' a1 a2

instance Optimizer Adam where
    step lr state parameters gradients = adam lr state parameters gradients
