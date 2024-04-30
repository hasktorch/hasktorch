{-# LANGUAGE RecordWildCards #-}

module Torch.Optim where

import Control.Monad.State
import Control.Monad (foldM)
import System.Mem (performGC)
import Torch.Autograd
import Torch.Functional
import Torch.Internal.GC (mallocTrim)
import Torch.NN
import Torch.Tensor
import Torch.TensorFactories
import Prelude hiding (sqrt)

type LearningRate = Tensor

type Loss = Tensor

newtype Gradients = Gradients [Tensor] deriving (Show)

newtype OptimizerState option = OptimizerState option

grad' :: Loss -> [Parameter] -> Gradients
grad' t p = Gradients (grad t p)

class Optimizer optimizer where
  step :: LearningRate -> Gradients -> [Tensor] -> optimizer -> ([Tensor], optimizer)

  -- | run a single iteration of an optimizer, returning new parameters and updated optimizer state
  runStep :: (Parameterized model) => model -> optimizer -> Loss -> LearningRate -> IO (model, optimizer)
  runStep paramState optState lossValue = runStep' paramState optState (grad' lossValue $ flattenParameters paramState)

  -- | run a single iteration of an optimizer, returning new parameters and updated optimizer state
  runStep' :: (Parameterized model) => model -> optimizer -> Gradients -> LearningRate -> IO (model, optimizer)
  runStep' paramState optState gradients lr = do
    performGC
    mallocTrim 0
    let (flatParameters', optState') = step lr gradients depParameters optState
    newFlatParam <- mapM makeIndependent flatParameters'
    pure (replaceParameters paramState newFlatParam, optState')
    where
      flatParameters = flattenParameters paramState
      depParameters = fmap toDependent flatParameters

--
-- Gradient Descent
--

data GD = GD deriving (Show)

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

sgd :: LearningRate -> [Parameter] -> [Tensor] -> [Tensor]
sgd lr parameters = zipWith step depParameters
  where
    step p dp = p - (lr * dp)
    depParameters = map toDependent parameters

--
-- Gradient Descent with Momentum
--

data GDM = GDM {beta :: Float, momentum :: [Tensor]} deriving (Show)

-- gradient descent with momentum step
gdm ::
  -- | learning rate
  LearningRate ->
  -- | model parameter gradients
  Gradients ->
  -- | model parameters
  [Tensor] ->
  -- | beta & momentum
  GDM ->
  -- | returns new parameters + updated momentum
  ([Tensor], GDM)
gdm lr (Gradients gradients) parameters (GDM beta momentum) =
  (fmap fst runStep, GDM beta (fmap snd runStep))
  where
    step p dp z = let z' = mulScalar beta z + dp in (p - lr * z', z')
    runStep = zipWith3 step parameters gradients momentum

instance Optimizer GDM where
  step = gdm

--
-- Adam
--

-- | State representation for Adam Optimizer
data Adam = Adam
  { beta1 :: Float, -- 1st moment forgetting factor
    beta2 :: Float, -- 2nd moment forgetting factor
    m1 :: [Tensor], -- 1st moment
    m2 :: [Tensor], -- 2nd moment
    iter :: Int -- iteration
  }
  deriving (Show)

mkAdam ::
  Int ->
  Float ->
  Float ->
  [Parameter] ->
  Adam
mkAdam iter beta1 beta2 parameters =
  Adam
    beta1
    beta2
    (initZeros <$> parameters)
    (initZeros <$> parameters)
    iter
  where
    initZeros = zerosLike . toDependent

-- | Adam step
adam ::
  -- | learning rate
  LearningRate ->
  -- | model parameter gradients
  Gradients ->
  -- | model parameters
  [Tensor] ->
  -- | adam parameters - beta1, beta2, moments, iteration
  Adam ->
  -- | returns new parameters + updated adam parameters
  ([Tensor], Adam)
adam lr (Gradients gradients) parameters Adam {..} = (parameters', Adam beta1 beta2 m1' m2' (iter + 1))
  where
    -- decaying averages of 1st & 2nd moments
    f1 m1 dp = mulScalar beta1 m1 + mulScalar (1 - beta1) dp
    f2 m2 dp = mulScalar beta2 m2 + mulScalar (1 - beta2) (dp * dp)
    m1' = zipWith f1 m1 gradients
    m2' = zipWith f2 m2 gradients
    -- bias adjustment
    a beta = divScalar (1 - beta ^ (iter + 1))
    a1 = fmap (a beta1) m1'
    a2 = fmap (a beta2) m2'
    -- parameter update
    eps = 1e-37
    update prevParam a1' a2' = prevParam - lr * a1' / (sqrt a2' + eps)
    parameters' = zipWith3 update parameters a1 a2

instance Optimizer Adam where
  step = adam

--
-- Adagrad
--

-- | State representation for Adagrad Optimizer
data Adagrad = Adagrad {gsum :: [Tensor]} -- sum of squared gradients
  deriving (Show)

-- | Adagrad step
adagrad ::
  -- | learning rate
  LearningRate ->
  -- | model parameter gradients
  Gradients ->
  -- | model parameters
  [Tensor] ->
  -- | adagrad parameters - gsum, iteration
  Adagrad ->
  -- | returns new parameters + updated adam parameters
  ([Tensor], Adagrad)
adagrad lr (Gradients gradients) parameters Adagrad {..} = (parameters', Adagrad gsum')
  where
    -- add gradient squared to running total
    f gsum dp = gsum + dp * dp
    gsum' = zipWith f gsum gradients

    -- parameter update
    eps = 1e-37
    update prevParam a1' a2' = prevParam - lr * a1' / (sqrt (a2' + eps))
    parameters' = zipWith3 update parameters gradients gsum'

instance Optimizer Adagrad where
  step = adagrad

-- | syntactic sugar for looping with foldM
foldLoop :: a -> Int -> (a -> Int -> IO a) -> IO a
foldLoop x count block = foldM block x [1 .. count]
