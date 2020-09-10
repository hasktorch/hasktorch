{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE TypeFamilies #-}

module Torch.Optim.Internal where

import Foreign.ForeignPtr

import Torch.Internal.Cast
import Torch.Internal.Class (Castable(..), CppTuple2(..), CppTuple3(..), CppTuple4(..), CppObject(..))
import qualified Torch.Internal.Type as ATen
import qualified Torch.Internal.Managed.Optim as LibTorch

import Torch.Tensor
import Torch.Autograd
import Torch.NN

import Data.Default.Class

type OptimizerRef = ForeignPtr ATen.Optimizer
data OptimizerState option p = OptimizerState option OptimizerRef p

class Optimizer option where
  initOptimizer :: Parameterized d => option -> d -> IO (OptimizerState option d)
  step :: Parameterized d => OptimizerState option d -> (d -> IO Tensor) -> IO Tensor
  -- Returned d depends on the state of optimizer.
  -- Do not call step function after this function is called.
  getParams :: Parameterized d => OptimizerState option d -> IO d
  step (OptimizerState _ optimizer initParams) loss = cast0 (LibTorch.step optimizer trans)
    where
      trans :: ForeignPtr ATen.TensorList -> IO (ForeignPtr ATen.Tensor)
      trans inputs =
        uncast inputs $ \inputs' -> do
          (Unsafe ret) <- loss $ replaceParameters initParams $  map (IndependentTensor . Unsafe) inputs'
          cast ret return
  getParams (OptimizerState _ optimizer initParams) = fmap (replaceParameters initParams . map (IndependentTensor . Unsafe)) $ cast0 (LibTorch.getParams optimizer)

data AdagradOptions = AdagradOptions
  { adagradLr :: Double
  , adagradLrDecay :: Double
  , adagradWeightDecay :: Double
  , adagradInitialAccumulatorValue :: Double
  , adagradEps :: Double
  } deriving (Show, Eq)

instance Default AdagradOptions where
  def = AdagradOptions
    { adagradLr = 1e-2
    , adagradLrDecay = 0
    , adagradWeightDecay = 0
    , adagradInitialAccumulatorValue = 0
    , adagradEps = 1e-10
    } 

instance Optimizer AdagradOptions where
  initOptimizer  opt@AdagradOptions{..} initParams = do
    v <- cast6 LibTorch.adagrad adagradLr adagradLrDecay adagradWeightDecay adagradInitialAccumulatorValue adagradEps initParams'
    return $ OptimizerState opt v initParams
    where
      initParams' = map toDependent $ flattenParameters initParams

data AdamOptions = AdamOptions
  { adamLr :: Double
  , adamBetas :: (Double,Double)
  , adamEps :: Double
  , adamWeightDecay :: Double
  , adamAmsgrad :: Bool
  } deriving (Show, Eq)


instance Default AdamOptions where
  def = AdamOptions
    { adamLr = 1e-3
    , adamBetas = (0.9, 0.999)
    , adamEps = 1e-8
    , adamWeightDecay = 0
    , adamAmsgrad = False
    } 

instance Optimizer AdamOptions where
  initOptimizer  opt@AdamOptions{..} initParams = do
    v <- cast7 LibTorch.adam adamLr (fst adamBetas) (snd adamBetas) adamEps adamWeightDecay adamAmsgrad initParams'
    return $ OptimizerState opt v initParams
    where
      initParams' = map toDependent $ flattenParameters initParams

data AdamwOptions = AdamwOptions
  { adamwLr :: Double
  , adamwBetas :: (Double,Double)
  , adamwEps :: Double
  , adamwWeightDecay :: Double
  , adamwAmsgrad :: Bool
  } deriving (Show, Eq)

instance Default AdamwOptions where
  def = AdamwOptions
    { adamwLr = 1e-3
    , adamwBetas = (0.9, 0.999)
    , adamwEps = 1e-8
    , adamwWeightDecay = 1e-2
    , adamwAmsgrad = False
    } 

instance Optimizer AdamwOptions where
  initOptimizer  opt@AdamwOptions{..} initParams = do
    v <- cast7 LibTorch.adamw adamwLr (fst adamwBetas) (snd adamwBetas) adamwEps adamwWeightDecay adamwAmsgrad initParams'
    return $ OptimizerState opt v initParams
    where
      initParams' = map toDependent $ flattenParameters initParams

data LbfgsOptions = LbfgsOptions
  { lbfgsLr :: Double
  , lbfgsMaxIter :: Int
  , lbfgsMaxEval :: Int
  , lbfgsToleranceGrad :: Double
  , lbfgsToleranceChange :: Double
  , lbfgsHistorySize :: Int
  , lbfgsLineSearchFn :: Maybe String
  } deriving (Show, Eq)
  
instance Default LbfgsOptions where
  def = LbfgsOptions
    { lbfgsLr = 1
    , lbfgsMaxIter = 20
    , lbfgsMaxEval = (20 * 5) `div` 4
    , lbfgsToleranceGrad = 1e-7
    , lbfgsToleranceChange = 1e-9
    , lbfgsHistorySize = 100
    , lbfgsLineSearchFn = Nothing
    } 

instance Optimizer LbfgsOptions where
  initOptimizer opt@LbfgsOptions{..} initParams = do
    v <- cast8 LibTorch.lbfgs lbfgsLr lbfgsMaxIter lbfgsMaxEval lbfgsToleranceGrad lbfgsToleranceChange lbfgsHistorySize lbfgsLineSearchFn initParams'
    return $ OptimizerState opt v initParams
    where
      initParams' = map toDependent $ flattenParameters initParams

data RmspropOptions = RmspropOptions
  { rmspropLr :: Double
  , rmspropAlpha :: Double
  , rmspropEps :: Double
  , rmspropWeightDecay :: Double
  , rmspropMomentum :: Double
  , rmspropCentered :: Bool
  } deriving (Show, Eq)

instance Default RmspropOptions where
  def = RmspropOptions
    { rmspropLr = 1e-2
    , rmspropAlpha = 0.99
    , rmspropEps = 1e-8
    , rmspropWeightDecay = 0
    , rmspropMomentum = 0
    , rmspropCentered = False
    } 

instance Optimizer RmspropOptions where
  initOptimizer opt@RmspropOptions{..} initParams = do
    v <- cast7 LibTorch.rmsprop rmspropLr rmspropAlpha rmspropEps rmspropWeightDecay rmspropMomentum rmspropCentered initParams'
    return $ OptimizerState opt v initParams
    where
      initParams' = map toDependent $ flattenParameters initParams
data SGDOptions = SGDOptions
  { sgdLr :: Double
  , sgdMomentum :: Double
  , sgdDampening :: Double
  , sgdWeightDecay :: Double
  , sgdNesterov :: Bool
  } deriving (Show, Eq)

instance Default SGDOptions where
  def = SGDOptions
    { sgdLr = 1e-3
    , sgdMomentum = 0
    , sgdDampening = 0
    , sgdWeightDecay = 0
    , sgdNesterov = False
    } 

instance Optimizer SGDOptions where
  initOptimizer  opt@SGDOptions{..} initParams = do
    v <- cast6 LibTorch.sgd sgdLr sgdMomentum sgdDampening sgdWeightDecay sgdNesterov initParams'
    return $ OptimizerState opt v initParams
    where
      initParams' = map toDependent $ flattenParameters initParams
