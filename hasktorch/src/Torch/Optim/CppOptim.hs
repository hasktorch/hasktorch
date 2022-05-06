{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}

module Torch.Optim.CppOptim where

import Data.Default.Class
import Foreign.ForeignPtr
import System.Mem (performGC)
import Torch.Autograd
import Torch.Internal.Cast
import Torch.Internal.Class (Castable (..), CppObject (..), CppTuple2 (..), CppTuple3 (..), CppTuple4 (..))
import Torch.Internal.GC (mallocTrim)
import qualified Torch.Internal.Managed.Optim as LibTorch
import qualified Torch.Internal.Type as ATen
import Torch.NN
import qualified Torch.Optim as Optim
import Torch.Tensor

type CppOptimizerRef = ForeignPtr ATen.Optimizer

data CppOptimizerState option = CppOptimizerState option CppOptimizerRef

-- class Optimizer option where
--   initOptimizer :: Parameterized model => option -> model -> IO (OptimizerState option model)
--   step :: Parameterized model => OptimizerState option model -> (model -> IO Tensor) -> IO Tensor
--   -- Returned d depends on the state of optimizer.
--   -- Do not call step function after this function is called.
--   getParams :: Parameterized model => OptimizerState option model -> IO model
--   step (OptimizerState _ optimizer initParams) loss = cast0 (LibTorch.step optimizer trans)
--     where
--       trans :: ForeignPtr ATen.TensorList -> IO (ForeignPtr ATen.Tensor)
--       trans inputs =
--         uncast inputs $ \inputs' -> do
--           (Unsafe ret) <- loss $ replaceParameters initParams $  map (IndependentTensor . Unsafe) inputs'
--           cast ret return
--   getParams (OptimizerState _ optimizer initParams) = fmap (replaceParameters initParams . map (IndependentTensor . Unsafe)) $ cast0 (LibTorch.getParams optimizer)

stepWithGenerator ::
  CppOptimizerState option ->
  ForeignPtr ATen.Generator ->
  ([Tensor] -> ForeignPtr ATen.Generator -> IO (Tensor, ForeignPtr ATen.Generator)) ->
  IO (Tensor, ForeignPtr ATen.Generator)
stepWithGenerator o@(CppOptimizerState _ ref) generator loss = do
  (v, nextGenerator) <- cast3 LibTorch.stepWithGenerator ref generator loss'
  return (v, nextGenerator)
  where
    loss' :: ForeignPtr ATen.TensorList -> ForeignPtr ATen.Generator -> IO (ForeignPtr (ATen.StdTuple '(ATen.Tensor, ATen.Generator)))
    loss' params gen = do
      (v :: Tensor, gen') <- uncast params $ \params' -> loss params' gen
      v' <- cast v pure :: IO (ForeignPtr ATen.Tensor)
      cast (v', gen') pure

class CppOptimizer option where
  initOptimizer :: Parameterized model => option -> model -> IO (CppOptimizerState option)
  unsafeStep :: Parameterized model => model -> CppOptimizerState option -> Tensor -> IO (model, CppOptimizerState option)
  unsafeStep model o@(CppOptimizerState _ optimizer) loss = do
    v <- cast2 LibTorch.unsafeStep optimizer loss
    let newModel = replaceParameters model $ map (IndependentTensor . Unsafe) v
    return (newModel, o)

instance {-# OVERLAPS #-} CppOptimizer option => Optim.Optimizer (CppOptimizerState option) where
  step = error "step is not implemented for CppOptimizer."
  runStep paramState optState lossValue lr = do
    performGC
    mallocTrim 0
    unsafeStep paramState optState lossValue

  runStep' = error "runStep' is not implemented for CppOptimizer."

data AdagradOptions = AdagradOptions
  { adagradLr :: Double,
    adagradLrDecay :: Double,
    adagradWeightDecay :: Double,
    adagradInitialAccumulatorValue :: Double,
    adagradEps :: Double
  }
  deriving (Show, Eq)

instance Default AdagradOptions where
  def =
    AdagradOptions
      { adagradLr = 1e-2,
        adagradLrDecay = 0,
        adagradWeightDecay = 0,
        adagradInitialAccumulatorValue = 0,
        adagradEps = 1e-10
      }

instance CppOptimizer AdagradOptions where
  initOptimizer opt@AdagradOptions {..} initParams = do
    v <- cast6 LibTorch.adagrad adagradLr adagradLrDecay adagradWeightDecay adagradInitialAccumulatorValue adagradEps initParams'
    return $ CppOptimizerState opt v
    where
      initParams' = map toDependent $ flattenParameters initParams

data AdamOptions = AdamOptions
  { adamLr :: Double,
    adamBetas :: (Double, Double),
    adamEps :: Double,
    adamWeightDecay :: Double,
    adamAmsgrad :: Bool
  }
  deriving (Show, Eq)

instance Default AdamOptions where
  def =
    AdamOptions
      { adamLr = 1e-3,
        adamBetas = (0.9, 0.999),
        adamEps = 1e-8,
        adamWeightDecay = 0,
        adamAmsgrad = False
      }

instance CppOptimizer AdamOptions where
  initOptimizer opt@AdamOptions {..} initParams = do
    v <- cast7 LibTorch.adam adamLr (fst adamBetas) (snd adamBetas) adamEps adamWeightDecay adamAmsgrad initParams'
    return $ CppOptimizerState opt v
    where
      initParams' = map toDependent $ flattenParameters initParams

data AdamwOptions = AdamwOptions
  { adamwLr :: Double,
    adamwBetas :: (Double, Double),
    adamwEps :: Double,
    adamwWeightDecay :: Double,
    adamwAmsgrad :: Bool
  }
  deriving (Show, Eq)

instance Default AdamwOptions where
  def =
    AdamwOptions
      { adamwLr = 1e-3,
        adamwBetas = (0.9, 0.999),
        adamwEps = 1e-8,
        adamwWeightDecay = 1e-2,
        adamwAmsgrad = False
      }

instance CppOptimizer AdamwOptions where
  initOptimizer opt@AdamwOptions {..} initParams = do
    v <- cast7 LibTorch.adamw adamwLr (fst adamwBetas) (snd adamwBetas) adamwEps adamwWeightDecay adamwAmsgrad initParams'
    return $ CppOptimizerState opt v
    where
      initParams' = map toDependent $ flattenParameters initParams

data LbfgsOptions = LbfgsOptions
  { lbfgsLr :: Double,
    lbfgsMaxIter :: Int,
    lbfgsMaxEval :: Int,
    lbfgsToleranceGrad :: Double,
    lbfgsToleranceChange :: Double,
    lbfgsHistorySize :: Int,
    lbfgsLineSearchFn :: Maybe String
  }
  deriving (Show, Eq)

instance Default LbfgsOptions where
  def =
    LbfgsOptions
      { lbfgsLr = 1,
        lbfgsMaxIter = 20,
        lbfgsMaxEval = (20 * 5) `div` 4,
        lbfgsToleranceGrad = 1e-7,
        lbfgsToleranceChange = 1e-9,
        lbfgsHistorySize = 100,
        lbfgsLineSearchFn = Nothing
      }

instance CppOptimizer LbfgsOptions where
  initOptimizer opt@LbfgsOptions {..} initParams = do
    v <- cast8 LibTorch.lbfgs lbfgsLr lbfgsMaxIter lbfgsMaxEval lbfgsToleranceGrad lbfgsToleranceChange lbfgsHistorySize lbfgsLineSearchFn initParams'
    return $ CppOptimizerState opt v
    where
      initParams' = map toDependent $ flattenParameters initParams

data RmspropOptions = RmspropOptions
  { rmspropLr :: Double,
    rmspropAlpha :: Double,
    rmspropEps :: Double,
    rmspropWeightDecay :: Double,
    rmspropMomentum :: Double,
    rmspropCentered :: Bool
  }
  deriving (Show, Eq)

instance Default RmspropOptions where
  def =
    RmspropOptions
      { rmspropLr = 1e-2,
        rmspropAlpha = 0.99,
        rmspropEps = 1e-8,
        rmspropWeightDecay = 0,
        rmspropMomentum = 0,
        rmspropCentered = False
      }

instance CppOptimizer RmspropOptions where
  initOptimizer opt@RmspropOptions {..} initParams = do
    v <- cast7 LibTorch.rmsprop rmspropLr rmspropAlpha rmspropEps rmspropWeightDecay rmspropMomentum rmspropCentered initParams'
    return $ CppOptimizerState opt v
    where
      initParams' = map toDependent $ flattenParameters initParams

data SGDOptions = SGDOptions
  { sgdLr :: Double,
    sgdMomentum :: Double,
    sgdDampening :: Double,
    sgdWeightDecay :: Double,
    sgdNesterov :: Bool
  }
  deriving (Show, Eq)

instance Default SGDOptions where
  def =
    SGDOptions
      { sgdLr = 1e-3,
        sgdMomentum = 0,
        sgdDampening = 0,
        sgdWeightDecay = 0,
        sgdNesterov = False
      }

instance CppOptimizer SGDOptions where
  initOptimizer opt@SGDOptions {..} initParams = do
    v <- cast6 LibTorch.sgd sgdLr sgdMomentum sgdDampening sgdWeightDecay sgdNesterov initParams'
    return $ CppOptimizerState opt v
    where
      initParams' = map toDependent $ flattenParameters initParams

saveState :: CppOptimizerState option -> FilePath -> IO ()
saveState (CppOptimizerState _ optimizer) file = cast2 LibTorch.save optimizer file

loadState :: CppOptimizerState option -> FilePath -> IO ()
loadState (CppOptimizerState _ optimizer) file = cast2 LibTorch.load optimizer file
