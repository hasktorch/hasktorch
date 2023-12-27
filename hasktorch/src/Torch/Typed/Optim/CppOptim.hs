{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}

module Torch.Typed.Optim.CppOptim
  ( module Torch.Typed.Optim.CppOptim,
    AdagradOptions (..),
    AdamOptions (..),
    AdamwOptions (..),
    LbfgsOptions (..),
    RmspropOptions (..),
    SGDOptions (..),
  )
where

import Data.Default.Class
import Data.Foldable (for_)
import Data.Kind (Type)
import qualified Debug.Trace as Debug
import Foreign.ForeignPtr
import System.Mem (performGC)
import qualified Torch as TD
import Torch.HList
import Torch.Internal.Cast
import Torch.Internal.Class (Castable (..), CppObject (..), CppTuple2 (..), CppTuple3 (..), CppTuple4 (..))
import Torch.Internal.GC (mallocTrim)
import qualified Torch.Internal.Managed.Optim as LibTorch
import qualified Torch.Internal.Type as ATen
import Torch.Optim.CppOptim
  ( AdagradOptions (..),
    AdamOptions (..),
    AdamwOptions (..),
    LbfgsOptions (..),
    RmspropOptions (..),
    SGDOptions (..),
  )
import Torch.Typed.Autograd
import Torch.Typed.NN
import qualified Torch.Typed.Optim as Optim
import Torch.Typed.Parameter
import Torch.Typed.Tensor

type CppOptimizerRef = ForeignPtr ATen.Optimizer

data CppOptimizerState option (params :: [Type])
  = CppOptimizerState option CppOptimizerRef

data ToParameter = ToParameter

instance Apply' ToParameter (Tensor dev dtype shape) (Parameter dev dtype shape) where
  apply' _ (UnsafeMkTensor tensor) = UnsafeMkParameter . TD.IndependentTensor $ tensor

class CppOptimizer option where
  initOptimizer ::
    forall model tensors.
    ( Parameterized model,
      HMap' ToDependent (Parameters model) tensors,
      Castable (HList tensors) [TD.ATenTensor]
    ) =>
    option ->
    model ->
    IO (CppOptimizerState option (Parameters model))

  unsafeStep ::
    forall model dev dtype lossShape tensors res.
    ( Parameterized model,
      HMap' ToDependent (Parameters model) tensors,
      HMap' ToParameter tensors (Parameters model),
      Castable (HList tensors) [TD.ATenTensor]
    ) =>
    model ->
    CppOptimizerState option (Parameters model) ->
    Tensor dev dtype lossShape ->
    IO (model, CppOptimizerState option (Parameters model))
  unsafeStep model o@(CppOptimizerState _ optimizer) loss = do
    -- let deps :: HList tensors
    --    deps = hmap' ToDependent $ flattenParameters model

    -- Debug.traceIO $ "Tensors in: "
    -- cast deps (Debug.traceIO . show . map (TD.shape . TD.Unsafe))
    v :: [TD.ATenTensor] <- cast2 LibTorch.unsafeStep optimizer loss
    -- Debug.traceIO $ "Params returned by unsafeStep: "<>show (length v)

    newParamTensors :: HList tensors <- uncast v pure
    -- Debug.traceIO $ "Tensors out: "
    -- cast newParamTensors (Debug.traceIO . show . map (TD.shape . TD.Unsafe))
    let newParams = hmap' ToParameter newParamTensors
    let newModel = replaceParameters model newParams
    return (newModel, o)

instance CppOptimizer AdamOptions where
  initOptimizer opt@AdamOptions {..} model = do
    v <-
      cast7
        LibTorch.adam
        adamLr
        (fst adamBetas)
        (snd adamBetas)
        adamEps
        adamWeightDecay
        adamAmsgrad
        initParams'
    return $ CppOptimizerState opt v
    where
      initParams' = hmap' ToDependent $ flattenParameters model

instance CppOptimizer AdamwOptions where
  initOptimizer opt@AdamwOptions {..} model = do
    v <- cast7 LibTorch.adamw adamwLr (fst adamwBetas) (snd adamwBetas) adamwEps adamwWeightDecay adamwAmsgrad initParams'
    return $ CppOptimizerState opt v
    where
      initParams' = hmap' ToDependent $ flattenParameters model

instance CppOptimizer LbfgsOptions where
  initOptimizer opt@LbfgsOptions {..} model = do
    v <- cast8 LibTorch.lbfgs lbfgsLr lbfgsMaxIter lbfgsMaxEval lbfgsToleranceGrad lbfgsToleranceChange lbfgsHistorySize lbfgsLineSearchFn initParams'
    return $ CppOptimizerState opt v
    where
      initParams' = hmap' ToDependent $ flattenParameters model

instance CppOptimizer RmspropOptions where
  initOptimizer opt@RmspropOptions {..} model = do
    v <- cast7 LibTorch.rmsprop rmspropLr rmspropAlpha rmspropEps rmspropWeightDecay rmspropMomentum rmspropCentered initParams'
    return $ CppOptimizerState opt v
    where
      initParams' = hmap' ToDependent $ flattenParameters model

instance CppOptimizer SGDOptions where
  initOptimizer opt@SGDOptions {..} model = do
    v <- cast6 LibTorch.sgd sgdLr sgdMomentum sgdDampening sgdWeightDecay sgdNesterov initParams'
    return $ CppOptimizerState opt v
    where
      initParams' = hmap' ToDependent $ flattenParameters model

runStep ::
  ( CppOptimizer option,
    Parameterized model,
    HMap' ToDependent (Parameters model) tensors,
    HMap' ToParameter tensors (Parameters model),
    Castable (HList tensors) [TD.ATenTensor]
  ) =>
  model ->
  CppOptimizerState option (Parameters model) ->
  Optim.Loss dev dtype ->
  IO (model, CppOptimizerState option (Parameters model))
runStep model optim loss = do
  performGC
  mallocTrim 0
  unsafeStep model optim loss
