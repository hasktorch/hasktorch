{-# LANGUAGE DataKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE RoleAnnotations #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}

module Torch.GraduallyTyped.Optim where

import Control.Concurrent.STM.TVar (newTVarIO)
import Control.Monad.State (evalStateT, execStateT)
import qualified Data.Map as Map
import Foreign.ForeignPtr (ForeignPtr)
import Torch.GraduallyTyped.NN.Class (HasStateDict (..), ModelSpec, StateDict, StateDictKey)
import Torch.GraduallyTyped.Prelude (Catch)
import Torch.GraduallyTyped.Random (Generator (..), SGetGeneratorDevice, getGenPtr)
import Torch.GraduallyTyped.RequiresGradient (Gradient (Gradient), RequiresGradient (WithGradient))
import Torch.GraduallyTyped.Shape.Type (Shape (Shape))
import Torch.GraduallyTyped.Tensor.Type (Tensor (..))
import Torch.GraduallyTyped.Unify (type (<+>))
import qualified Torch.Internal.Cast as ATen
import qualified Torch.Internal.Class as ATen
import qualified Torch.Internal.Managed.Optim as ATen
import qualified Torch.Internal.Type as ATen

-- | Options for the Adam optimizer.
data AdamOptions = AdamOptions
  { -- | learning rate
    learningRate :: Double,
    -- | beta1
    beta1 :: Double,
    -- | beta2
    beta2 :: Double,
    -- | epsilon
    epsilon :: Double,
    -- | weight decay
    weightDecay :: Double,
    -- | use amsgrad
    amsgrad :: Bool
  }

-- | Default Adam options.
defaultAdamOptions :: AdamOptions
defaultAdamOptions =
  AdamOptions
    { learningRate = 0.001,
      beta1 = 0.9,
      beta2 = 0.999,
      epsilon = 1e-8,
      weightDecay = 0.0,
      amsgrad = False
    }

-- | Optimizer data type.
data Optimizer model where
  UnsafeOptimizer ::
    forall model.
    { optimizerStateDictKeys :: [StateDictKey],
      optimizerPtr :: ForeignPtr ATen.Optimizer
    } ->
    Optimizer model

type role Optimizer nominal

-- | Get the model state dictionary from an optimizer.
getStateDict ::
  forall model. Optimizer model -> IO StateDict
getStateDict UnsafeOptimizer {..} =
  do
    tPtrs :: [ForeignPtr ATen.Tensor] <- ATen.cast1 ATen.getParams optimizerPtr
    pure . Map.fromList $ zip optimizerStateDictKeys tPtrs

-- | Extract a model from an optimizer.
getModel ::
  forall model. HasStateDict model => ModelSpec model -> Optimizer model -> IO model
getModel modelSpec optimizer = do
  stateDict <- getStateDict optimizer
  flip evalStateT stateDict $ fromStateDict modelSpec mempty

-- | Create a new Adam optimizer from a model.
mkAdam ::
  forall model.
  HasStateDict model =>
  -- | Adam options
  AdamOptions ->
  -- | initial model
  model ->
  -- | Adam optimizer
  IO (Optimizer model)
mkAdam AdamOptions {..} model = do
  stateDict <- flip execStateT Map.empty $ toStateDict mempty model
  let (stateDictKeys, tPtrs) = unzip $ Map.toList stateDict
  UnsafeOptimizer stateDictKeys
    <$> ATen.cast7 ATen.adam learningRate beta1 beta2 epsilon weightDecay amsgrad tPtrs

-- | Perform one step of optimization.
stepWithGenerator ::
  forall model generatorDevice lossGradient lossLayout lossDataType lossDevice lossShape generatorOutputDevice.
  ( HasStateDict model,
    SGetGeneratorDevice generatorDevice,
    SGetGeneratorDevice generatorOutputDevice,
    Catch (lossShape <+> 'Shape '[]),
    Catch (lossGradient <+> 'Gradient 'WithGradient)
  ) =>
  -- | optimizer for the model
  Optimizer model ->
  -- | model specification
  ModelSpec model ->
  -- | loss function to minimize
  ( model ->
    Generator generatorDevice ->
    IO
      ( Tensor lossGradient lossLayout lossDataType lossDevice lossShape,
        Generator generatorOutputDevice
      )
  ) ->
  -- | random generator
  Generator generatorDevice ->
  -- | loss and updated generator
  IO (Tensor lossGradient lossLayout lossDataType lossDevice lossShape, Generator generatorOutputDevice)
stepWithGenerator UnsafeOptimizer {..} modelSpec lossFn (UnsafeGenerator tvar) =
  do
    genPtr <- getGenPtr tvar
    let rawLossFn :: ForeignPtr ATen.TensorList -> ForeignPtr ATen.Generator -> IO (ForeignPtr (ATen.StdTuple '(ATen.Tensor, ATen.Generator)))
        rawLossFn tlPtr genPtr'' = do
          g'' <- UnsafeGenerator <$> newTVarIO (Right genPtr'')
          stateDict' :: StateDict <- ATen.uncast tlPtr (pure . Map.fromList . zip optimizerStateDictKeys)
          model <- flip evalStateT stateDict' $ fromStateDict modelSpec mempty
          -- model <- getModel modelSpec optim
          (UnsafeTensor tPtr, UnsafeGenerator tvar''') <- lossFn model g''
          genPtr''' <- getGenPtr tvar'''
          ATen.cast (tPtr, genPtr''') pure
    (lossPtr, genPtr' :: ForeignPtr ATen.Generator) <- ATen.cast3 ATen.stepWithGenerator optimizerPtr genPtr rawLossFn
    g' <- newTVarIO (Right genPtr')
    pure (UnsafeTensor lossPtr, UnsafeGenerator g')
