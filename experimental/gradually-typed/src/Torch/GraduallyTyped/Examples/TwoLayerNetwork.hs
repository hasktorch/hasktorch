{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE NamedFieldPuns #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}

module Torch.GraduallyTyped.Examples.TwoLayerNetwork where

import Control.Lens (element, (^?))
import Control.Monad (replicateM)
import Control.Monad.Cont (ContT, runContT)
import Control.Monad.IO.Class (MonadIO (liftIO))
import Control.Monad.State (MonadState (..), StateT, evalStateT, gets)
import Control.Monad.Trans (MonadTrans (..))
import Data.Random.Normal (normal)
import qualified Data.Set as Set
import Data.Text (Text)
import qualified Data.Vector.Sized as VS
import GHC.Generics (Generic)
import qualified Pipes as P
import qualified Pipes.Concurrent as P
import qualified Pipes.Prelude as P
import System.Random (Random, RandomGen, getStdGen)
import Torch.GraduallyTyped

-- | Compute the sine cardinal (sinc) function,
-- see https://mathworld.wolfram.com/SincFunction.html.
sinc :: Floating a => a -> a
sinc a = Prelude.sin a / a

-- | Compute the sine cardinal (sinc) function and add normally distributed noise
-- of strength epsilon. We use the 'normal' function from 'Data.Random.Normal'
-- which requires a random number generator with 'RandomGen' instance.
noisySinc :: (Floating a, Random a, RandomGen g) => a -> a -> g -> (a, g)
noisySinc eps a g = let (noise, g') = normal g in (sinc a + eps * noise, g')

-- | Datatype to represent a dataset of sine cardinal (sinc) inputs and outputs.
-- The 'name' field is used to identify the split of the dataset.
data SincData = SincData {name :: Text, unSincData :: [(Float, Float)]} deriving (Eq, Ord)

-- | Create a dataset of noisy sine cardinal (sinc) values of a desired size.
mkSincData ::
  (RandomGen g, Monad m) =>
  -- | name of the dataset
  Text ->
  -- | number of samples
  Int ->
  -- | dataset in the state monad over the random generator
  StateT g m SincData
mkSincData name' size =
  let next' = do
        x <- (* 20) <$> state normal
        y <- state (noisySinc 0.05 x)
        pure (x, y)
   in SincData name' <$> replicateM size next'

-- | 'Dataset' instance used for streaming sine cardinal (sinc) examples.
instance Dataset IO SincData Int (Float, Float) where
  getItem (SincData _ d) k = maybe (fail "invalid key") pure $ d ^? element k
  keys (SincData _ d) = Set.fromList [0 .. Prelude.length d -1]

-- | Data type to represent a simple two-layer neural network.
-- It is a product type of two layer types, @fstLayer@ and @sndLayer@,
-- an activation function, @activation@, and a dropout layer, @dropout@.
newtype TwoLayerNetwork fstLayer activation dropout sndLayer
  = TwoLayerNetwork (ModelStack '[fstLayer, activation, dropout, sndLayer])
  deriving stock (Generic)

-- | The specification of a two-layer network is the product of the
-- specifications of its two layers and the activation function.
type instance
  ModelSpec (TwoLayerNetwork fstLayer activation dropout sndLayer) =
    TwoLayerNetwork (ModelSpec fstLayer) (ModelSpec activation) (ModelSpec dropout) (ModelSpec sndLayer)

-- | To initialize a two-layer network,
-- we need its specification and a random generator.
-- The random generator is used to initialize the weights of the network.
-- The specification is used to determine the properties of the two neural layers,
-- the activation function, and the dropout layer.
-- The four components are initialized separately and then combined into a single
-- network.
instance
  HasInitialize
    (ModelStack '[fstLayer, activation, dropout, sndLayer])
    generatorDevice
    (ModelStack '[fstLayer', activation', dropout', sndLayer'])
    generatorOutputDevice =>
  HasInitialize
    (TwoLayerNetwork fstLayer activation dropout sndLayer)
    generatorDevice
    (TwoLayerNetwork fstLayer' activation' dropout' sndLayer')
    generatorOutputDevice

-- | @HasStateDict@ instance for a two-layer network.
-- It allows for conversion of a two-layer network into a state dictionary and back.
--
-- To create a two-layer network from a state dictionary,
-- we need to first create its two neural layers, the activation function, and the dropout layer from the state dictionary.
-- Afterwards, we combine the four components into a single network.
--
-- The state dictionary of the two-layer network is the union of the
-- state dictionaries its layers.
instance
  (HasStateDict fstLayer, HasStateDict activation, HasStateDict dropout, HasStateDict sndLayer) =>
  HasStateDict (TwoLayerNetwork fstLayer activation dropout sndLayer)

-- | Specifies the type of the two-layer network.
type TwoLayerNetworkF gradient hasDropout device dataType inputDim outputDim hiddenDim =
  NamedModel
    ( TwoLayerNetwork
        (NamedModel (GLinearF 'WithBias gradient device dataType inputDim hiddenDim))
        Tanh
        (TLNDropoutF hasDropout)
        (NamedModel (GLinearF 'WithBias gradient device dataType hiddenDim outputDim))
    )

-- | Specifies the type of the dropout layer
type family TLNDropoutF hasDropout where
  TLNDropoutF 'WithDropout = Dropout
  TLNDropoutF 'WithoutDropout = ()

-- | Creates a value that specifies the parameters of a two-layer neural network.
twoLayerNetworkSpec ::
  forall gradient hasDropout device dataType inputDim outputDim hiddenDim.
  -- | whether or not to compute gradients for the parametrs
  SGradient gradient ->
  -- | whether or not to use dropout
  SHasDropout hasDropout ->
  -- | which device to use
  SDevice device ->
  -- | which data type to use
  SDataType dataType ->
  -- | input dimension
  SDim inputDim ->
  -- | output dimension
  SDim outputDim ->
  -- | hidden dimension
  SDim hiddenDim ->
  -- | dropout rate
  Double ->
  -- | specification for the network
  ModelSpec (TwoLayerNetworkF gradient hasDropout device dataType inputDim outputDim hiddenDim)
twoLayerNetworkSpec gradient hasDropout device dataType inputDim outputDim hiddenDim dropoutP =
  "twoLayerNetwork"
    ::> TwoLayerNetwork
      ( ModelStack
          ( "fstLayer" ::> linearSpec SWithBias gradient device dataType inputDim hiddenDim,
            Tanh,
            case hasDropout of
              SWithDropout -> Dropout dropoutP
              SWithoutDropout -> (),
            "sndLayer" ::> linearSpec SWithBias gradient device dataType hiddenDim outputDim
          )
      )

-- | 'HasForward' instance used to define the forward pass of the model.
-- The forward pass is defined as the composition of the forward passes of the two layers.
-- A forward pass is a function that takes a two-layer network, an input tensor, and a random generator,
-- and returns the output tensor and the updated generator.
-- A nonlinearity, @tanh@, is applied to the output of the first layer.
-- The input to the second layer is the output of the nonlinearity.
instance
  ( HasForward
      (ModelStack '[fstLayer, activation, dropout, sndLayer])
      input
      generatorDevice
      prediction
      generatorOutputDevice,
    HasForward MSELoss (prediction, target) generatorOutputDevice output generatorOutputDevice
  ) =>
  HasForward
    (TwoLayerNetwork fstLayer activation dropout sndLayer)
    (input, target)
    generatorDevice
    output
    generatorOutputDevice
  where
  forward (TwoLayerNetwork modelStack) (input, target) g = do
    (prediction, g') <- forward modelStack input g
    (loss, g'') <- forward MSELoss (prediction, target) g'
    pure (loss, g'')

-- | Data type for monitoring the training and evaluation losses.
data Monitor
  = -- | monitor for training loss
    TrainingMonitor {mtLoss :: Float, mtEpoch :: Int}
  | -- | monitor for evaluation loss
    EvaluationMonitor {meLoss :: Float, meEpoch :: Int}
  deriving stock (Show)

-- | A simple monitor that prints the training and evaluation losses to stdout.
monitor :: MonadIO m => P.Consumer Monitor m r
monitor = P.map show P.>-> P.stdoutLn'

-- | Collate a stream of examples into batches.
-- The returned batches are of the form @(input, target)@,
-- where @(input, target)@ is a pair of tensors.
-- @input@ and @target@ are both of shape:
--
-- > Dim (SName @"*") (UncheckedSize batchSize) :|: Dim (SName @"*") (SSize @1) :|: SNil
--
-- where @batchSize@ is the number of examples in the batch.
collate ::
  forall device r input target.
  ( input
      ~ Tensor
          ('Gradient 'WithoutGradient)
          ('Layout 'Dense)
          device
          ('DataType 'Float)
          ( 'Shape
              '[ 'Dim ('Name "*") 'UncheckedSize, 'Dim ('Name "*") ('Size 1)]
          ),
    target
      ~ Tensor
          ('Gradient 'WithoutGradient)
          ('Layout 'Dense)
          device
          ('DataType 'Float)
          ( 'Shape
              '[ 'Dim ('Name "*") 'UncheckedSize, 'Dim ('Name "*") ('Size 1)]
          )
  ) =>
  SDevice device ->
  Int ->
  P.ListT IO (Float, Float) ->
  ContT r IO (P.ListT IO (input, target))
collate device batchSize =
  let collateFn chunk =
        let (xs, ys) = unzip chunk
            xs' = VS.singleton <$> xs
            ys' = VS.singleton <$> ys
            sToTensor' = sToTensor (SGradient SWithoutGradient) (SLayout SDense) device
         in (,) <$> sToTensor' xs' <*> sToTensor' ys'
   in bufferedCollate (P.bounded 1) batchSize collateFn

-- | Run the two-layer network training loop on a toy dataset.
runTwoLayerNetworkExample :: IO ()
runTwoLayerNetworkExample = do
  let -- seed for the random number generator
      seed = 0

  let -- compute device
      device = SDevice SCPU

  let -- input dimension of the network
      inputDim = SNoName :&: SSize @1
      -- output dimension of the network
      outputDim = SNoName :&: SSize @1
      -- hidden dimension of the network
      hiddenDim = SNoName :&: SSize @100

  let -- create the model specifications
      mkModelSpec hasGradient hasDropout =
        twoLayerNetworkSpec
          (SGradient hasGradient)
          hasDropout
          device
          (SDataType SFloat)
          inputDim
          outputDim
          hiddenDim
          0.1
      -- during training, we need to turn dropout on and keep track of the gradient
      trainingModelSpec = mkModelSpec SWithGradient SWithDropout
      -- during evaluation, we don't need to turn dropout on, nor do we need to keep track of the gradient
      evaluationModelSpec = mkModelSpec SWithoutGradient SWithoutDropout

  -- create a Torch random generator from the seed
  g0 <- sMkGenerator device seed

  -- initialize the model from the model specification using the generator
  (model, g1) <- initialize trainingModelSpec g0

  -- define collation function
  let collate' =
        let -- batch size
            batchSize = 100
         in Torch.GraduallyTyped.Examples.TwoLayerNetwork.collate device batchSize

  let -- total number of epochs
      numEpochs = 100
      -- learning rate schedule
      learningRateSchedule =
        let -- peak learning rate after warmup
            maxLearningRate = 1e-2
            -- learning rate at the end of the schedule
            finalLearningRate = 1e-4
            -- warmup epochs
            numWarmupEpochs = 10
            -- cooldown epochs
            numCooldownEpochs = 10
         in singleCycleLearningRateSchedule maxLearningRate finalLearningRate numEpochs numWarmupEpochs numCooldownEpochs

  -- create the dataset(s) using a Haskell random generator
  (trainingData, evaluationData, streamingState) <-
    getStdGen
      >>= evalStateT
        ( (,,)
            -- create a dataset of 10000 unique training examples
            <$> mkSincData "training" 10000
              -- create a dataset of 500 unique evaluation examples
              <*> mkSincData "evaluation" 500
              -- configure the data loader for random shuffling
              <*> gets (\stdGen -> (datasetOpts 1) {shuffle = Shuffle stdGen})
        )

  -- create an Adam optimizer with learning rate 1e-4 using the model.
  -- the optimizer is responsible for computing the gradient of the loss
  -- with respect to the model parameters and updating the model parameters.
  optim <- liftIO $ mkAdam defaultAdamOptions {learningRate = 1e-4} model

  let -- one epoch of training and evaluation
      step (streamingState', g) epoch = do
        -- let learningRate = learningRateSchedule epoch
        -- ATen.setLearningRate optim learningRate

        -- train for one epoch on the training set
        (g', shuffle) <- do
          (trainingStream, shuffle) <- P.lift $ streamFromMap streamingState' trainingData
          trainingLoss <- P.lift $ do
            batchedStream <- collate' trainingStream
            lift $ train optim trainingModelSpec batchedStream g
          -- returned is either the original generator or
          -- a pair of a generator and the average training loss
          case trainingLoss of
            Left g' -> pure (g', shuffle)
            Right (loss, g') -> do
              P.yield (TrainingMonitor (fromTensor loss) epoch)
              pure (g', shuffle)

        -- evaluate on the evaluation set
        (g'', shuffle') <- do
          (evalStream, shuffle') <- P.lift $ streamFromMap streamingState' {shuffle} evaluationData
          evalLoss <- P.lift $ do
            batchedStream <- collate' evalStream
            lift $ do
              stateDict <- getStateDict optim
              evaluationModel <- flip evalStateT stateDict $ fromStateDict evaluationModelSpec mempty
              eval evaluationModel batchedStream g'
          -- returned is either the original generator or
          -- a pair of a generator and the average evaluation loss
          case evalLoss of
            Left g'' -> pure (g'', shuffle')
            Right (loss, g'') -> do
              P.yield (EvaluationMonitor (fromTensor loss) epoch)
              pure (g'', shuffle')

        pure (streamingState' {shuffle = shuffle'}, g'')

  let -- initialize the training loop
      init' = pure (streamingState, g1)

  let -- finalize the training loop
      done (_streamingState', _g2) = pure ()

  -- run the training loop
  flip runContT pure . P.runEffect $
    P.foldM step init' done (P.each [1 .. numEpochs])
      P.>-> monitor

  -- save the model's state dictionary to a file
  stateDict' <- getStateDict optim
  stateDictToFile stateDict' "twoLayerNetwork.pt"
