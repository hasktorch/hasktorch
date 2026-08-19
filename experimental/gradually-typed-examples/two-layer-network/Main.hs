{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE NamedFieldPuns #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}

module Main where

import Control.Lens (element, (^?))
import Control.Monad (replicateM)
import Control.Monad.Cont (runContT)
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
import System.Random (Random, RandomGen, mkStdGen)
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
-- It is a product type of two fully connected layers, @fstLayer@ and @sndLayer@,
-- an activation function, @activation@, and a dropout layer, @dropout@.
newtype TwoLayerNetwork fstLayer activation dropout sndLayer
  = TwoLayerNetwork (ModelStack '[fstLayer, activation, dropout, sndLayer])
  deriving stock (Generic)

-- | The specification of a two-layer network is the product of the specifications of its layers.
type instance
  ModelSpec (TwoLayerNetwork fstLayer activation dropout sndLayer) =
    TwoLayerNetwork (ModelSpec fstLayer) (ModelSpec activation) (ModelSpec dropout) (ModelSpec sndLayer)

-- | To initialize a two-layer network, we need a 'HasInitialize' instance.
-- The instance is auto-generated from the instance for the 'ModelStack' data type.
instance
  HasInitialize (ModelStack '[fstLayer, activation, dropout, sndLayer]) generatorDevice (ModelStack '[fstLayer', activation', dropout', sndLayer']) generatorOutputDevice =>
  HasInitialize (TwoLayerNetwork fstLayer activation dropout sndLayer) generatorDevice (TwoLayerNetwork fstLayer' activation' dropout' sndLayer') generatorOutputDevice

-- | For conversion of a two-layer network into a state dictionary and back,
-- we need a 'HasStateDict' instance.
-- The instance is auto-generated from the 'HasStateDict' instances of the individual layer data types.
instance
  (HasStateDict fstLayer, HasStateDict activation, HasStateDict dropout, HasStateDict sndLayer) =>
  HasStateDict (TwoLayerNetwork fstLayer activation dropout sndLayer)

-- | The 'HasForward' instance defines the 'forward' pass of the two-layer network and the loss.
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

main :: IO ()
main = do
  let seed = 31415
      device = SDevice SCPU
      inputDim = SNoName :&: SSize @1
      outputDim = SNoName :&: SSize @1
      hiddenDim = SNoName :&: SSize @100

  let mkModelSpec hasGradient dropout' =
        "myFirstTwoLayerNetwork"
          ::> TwoLayerNetwork
            ( ModelStack
                ( "fstFullyConnectedLayer" ::> linearSpec SWithBias (SGradient hasGradient) device (SDataType SFloat) inputDim hiddenDim,
                  Tanh,
                  dropout',
                  "sndFullyConnectedLayer" ::> linearSpec SWithBias (SGradient hasGradient) device (SDataType SFloat) hiddenDim outputDim
                )
            )
      -- during training, we need to turn dropout on and keep track of the gradient
      trainingModelSpec = mkModelSpec SWithGradient (Dropout 0.1)
      -- during evaluation, we don't need to turn dropout on, nor do we need to keep track of the gradient
      evaluationModelSpec = mkModelSpec SWithoutGradient ()

  -- create a Torch random generator from the seed
  g0 <- sMkGenerator device seed

  -- initialize the model from the model specification
  (model, g1) <- initialize trainingModelSpec g0

  -- buffered collation function that converts a stream of examples into one of batched tensors
  let collate' =
        let batchSize = 100
            collateFn chunk =
              let (xs, ys) = unzip chunk
                  xs' = VS.singleton <$> xs
                  ys' = VS.singleton <$> ys
                  sToTensor' = sToTensor (SGradient SWithoutGradient) (SLayout SDense) device
               in (,) <$> sToTensor' xs' <*> sToTensor' ys'
         in bufferedCollate (P.bounded 10) batchSize collateFn

  let numEpochs = 100
      learningRateSchedule =
        let maxLearningRate = 1e-2
            finalLearningRate = 1e-4
            numWarmupEpochs = 10
            numCooldownEpochs = 10
         in singleCycleLearningRateSchedule maxLearningRate finalLearningRate numEpochs numWarmupEpochs numCooldownEpochs

  (trainingData, evaluationData, streamingState) <-
    evalStateT
      ( (,,) <$> mkSincData "training" 10000 <*> mkSincData "evaluation" 500
          -- configure the data loader with 1 worker thread and for random shuffling
          <*> gets (\stdGen -> (datasetOpts 1) {shuffle = Shuffle stdGen})
      )
      (mkStdGen 13)

  -- create an Adam optimizer from the model
  optim <- liftIO $ mkAdam defaultAdamOptions {learningRate = 1e-4} model

  let -- one epoch of training and evaluation
      step (streamingState', g) epoch = do
        -- let learningRate = learningRateSchedule epoch
        -- setLearningRate optim learningRate

        -- helper function for streaming
        let go streamingState'' data' monitor' closure' = do
              (stream, shuffle) <- P.lift $ streamFromMap streamingState'' data'
              trainingLoss <- P.lift $ do
                batchedStream <- collate' stream
                lift $ closure' batchedStream
              case trainingLoss of
                Left g' -> pure (g', shuffle)
                Right (loss, g') -> do
                  P.yield (monitor' (fromTensor loss) epoch)
                  pure (g', shuffle)

        -- train for one epoch on the training set
        (g', shuffle) <- go streamingState' trainingData TrainingMonitor $ \batchedStream ->
          train optim trainingModelSpec batchedStream g

        -- evaluate on the evaluation set
        (g'', shuffle') <- go streamingState' {shuffle} evaluationData EvaluationMonitor $ \batchedStream -> do
          stateDict <- getStateDict optim
          evaluationModel <- flip evalStateT stateDict $ fromStateDict evaluationModelSpec mempty
          eval evaluationModel batchedStream g'

        pure (streamingState' {shuffle = shuffle'}, g'')

  let -- initialize the training loop
      init' = pure (streamingState, g1)

  let -- finalize the training loop
      done (_streamingState', _g2) = pure ()

  -- run the training loop
  flip runContT pure . P.runEffect $
    P.foldM step init' done (P.each [1 .. numEpochs]) P.>-> monitor

  -- save the model's state dictionary to a file
  stateDict' <- getStateDict optim
  stateDictToFile stateDict' "twoLayerNetwork.pt"
