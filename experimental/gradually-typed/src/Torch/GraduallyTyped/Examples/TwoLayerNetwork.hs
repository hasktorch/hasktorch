{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
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
import Control.Monad.Catch (MonadThrow)
import Control.Monad.Cont (runContT)
import Control.Monad.Indexed ((>>>=))
import Control.Monad.Indexed.State (IxStateT (..))
import Control.Monad.Indexed.Trans (IxMonadTrans (ilift))
import Control.Monad.State (MonadState (..), StateT, evalStateT, execStateT, gets)
import Control.Monad.Trans (MonadTrans (..))
import Data.Functor.Indexed (IxPointed (ireturn), (<<$>>), (<<*>>))
import qualified Data.Map as Map
import Data.Maybe (listToMaybe)
import Data.Random.Normal (normal)
import qualified Data.Set as Set
import Data.Text (Text)
import Data.Word (Word64)
import Foreign.ForeignPtr (ForeignPtr)
import qualified Pipes as P
import qualified Pipes.Concurrent as P
import qualified Pipes.Prelude as P
import System.Random (Random, RandomGen, getStdGen)
import Torch.GraduallyTyped
import qualified Torch.Internal.Cast as ATen
import qualified Torch.Internal.Class as ATen
import Torch.Internal.GC (unsafeThrowableIO)
import qualified Torch.Internal.Managed.Native as ATen
import qualified Torch.Internal.Managed.Optim as ATen
import qualified Torch.Internal.Type as ATen

-- | Compute the sine cardinal (sinc) function,
-- see https://mathworld.wolfram.com/SincFunction.html
sinc :: Floating a => a -> a
sinc a = Prelude.sin a / a

-- | Compute the sine cardinal (sinc) function and add normally distributed noise
-- of strength epsilon
noisySinc :: (Floating a, Random a, RandomGen g) => a -> a -> g -> (a, g)
noisySinc eps a g = let (noise, g') = normal g in (sinc a + eps * noise, g')

-- | Datatype to represent a dataset of sine cardinal (sinc) inputs and outputs
data SincData = SincData {name :: Text, unSincData :: [(Float, Float)]} deriving (Eq, Ord)

-- | Create a dataset of noisy sine cardinal (sinc) values of a desired size
mkSincData :: (RandomGen g, Monad m) => Text -> Int -> StateT g m SincData
mkSincData name' size =
  let next' = do
        x <- (* 20) <$> state normal
        y <- state (noisySinc 0.05 x)
        pure (x, y)
   in SincData name' <$> replicateM size next'

-- | 'Dataset' instance used for streaming sine cardinal (sinc) examples
instance Dataset IO SincData Int (Float, Float) where
  getItem (SincData _ d) k = maybe (fail "invalid key") pure $ d ^? element k
  keys (SincData _ d) = Set.fromList [0 .. Prelude.length d -1]

-- | Data type to represent a simple two-layer neural network.
data TwoLayerNetwork fstLayer sndLayer = TwoLayerNetwork
  { fstLayer :: fstLayer,
    sndLayer :: sndLayer
  }

type instance
  ModelSpec (TwoLayerNetwork fstLayer sndLayer) =
    TwoLayerNetwork (ModelSpec fstLayer) (ModelSpec sndLayer)

instance
  ( HasInitialize fstLayer generatorDevice fstLayer generatorDevice,
    HasInitialize sndLayer generatorDevice sndLayer generatorDevice
  ) =>
  HasInitialize
    (TwoLayerNetwork fstLayer sndLayer)
    generatorDevice
    (TwoLayerNetwork fstLayer sndLayer)
    generatorDevice
  where
  initialize (TwoLayerNetwork fstLayerSpec sndLayerSpec) =
    runIxStateT $
      TwoLayerNetwork
        <<$>> (IxStateT . initialize $ fstLayerSpec)
        <<*>> (IxStateT . initialize $ sndLayerSpec)

instance
  (HasStateDict fstLayer, HasStateDict sndLayer) =>
  HasStateDict (TwoLayerNetwork fstLayer sndLayer)
  where
  fromStateDict (TwoLayerNetwork fstLayerSpec sndLayerSpec) k =
    TwoLayerNetwork <$> fromStateDict fstLayerSpec k <*> fromStateDict sndLayerSpec k
  toStateDict k TwoLayerNetwork {..} = do
    () <- toStateDict k fstLayer
    () <- toStateDict k sndLayer
    pure ()

type TwoLayerNetworkF gradient device dataType inputDim outputDim hiddenDim =
  NamedModel
    ( TwoLayerNetwork
        (FstLayerF gradient device dataType inputDim hiddenDim)
        (SndLayerF gradient device dataType outputDim hiddenDim)
    )

-- | Specifies the first layer of the neural network.
type FstLayerF gradient device dataType inputDim hiddenDim =
  NamedModel
    ( GLinear
        (NamedModel (LinearWeightF gradient device dataType inputDim hiddenDim))
        (NamedModel (LinearBiasF 'WithBias gradient device dataType hiddenDim))
    )

-- | Specifies the second layer of the neural network.
type SndLayerF gradient device dataType outputDim hiddenDim =
  NamedModel
    ( GLinear
        (NamedModel (LinearWeightF gradient device dataType hiddenDim outputDim))
        (NamedModel (LinearBiasF 'WithBias gradient device dataType outputDim))
    )

-- | Specify the parameters of the two-layer neural network
twoLayerNetworkSpec ::
  forall gradient device dataType inputDim outputDim hiddenDim.
  -- | whether or not to compute gradients for the parametrs
  SGradient gradient ->
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
  -- | specification for the network
  ModelSpec (TwoLayerNetworkF gradient device dataType inputDim outputDim hiddenDim)
twoLayerNetworkSpec gradient device dataType inputDim outputDim hiddenDim =
  NamedModel "twoLayerNetwork" $
    TwoLayerNetwork
      (NamedModel "fstLayer" $ linearSpec SWithBias gradient device dataType inputDim hiddenDim)
      (NamedModel "sndLayer" $ linearSpec SWithBias gradient device dataType hiddenDim outputDim)

-- | 'HasForward' instance used to define the forward pass of the model
instance
  ( HasForward
      fstLayer
      (Tensor gradient layout dataType device shape)
      generatorDevice
      (Tensor gradient0 layout0 dataType0 device0 shape0)
      generatorDevice0,
    HasForward
      sndLayer
      (Tensor gradient0 layout0 dataType0 device0 shape0)
      generatorDevice0
      output
      generatorOutputDevice
  ) =>
  HasForward
    (TwoLayerNetwork fstLayer sndLayer)
    (Tensor gradient layout dataType device shape)
    generatorDevice
    output
    generatorOutputDevice
  where
  forward TwoLayerNetwork {..} input =
    runIxStateT $
      ireturn input
        >>>= IxStateT . forward fstLayer
        >>>= ireturn . Torch.GraduallyTyped.tanh
        >>>= IxStateT . forward sndLayer

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

-- | Create a new Adam optimizer.
mkAdam ::
  -- | Adam options
  AdamOptions ->
  -- | initial model parameters
  [ForeignPtr ATen.Tensor] ->
  -- | Adam optimizer
  IO (ForeignPtr ATen.Optimizer)
mkAdam AdamOptions {..} =
  ATen.cast7 ATen.adam learningRate beta1 beta2 epsilon weightDecay amsgrad

mseLoss ::
  forall m gradient layout device dataType shape gradient' layout' device' dataType' shape'.
  MonadThrow m =>
  -- | prediction tensor
  Tensor gradient layout device dataType shape ->
  -- | target tensor
  Tensor gradient' layout' device' dataType' shape' ->
  -- | output tensor
  m
    ( Tensor
        (gradient <|> gradient')
        (layout <+> layout')
        (device <+> device')
        (dataType <+> dataType')
        (Seq (shape <+> shape') ('Shape '[]))
    )
mseLoss prediction target =
  unsafeThrowableIO $
    ATen.cast3
      ATen.mse_loss_ttl
      prediction
      target
      (1 :: Int) -- reduce mean

computeLoss ::
  _ =>
  -- | device
  SDevice device ->
  -- | model specification
  ModelSpec model ->
  -- | model state dict
  StateDict ->
  -- | input
  [[Float]] ->
  -- | expected output
  [[Float]] ->
  -- | random generator
  Generator generatorDevice ->
  -- | loss
  m (_, Generator generatorOutputDevice)
computeLoss device modelSpec stateDict input expectedOutput =
  runIxStateT $
    ilift
      ( let inputTensor = sToTensor (SGradient SWithoutGradient) (SLayout SDense) device input
            expectedOutputTensor = sToTensor (SGradient SWithoutGradient) (SLayout SDense) device expectedOutput
            model = flip evalStateT stateDict $ fromStateDict modelSpec mempty
         in (,,) <$> inputTensor <*> expectedOutputTensor <*> model
      )
      >>>= ( \(inputTensor, expectedOutputTensor, model) ->
               ireturn inputTensor
                 >>>= IxStateT . forward model
                 >>>= ilift . (`mseLoss` expectedOutputTensor)
           )

train ::
  _ =>
  -- | random seed for the training
  Word64 ->
  -- | device
  SDevice device ->
  -- | model specification
  ModelSpec model ->
  -- | model state dict
  StateDict ->
  -- | learning rate
  Double ->
  -- | stream of training examples
  P.ListT IO ([[Float]], [[Float]], int) ->
  IO StateDict
train seed device modelSpec stateDict learningRate examples = do
  g <- sMkGenerator device seed
  optim <- mkAdam defaultAdamOptions {learningRate} (Map.elems stateDict)
  let step g' (x, y, _iter) = do
        let f :: ForeignPtr ATen.TensorList -> IO (ForeignPtr ATen.Tensor)
            f tensorList = do
              stateDict' :: StateDict <- ATen.uncast tensorList (pure . Map.fromList . zip (Map.keys stateDict))
              UnsafeTensor t <- fst <$> computeLoss device modelSpec stateDict' x y g'
              pure t
        _loss :: ForeignPtr ATen.Tensor <- ATen.cast2 ATen.step optim f
        print $ UnsafeTensor _loss
        pure g'
      init' = pure g
      done = pure
  _ <- P.foldM step init' done . P.enumerate $ examples
  pure stateDict

-- | Single-cycle learning rate schedule.
-- See, for instance, https://arxiv.org/abs/1803.09820.
singleCycleLearningRateSchedule :: Double -> Double -> Int -> Int -> Int -> Int -> Double
singleCycleLearningRateSchedule maxLearningRate finalLearningRate numEpochs numWarmupEpochs numCooldownEpochs epoch
  | epoch <= 0 = 0.0
  | 0 < epoch && epoch <= numWarmupEpochs =
    let a :: Double = fromIntegral epoch / fromIntegral numWarmupEpochs
     in a * maxLearningRate
  | numWarmupEpochs < epoch && epoch < numEpochs - numCooldownEpochs =
    let a :: Double =
          fromIntegral (numEpochs - numCooldownEpochs - epoch)
            / fromIntegral (numEpochs - numCooldownEpochs - numWarmupEpochs)
     in a * maxLearningRate + (1 - a) * finalLearningRate
  | otherwise = finalLearningRate

runTwoLayerNetworkExample :: IO ()
runTwoLayerNetworkExample = do
  let -- seed for random number generator
      seed = 0

  let -- compute device
      device = SDevice SCPU

  let -- input dimension of the network
      inputDim = SName @"*" :&: SSize @1
      -- output dimension of the network
      outputDim = SName @"*" :&: SSize @1
      -- hidden dimension of the network
      hiddenDim = SName @"*" :&: SSize @100
  let modelSpec =
        twoLayerNetworkSpec
          (SGradient SWithGradient)
          device
          (SDataType SFloat)
          inputDim
          outputDim
          hiddenDim
  g <- sMkGenerator device seed
  (model, _) <- initialize modelSpec g

  let -- batch size
      batchSize = 100

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

  let stepEpoch (stateDict, streamingState') epoch = do
        let learningRate = learningRateSchedule epoch
        (examples, shuffle) <- streamFromMap streamingState' trainingData
        let collateFn exs =
              let (xys, iters) = unzip exs
                  (xs, ys) = unzip xys
               in (,,) (pure <$> xs) (pure <$> ys) <$> listToMaybe iters
        stateDict' <-
          lift
            . train seed device modelSpec stateDict learningRate
            =<< bufferedCollate
              (P.bounded 1)
              batchSize
              collateFn
              ( P.Select $
                  P.zip
                    (P.enumerate examples)
                    (P.each [0 :: Int ..])
              )
        pure (stateDict', streamingState' {shuffle})
      initEpoch = do
        stateDict <- flip execStateT Map.empty $ toStateDict mempty model
        pure (stateDict, streamingState)
      doneEpoch (stateDict, _) = pure stateDict

  finalStateDict <- flip runContT pure . P.foldM stepEpoch initEpoch doneEpoch . P.each $ [1 .. numEpochs]

  pure ()
