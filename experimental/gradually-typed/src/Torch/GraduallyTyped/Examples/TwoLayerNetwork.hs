{-# LANGUAGE DataKinds #-}
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
{-# LANGUAGE RoleAnnotations #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}

module Torch.GraduallyTyped.Examples.TwoLayerNetwork where

import Control.Concurrent.STM.TVar (newTVarIO)
import Control.Lens (element, (^?))
import Control.Monad (replicateM)
import Control.Monad.Catch (MonadThrow)
import Control.Monad.Cont (runContT)
import Control.Monad.IO.Class (MonadIO (liftIO))
import Control.Monad.Indexed ((>>>=))
import Control.Monad.Indexed.State (IxStateT (..))
import Control.Monad.State (MonadState (..), StateT, evalStateT, execStateT, gets)
import Control.Monad.Trans (MonadTrans (..))
import Data.Functor.Indexed (IxPointed (ireturn), (<<$>>), (<<*>>))
import qualified Data.Map as Map
import Data.Random.Normal (normal)
import qualified Data.Set as Set
import Data.Text (Text)
import qualified Data.Vector.Sized as VS
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
-- It is a product type of two layer types, @fstLayer@ and @sndLayer@,
-- and an activation function, @activation@.
data TwoLayerNetwork fstLayer activation dropout sndLayer = TwoLayerNetwork
  { tlnFstLayer :: fstLayer,
    tlnActivation :: activation,
    tlnDropout :: dropout,
    tlnSndLayer :: sndLayer
  }

-- | The specification of a two-layer network is the product of the
-- specifications of its two layers and the activation function.
type instance
  ModelSpec (TwoLayerNetwork fstLayer activation dropout sndLayer) =
    TwoLayerNetwork (ModelSpec fstLayer) (ModelSpec activation) (ModelSpec dropout) (ModelSpec sndLayer)

-- | To initialize a two-layer network,
-- we need its specification and a random generator.
-- The random generator is used to initialize the weights of the network.
-- The specification is used to determine the properties of the two layers
-- and the activation function.
-- The three components initialized separately and then combined into a single
-- network.
instance
  ( HasInitialize fstLayer generatorDevice fstLayer generatorDevice,
    HasInitialize activation generatorDevice activation generatorDevice,
    HasInitialize dropout generatorDevice dropout generatorDevice,
    HasInitialize sndLayer generatorDevice sndLayer generatorDevice
  ) =>
  HasInitialize
    (TwoLayerNetwork fstLayer activation dropout sndLayer)
    generatorDevice
    (TwoLayerNetwork fstLayer activation dropout sndLayer)
    generatorDevice
  where
  initialize (TwoLayerNetwork fstLayerSpec activationSpec dropoutSpec sndLayerSpec) =
    runIxStateT $
      TwoLayerNetwork
        <<$>> (IxStateT . initialize $ fstLayerSpec)
        <<*>> (IxStateT . initialize $ activationSpec)
        <<*>> (IxStateT . initialize $ dropoutSpec)
        <<*>> (IxStateT . initialize $ sndLayerSpec)

-- | @HasStateDict@ instance for a two-layer network.
-- It allows for conversion of a two-layer network into a state dictionary and back.
--
-- To create a two-layer network from a state dictionary,
-- we need to first create its two layers and the activation function from the state dictionary.
-- Afterwards, we combine the three components into a single network.
--
-- The state dictionary of the two-layer network is the union of the
-- state dictionaries its layers.
instance
  (HasStateDict fstLayer, HasStateDict activation, HasStateDict dropout, HasStateDict sndLayer) =>
  HasStateDict (TwoLayerNetwork fstLayer activation dropout sndLayer)
  where
  fromStateDict (TwoLayerNetwork fstLayerSpec activationSpec dropoutSpec sndLayerSpec) k =
    TwoLayerNetwork
      <$> fromStateDict fstLayerSpec k
      <*> fromStateDict activationSpec k
      <*> fromStateDict dropoutSpec k
      <*> fromStateDict sndLayerSpec k
  toStateDict k TwoLayerNetwork {..} = do
    () <- toStateDict k tlnFstLayer
    () <- toStateDict k tlnActivation
    () <- toStateDict k tlnDropout
    () <- toStateDict k tlnSndLayer
    pure ()

-- | Specifies the type of the two-layer network.
type TwoLayerNetworkF gradient hasDropout device dataType inputDim outputDim hiddenDim =
  NamedModel
    ( TwoLayerNetwork
        (TLNFstLayerF gradient device dataType inputDim hiddenDim)
        TLNActivationF
        (TLNDropoutF hasDropout)
        (TLNSndLayerF gradient device dataType outputDim hiddenDim)
    )

-- | Specifies the type of the first layer of the neural network.
type TLNFstLayerF gradient device dataType inputDim hiddenDim =
  NamedModel
    ( GLinear
        (NamedModel (LinearWeightF gradient device dataType inputDim hiddenDim))
        (NamedModel (LinearBiasF 'WithBias gradient device dataType hiddenDim))
    )

-- | Specifies the type of the activation function
type TLNActivationF = Tanh

-- | Specifies the type of the dropout layer
type family TLNDropoutF hasDropout where
  TLNDropoutF 'WithDropout = Dropout
  TLNDropoutF 'WithoutDropout = ()

-- | Specifies the type of the second layer of the neural network.
type TLNSndLayerF gradient device dataType outputDim hiddenDim =
  NamedModel
    ( GLinear
        (NamedModel (LinearWeightF gradient device dataType hiddenDim outputDim))
        (NamedModel (LinearBiasF 'WithBias gradient device dataType outputDim))
    )

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
  NamedModel "twoLayerNetwork" $
    TwoLayerNetwork
      (NamedModel "fstLayer" $ linearSpec SWithBias gradient device dataType inputDim hiddenDim)
      Tanh
      ( case hasDropout of
          SWithDropout -> Dropout dropoutP
          SWithoutDropout -> ()
      )
      (NamedModel "sndLayer" $ linearSpec SWithBias gradient device dataType hiddenDim outputDim)

-- | 'HasForward' instance used to define the forward pass of the model.
-- The forward pass is defined as the composition of the forward passes of the two layers.
-- A forward pass is a function that takes a two-layer network, an input tensor, and a random generator,
-- and returns the output tensor and the updated generator.
-- A nonlinearity, @tanh@, is applied to the output of the first layer.
-- The input to the second layer is the output of the nonlinearity.
instance
  ( HasForward
      fstLayer
      (Tensor gradient layout dataType device shape)
      generatorDevice
      output0
      generatorDevice0,
    HasForward
      activation
      output0
      generatorDevice0
      output1
      generatorDevice1,
    HasForward
      dropout
      output1
      generatorDevice1
      output2
      generatorDevice2,
    HasForward
      sndLayer
      output2
      generatorDevice2
      output
      generatorOutputDevice
  ) =>
  HasForward
    (TwoLayerNetwork fstLayer activation dropout sndLayer)
    (Tensor gradient layout dataType device shape)
    generatorDevice
    output
    generatorOutputDevice
  where
  forward TwoLayerNetwork {..} input =
    runIxStateT $
      ireturn input
        >>>= IxStateT . forward tlnFstLayer
        >>>= IxStateT . forward tlnActivation
        >>>= IxStateT . forward tlnDropout
        >>>= IxStateT . forward tlnSndLayer

instance
  ( HasForward
      (TwoLayerNetwork fstLayer activation dropout sndLayer)
      input
      generatorDevice
      (Tensor gradient layout device dataType shape)
      generatorOutputDevice,
    Catch (shape <+> shape'),
    output
      ~ Tensor
          (gradient <|> gradient')
          (layout <+> layout')
          (device <+> device')
          (dataType <+> dataType')
          ('Shape '[])
  ) =>
  HasForward
    (TwoLayerNetwork fstLayer activation dropout sndLayer)
    (input, Tensor gradient' layout' device' dataType' shape')
    generatorDevice
    output
    generatorOutputDevice
  where
  forward model (input, expectedOutput) g = do
    (output, g') <- forward model input g
    loss <- output `mseLoss` expectedOutput
    pure (loss, g')

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

data Optimizer model where
  UnsafeOptimizer ::
    forall model.
    { optimizerStateDictKeys :: [StateDictKey],
      optimizerPtr :: ForeignPtr ATen.Optimizer
    } ->
    Optimizer model

type role Optimizer nominal

getStateDict ::
  forall model. Optimizer model -> IO StateDict
getStateDict UnsafeOptimizer {..} =
  do
    tPtrs :: [ForeignPtr ATen.Tensor] <- ATen.cast1 ATen.getParams optimizerPtr
    pure . Map.fromList $ zip optimizerStateDictKeys tPtrs

getModel ::
  forall model. HasStateDict model => ModelSpec model -> Optimizer model -> IO model
getModel modelSpec optimizer = do
  stateDict <- getStateDict optimizer
  flip evalStateT stateDict $ fromStateDict modelSpec mempty

-- | Create a new Adam optimizer.
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

-- | Compute the mean squared error between two tensors.
mseLoss ::
  forall m gradient layout device dataType shape gradient' layout' device' dataType' shape'.
  (MonadThrow m, Catch (shape <+> shape')) =>
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
        ('Shape '[])
    )
prediction `mseLoss` target =
  unsafeThrowableIO $
    ATen.cast3
      ATen.mse_loss_ttl
      prediction
      target
      (1 :: Int) -- reduce mean

stepWithGenerator ::
  ( HasStateDict model,
    SGetGeneratorDevice generatorDevice,
    SGetGeneratorDevice generatorOutputDevice,
    Catch (lossShape <+> 'Shape '[]),
    Catch (lossGradient <+> 'Gradient 'WithGradient)
  ) =>
  Optimizer model ->
  ModelSpec model ->
  (model -> Generator generatorDevice -> IO (Tensor lossGradient lossLayout lossDataType lossDevice lossShape, Generator generatorOutputDevice)) ->
  Generator generatorDevice ->
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

-- | Train the model for one epoch.
train ::
  forall m model input generatorDevice lossGradient lossLayout lossDataType lossDevice lossShape generatorOutputDevice.
  ( MonadIO m,
    HasStateDict model,
    HasForward
      model
      input
      generatorDevice
      (Tensor lossGradient lossLayout lossDataType lossDevice lossShape)
      generatorOutputDevice,
    HasForward
      model
      input
      generatorOutputDevice
      (Tensor lossGradient lossLayout lossDataType lossDevice lossShape)
      generatorOutputDevice,
    SGetGeneratorDevice generatorDevice,
    SGetGeneratorDevice generatorOutputDevice,
    Catch (lossShape <+> 'Shape '[]),
    Catch (lossGradient <+> 'Gradient 'WithGradient)
  ) =>
  -- | optimizer
  Optimizer model ->
  -- | model specification
  ModelSpec model ->
  -- | stream of training examples
  P.ListT m input ->
  -- | random generator
  Generator generatorDevice ->
  -- | returned is either the original generator or the loss and a new generator
  m
    ( Either
        (Generator generatorDevice)
        (Tensor lossGradient lossLayout lossDataType lossDevice lossShape, Generator generatorOutputDevice)
    )
train optim modelSpec examples g = do
  let producer = P.enumerate examples
  x <- P.next producer
  case x of
    Left _ -> pure . Left $ g
    Right (input, producer') -> do
      let step (_, g') input' = liftIO $ stepWithGenerator optim modelSpec (`forward` input') g'
          init' = liftIO $ stepWithGenerator optim modelSpec (`forward` input) g
          done = pure . Right
      P.foldM step init' done producer'

eval ::
  ( MonadIO m,
    HasStateDict model,
    HasForward model input generatorDevice (Tensor lossGradient lossLayout lossDataType lossDevice lossShape) generatorOutputDevice,
    HasForward model input generatorOutputDevice (Tensor lossGradient lossLayout lossDataType lossDevice lossShape) generatorOutputDevice
  ) =>
  model ->
  P.ListT m input ->
  Generator generatorDevice ->
  m
    ( Either
        (Generator generatorDevice)
        (Tensor lossGradient lossLayout lossDataType lossDevice lossShape, Generator generatorOutputDevice)
    )
eval model examples g = do
  let producer = P.zip (P.enumerate examples) (P.each [0 :: Int ..])
  x <- P.next producer
  case x of
    Left _ -> pure . Left $ g
    Right ((input, iter), producer') -> do
      let step ((loss, _), g') (input', iter') = liftIO $ do
            (loss', g'') <- forward model input' g'
            pure ((loss + loss', iter'), g'')
          init' = liftIO $ do
            (loss, g') <- forward model input g
            pure ((loss, iter), g')
          done ((loss, iter'), g'') = pure . Right $ (loss `divScalar` iter', g'')
      P.foldM step init' done producer'

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

data Monitor
  = TrainingMonitor {mtLoss :: Float, mtEpoch :: Int}
  | EvalMonitor {meLoss :: Float, meEpoch :: Int}
  deriving stock (Show)

collate ::
  SDevice device ->
  Int ->
  P.ListT IO (Float, Float) ->
  _
collate device batchSize =
  let collateFn chunk =
        let (xs, ys) = unzip chunk
            xs' = VS.singleton <$> xs
            ys' = VS.singleton <$> ys
            sToTensor' = sToTensor (SGradient SWithoutGradient) (SLayout SDense) device
         in (,) <$> sToTensor' xs' <*> sToTensor' ys'
   in P.lift . bufferedCollate (P.bounded 1) batchSize collateFn

-- | Run the two-layer network training loop on a toy dataset.
runTwoLayerNetworkExample :: IO ()
runTwoLayerNetworkExample = do
  let -- seed for the random number generator
      seed = 0

  let -- compute device
      device = SDevice SCPU

  let -- input dimension of the network
      inputDim = SName @"*" :&: SSize @1
      -- output dimension of the network
      outputDim = SName @"*" :&: SSize @1
      -- hidden dimension of the network
      hiddenDim = SName @"*" :&: SSize @100
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

  optim <- liftIO $ mkAdam defaultAdamOptions {learningRate = 1e-4} model

  let step (streamingState', g) epoch = do
        -- let learningRate = learningRateSchedule epoch
        -- ATen.setLearningRate optim learningRate

        -- train for one epoch on the training set
        (g', shuffle) <- do
          (trainingStream, shuffle) <- P.lift $ streamFromMap streamingState' trainingData
          trainingLoss <- do
            batchedStream <- collate' trainingStream
            P.lift . lift $ train optim trainingModelSpec batchedStream g
          case trainingLoss of
            Left g' -> pure (g', shuffle)
            Right (loss, g') -> do
              P.yield (TrainingMonitor (fromTensor loss) epoch)
              pure (g', shuffle)

        -- evaluate on the evaluation set
        (g'', shuffle') <- do
          (evalStream, shuffle') <- P.lift $ streamFromMap streamingState' {shuffle} evaluationData
          evalLoss <- do
            batchedStream <- collate' evalStream
            P.lift . lift $ do
              stateDict <- getStateDict optim
              evaluationModel <- flip evalStateT stateDict $ fromStateDict evaluationModelSpec mempty
              eval evaluationModel batchedStream g'
          case evalLoss of
            Left g'' -> pure (g'', shuffle')
            Right (loss, g'') -> do
              P.yield (EvalMonitor (fromTensor loss) epoch)
              pure (g'', shuffle')

        pure (streamingState' {shuffle = shuffle'}, g'')

  let init' = pure (streamingState, g1)

  let done = pure

  (streamingState', g2) <-
    flip runContT pure . P.runEffect $
      P.foldM step init' done (P.each [1 .. numEpochs])
        P.>-> P.print

  stateDict' <- getStateDict optim
  stateDictToFile stateDict' "twoLayerNetwork.pt"
