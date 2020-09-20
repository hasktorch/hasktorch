{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE RecordWildCards #-}

module TimeSeriesModel where

import Prelude as P
import Control.Monad (when)
import CovidData
import GHC.Generics
import Torch as T
import Torch.NN.Recurrent.Cell.LSTM as L
import Torch.NN.Recurrent.Cell.GRU as G
import Text.Printf

data OptimSpec o p where
  OptimSpec ::
    (Optimizer o, Parameterized p) =>
    { optimizer :: o,
      batchSize :: Int,
      numIters :: Int,
      learningRate :: Tensor,
      lossFn :: Tensor -> Tensor -> Tensor -- model, input, target
    } ->
    OptimSpec o p

data MLPSpec = MLPSpec {
    inputFeatures :: Int,
    hiddenFeatures0 :: Int,
    hiddenFeatures1 :: Int,
    outputFeatures :: Int
    } deriving (Show, Eq)

data MLP = MLP { 
    l0 :: Linear,
    l1 :: Linear,
    l2 :: Linear
    } deriving (Generic, Show)

instance Parameterized MLP
instance Randomizable MLPSpec MLP where
    sample MLPSpec {..} = MLP 
        <$> sample (LinearSpec inputFeatures hiddenFeatures0)
        <*> sample (LinearSpec hiddenFeatures0 hiddenFeatures1)
        <*> sample (LinearSpec hiddenFeatures1 outputFeatures)

mlp :: MLP -> Tensor -> Tensor
mlp MLP{..} input = 
    linear l2
    . relu
    . linear l1
    . relu
    . linear l0
    $ input

instance HasForward MLP Tensor Tensor where
  forward = mlp
  forwardStoch model x = pure $ mlp model x -- TODO - have make this the default defn?

{- Trivial 1D baseline -}

data Simple1dSpec = Simple1dSpec
  { 
    encoderSpec :: MLPSpec,
    gru1dSpec :: GRUSpec,
    decoderSpec :: MLPSpec
  }
  deriving (Eq, Show)

data Simple1dModel = Simple1dModel
  { 
    encoder :: MLP,
    gru1d :: GRUCell,
    decoder :: MLP
  }
  deriving (Generic, Show, Parameterized)

instance Randomizable Simple1dSpec Simple1dModel where
  sample Simple1dSpec {..} =
    Simple1dModel
      <$> sample encoderSpec
      <*> sample gru1dSpec
      <*> sample decoderSpec

swish x = T.mul x (sigmoid x)

instance HasForward Simple1dModel [Tensor] Tensor where
  forward Simple1dModel {..} inputs = 
    swish $ forward decoder (forward encoder $ seqOutput)
    where
      cell = gru1d
      hSize = P.div ((shape . toDependent . G.weightsIH $ cell) !! 0) 3 -- 3 for GRU
      hiddenInit = zeros' [1, hSize ] -- what should this be for GRU
      seqOutput = foldl (gruCellForward cell . forward encoder) hiddenInit inputs
  forwardStoch m x = pure (forward m x)

{- Attempt to model cross-correlations -}

data Time2VecSpec = Time2VecSpec
  { t2vDim :: Int -- note output dimensions is +1 of this value due to non-periodic term
  }
  deriving (Eq, Show)

data Time2Vec = Time2Vec
  { w0 :: Parameter, -- 0 dim
    b0 :: Parameter, -- 0 dim
    w :: Parameter,
    b :: Parameter
  }
  deriving (Generic, Show, Parameterized)

instance Randomizable Time2VecSpec Time2Vec where
  sample Time2VecSpec {..} = do
    w0' <- makeIndependent =<< randIO' [1]
    b0' <- makeIndependent =<< randIO' [1]
    w' <- makeIndependent =<< randIO' [t2vDim]
    b' <- makeIndependent =<< randIO' [t2vDim]
    pure $
      Time2Vec
        { w0 = w0',
          b0 = b0',
          w = w',
          b = b'
        }

t2vForward :: Float -> Time2Vec -> Tensor
t2vForward t Time2Vec {..} =
  T.cat
    (Dim 0)
    [ mulScalar t w0' + b0',
      T.sin $ mulScalar t w' + b'
    ]
  where
    (w0', b0', w', b') =
      ( toDependent w0,
        toDependent b0,
        toDependent w,
        toDependent b
      )

data TSModelSpec = TSModelSpec
  { nCounties :: Int,
    countyEmbedDim :: Int,
    t2vSpec :: Time2VecSpec,
    lstmSpec :: LSTMSpec
  }
  deriving (Eq, Show)

data TSModel = TSModel
  { countyEmbed :: Linear,
    t2v :: Time2Vec,
    lstm :: LSTMCell
  }
  deriving (Generic, Show, Parameterized)

instance Randomizable TSModelSpec TSModel where
  sample TSModelSpec {..} =
    TSModel
      <$> sample (LinearSpec nCounties countyEmbedDim)
      <*> sample t2vSpec
      <*> sample lstmSpec


data ModelInputs = ModelInputs
  { time :: Float,
    tensorData :: TensorData,
    lstmState :: (Tensor, Tensor)
  }
  deriving (Eq, Show)

tsmodelForward ::
  Float -> -- time
  TSModel -> -- model state
  (Tensor, Tensor) -> -- lstm (hidden state, cell state)
  Tensor -> -- all counties context
  Tensor -> -- input
  (Tensor, Tensor) -- output
tsmodelForward t TSModel {..} (hiddenState, cellState) allCounties countyCount =
  lstmCellForward
    lstm
    (hiddenState, cellState)
    (T.cat (Dim 0) [linearForward countyEmbed allCounties, t2vForward t t2v, countyCount])

instance HasForward TSModel Tensor Tensor where
  forward model modelInputs = undefined
  forwardStoch model modelInputs = undefined

{-
clipGradient :: T.Scalar a => a -> Gradients -> Gradients
clipGradient maxScale (Gradients gradients) =  
  if scale > maxScale then
    Gradients zipWith (mulScalar (scale / maxScale) <$> gradients)
  else
    Gradients gradients
  where
    scale = (asValue . T.sumAll . T.abs <$> gradients)
-}

train ::
  (Optimizer o, Parameterized p, HasForward p [Tensor] Tensor) =>
  OptimSpec o Simple1dModel ->
  TimeSeriesData ->
  p ->
  IO p
train OptimSpec {..} dataset init = do
  (trained, _) <- foldLoop (init, optimizer) numIters $
    \(state, optimizer) iter -> do
      obs <- randintIO' 0 190 []
      let startTime = 0 :: Int
          obs' = asValue obs :: Float
          time = round obs'
          (past, future) = getItem dataset time 1
          output = forward state (getObs' 0 past)
          actual = (getTime' 0 0 future)
          loss = T.sqrt $ lossFn actual output -- get absolute value of error
          flatParameters = flattenParameters state
          (Gradients gradients) = grad' loss flatParameters
      when (iter `mod` 10 == 0) $ do
        let  
            output' = asValue output :: Float
            actual' = P.round (asValue actual :: Float) :: Int
            loss' = asValue loss :: Float
        putStrLn $ printf "it %6d | seqlen (t) %4d | pred %6.1f | actual %4d | error %5.1f" iter time output' actual' loss'
      (newModel, optimizer) <- runStep state optimizer loss learningRate
      pure (newModel, optimizer)
      -- (newParam, _) <- runStep state optimizer loss learningRate
      -- pure $ replaceParameters state newParam
  pure trained
