{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}

module TimeSeriesModel where

import Control.Monad (when)
import GHC.Generics
import Torch as T
import Torch.NN.Recurrent.Cell.LSTM

import CovidData

data OptimSpec o p where
    OptimSpec :: (Optimizer o, Parameterized p) => {
    optimizer :: o,
    batchSize :: Int,
    numIters :: Int,
    learningRate :: Tensor,
    lossFn :: p -> Tensor -> Tensor -> Tensor -- model, input, target
    } -> OptimSpec o p

data TSModelSpec = TSModelSpec {
  nCounties :: Int,
  countyEmbedDim :: Int,
  t2vSpec :: Time2VecSpec,
  lstmSpec :: LSTMSpec
} deriving (Eq, Show)

data TSModel = TSModel {
  countyEmbed :: Linear,
  t2v :: Time2Vec,
  lstm :: LSTMCell
} deriving (Generic, Show, Parameterized)

instance Randomizable TSModelSpec TSModel where
    sample TSModelSpec {..} = TSModel
        <$> sample (LinearSpec nCounties countyEmbedDim)
        <*> sample t2vSpec
        <*> sample lstmSpec

data ModelInputs = ModelInputs {
    time :: Float,
    lstmState :: (Tensor, Tensor),
    allCounties :: Tensor,
    input :: Tensor
} deriving (Eq, Show)

tsmodelForward 
  :: Float 
  -> TSModel -- ^ model state
  -> (Tensor, Tensor) -- ^ (hidden state, cell state)
  -> Tensor  -- ^ all counties context
  -> Tensor -- ^ input
  -> (Tensor, Tensor)
tsmodelForward t TSModel{..} (hiddenState, cellState) allCounties countyCount = 
  lstmCellForward lstm
    (hiddenState, cellState)    
    (T.cat (Dim 0) [linearForward countyEmbed allCounties, t2vForward t t2v, countyCount])

instance HasForward TSModel ModelInputs (Tensor, Tensor) where
    -- forward:: TSModel -> ModelInputs -> (Tensor, Tensor)
    forward model modelInputs = undefined

data Time2VecSpec = Time2VecSpec {
  t2vDim :: Int -- note output dimensions is +1 of this value due to non-periodic term
} deriving (Eq, Show)

data Time2Vec  = Time2Vec {
  w0 :: Parameter, -- 0 dim
  b0 :: Parameter, -- 0 dim
  w :: Parameter,
  b :: Parameter
} deriving (Generic, Show, Parameterized)

instance Randomizable Time2VecSpec Time2Vec where
  sample Time2VecSpec{..} = do
    w0' <- makeIndependent =<< randIO' [1]
    b0' <- makeIndependent =<< randIO' [1]
    w' <- makeIndependent =<< randIO' [t2vDim]
    b' <- makeIndependent =<< randIO' [t2vDim]
    pure $ Time2Vec {
        w0=w0',
        b0=b0',
        w=w',
        b=b' }

t2vForward :: Float -> Time2Vec -> Tensor
t2vForward t Time2Vec{..} =
  T.cat (Dim 0) [mulScalar t w0' + b0',
                 T.sin $ mulScalar t w' + b']
  where (w0', b0', w', b') = (toDependent w0, toDependent b0, 
                              toDependent w, toDependent b)


train 
    :: (Dataset d (Tensor, Tensor), Optimizer o, Parameterized p, HasForward p Tensor Tensor) 
    => OptimSpec o p -> d -> p -> IO p
train OptimSpec{..} dataset init = do
    trained <- foldLoop init numIters $
        \state iter -> do
            let (input, dep) = getItem dataset (iter*batchSize) batchSize
            let loss = lossFn state input dep
            let flatParameters = flattenParameters state
            let (Gradients gradients) = grad' loss flatParameters

            when (iter `mod` 50 == 0) $ do
                putStrLn $ "Iteration: " ++ show iter ++ " | Loss: " ++ show loss
            (newParam, _) <- runStep state optimizer loss learningRate
            pure $ replaceParameters state newParam
    pure trained

checkOutputs = do
  -- check time2vec
  t2v <- sample $ Time2VecSpec 10
  lstmLayer <- sample $ LSTMSpec (10 + 1) 2
  let result = t2vForward 3.0 t2v
  print result

  -- check end-to-end
  let inputDim = 3193 + 1 + 1 -- # counties + t2vDim + county of interest count
  model <- sample TSModelSpec {
    nCounties = 3193,
    countyEmbedDim = 6,
    t2vSpec = Time2VecSpec { t2vDim=6 },
    lstmSpec = LSTMSpec { inputSize=inputDim, hiddenSize=12 } 
    }
  let result = tsmodelForward 10.0 model (ones' [inputDim], ones' [inputDim]) (ones' [3193]) 15.0
  print result

testModel = do
  let inputDim = 3193 + 1 + 1 -- # counties + t2vDim + county of interest count
  initializedModel <- sample TSModelSpec {
    nCounties = 3193,
    countyEmbedDim = 6,
    t2vSpec = Time2VecSpec { t2vDim=6 },
    lstmSpec = LSTMSpec { inputSize=3195, hiddenSize=12 }  -- 3195=3193 + t2vDim + count
    }
  let optimSpec = OptimSpec {
        optimizer = mkAdam 0 0.9 0.999 (flattenParameters initializedModel),
        batchSize = 128,
        numIters = 500,
        learningRate = 5e-5, 
        lossFn = undefined -- \model input target -> 
            -- let (h, c) = forward model input 
            -- in mseLoss target c
    } :: OptimSpec Adam TSModel

  -- model <- train optimSpec undefined initializedModel
  pure ()
