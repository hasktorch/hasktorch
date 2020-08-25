{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}

module TimeSeriesModel where

import Control.Monad (when)
import CovidData
import GHC.Generics
import Torch as T
import Torch.NN.Recurrent.Cell.LSTM

data OptimSpec o p where
  OptimSpec ::
    (Optimizer o, Parameterized p) =>
    { optimizer :: o,
      batchSize :: Int,
      numIters :: Int,
      learningRate :: Tensor,
      lossFn :: p -> Tensor -> Tensor -> Tensor -- model, input, target
    } ->
    OptimSpec o p

{- Trivial 1D baseline -}

data Simple1dSpec = Simple1dSpec
  { lstm1dSpec :: LSTMSpec,
    mlp1dSpec :: LinearSpec
  }
  deriving (Eq, Show)

data Simple1dModel = Simple1dModel
  { lstm1d :: LSTMCell,
    mlp1d :: Linear
  }
  deriving (Generic, Show, Parameterized)

instance Randomizable Simple1dSpec Simple1dModel where
  sample Simple1dSpec {..} =
    Simple1dModel
      <$> sample lstm1dSpec
      <*> sample mlp1dSpec

instance HasForward Simple1dModel [Tensor] ((Tensor, Tensor), Tensor) where
  forward model inputs = (lstmOutput, (forward (mlp1d model)) (fst lstmOutput))
    where
      cell = lstm1d model
      hSize = Prelude.div ((shape . toDependent . weightsIH $ cell) !! 0) 4
      iSize = (shape . toDependent . weightsIH $ cell) !! 1
      cellInit = zeros' [1, hSize]
      hiddenInit = zeros' [1, hSize]
      stateInit = (hiddenInit, cellInit)
      lstmOutput = foldl (lstmCellForward cell) stateInit inputs

{- Attempt to model cross-correlations -}

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

-- forward:: TSModel -> ModelInputs -> (Tensor, Tensor)

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

{- Computation Setups for Time Series -}

-- | Check forward computation
checkOutputs = do
  -- check time2vec
  t2v <- sample $ Time2VecSpec 10
  lstmLayer <- sample $ LSTMSpec (10 + 1) 2
  let result = t2vForward 3.0 t2v
  print result
  -- check end-to-end
  let inputDim = 3193 + 1 + 1 -- # counties + t2vDim + county of interest count
  model <-
    sample
      TSModelSpec
        { nCounties = 3193,
          countyEmbedDim = 6,
          t2vSpec = Time2VecSpec {t2vDim = 6},
          lstmSpec = LSTMSpec {inputSize = inputDim, hiddenSize = 12}
        }
  let result = tsmodelForward 10.0 model (ones' [inputDim], ones' [inputDim]) (ones' [3193]) 15.0
  print result

initModel nRegions t2vDim lstmHDim =
  sample
    TSModelSpec
      { nCounties = nRegions,
        countyEmbedDim = t2vDim,
        t2vSpec = Time2VecSpec {t2vDim = t2vDim},
        lstmSpec = LSTMSpec {inputSize = t2vDim + 1, hiddenSize = lstmHDim} -- t2vDim + this region's count (1D for now)
      }

optimSpec initializedModel lossFn =
  OptimSpec
    { optimizer = mkAdam 0 0.9 0.999 (flattenParameters initializedModel),
      batchSize = 128,
      numIters = 500,
      learningRate = 5e-5,
      lossFn = lossFn
    } ::
    OptimSpec Adam TSModel

{-
testModel = do
  let t2vd = 6
  let inputDim = 3193 + t2vd + 1 -- # counties + t2vDim + county of interest count
  initializedModel <- initModel 3193 t2vd 6
  let spec = optimSpec initializedModel undefined
  model <- train spec undefined initializedModel
  pure ()
-}
{- Computation Setups for 1D baseline -}

train ::
  (Dataset TensorData TensorData, Optimizer o, Parameterized p, HasForward p Tensor Tensor) =>
  OptimSpec o p ->
  TensorData ->
  p ->
  IO p
train OptimSpec {..} dataset init = do
  trained <- foldLoop init numIters $
    \state iter -> do
      -- let (input, dep) = getItem dataset (iter * batchSize) batchSize
      -- let timeSlices = getItem dataset iter batchSize
      -- let loss = lossFn state input dep
      let loss = undefined
      let flatParameters = flattenParameters state
      let (Gradients gradients) = grad' loss flatParameters
      -- when (iter `mod` 50 == 0) $ do
      -- putStrLn $ "Iteration: " ++ show iter ++ " | Loss: " ++ show loss
      (newParam, _) <- runStep state optimizer loss learningRate
      pure $ replaceParameters state newParam
  pure trained
