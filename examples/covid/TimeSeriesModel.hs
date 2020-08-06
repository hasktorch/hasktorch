{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE MultiParamTypeClasses #-}

module TimeSeriesModel where

import GHC.Generics
import Torch as T
import Torch.NN.Recurrent.Cell.LSTM

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
} deriving (Generic, Show)

instance Parameterized TSModel
instance Randomizable TSModelSpec TSModel where
    sample TSModelSpec {..} = TSModel
        <$> sample (LinearSpec nCounties countyEmbedDim)
        <*> sample t2vSpec
        <*> sample lstmSpec

data Time2VecSpec = Time2VecSpec {
  t2vDim :: Int -- note output dimensions is +1 of this value due to non-periodic term
} deriving (Eq, Show)

tsmodelForward :: Float -> TSModel -> Tensor -> Tensor -> Tensor
tsmodelForward t TSModel{..} allCounties = 
  lstmCellForward lstm
    (T.cat [linearForward countyEmbed allCounties, t2vForward t t2v])
    (undefined, undefined) -- hidden, cell
  where
    t2v' = toDependent t2v




data Time2Vec  = Time2Vec {
  w0 :: Parameter, -- 0 dim
  b0 :: Parameter, -- 0 dim
  w :: Parameter,
  b :: Parameter
} deriving (Generic, Show)

instance Parameterized Time2Vec
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

checkModel = do
  t2v <- sample $ Time2VecSpec 10
  lstmLayer <- sample $ LSTMSpec (10 + 1) 2
  let result = t2vForward t2v 3.0
  print result
