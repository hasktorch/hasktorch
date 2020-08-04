{-# LANGUAGE RecordWildCards #-}

module TimeSeriesModel where

import Torch as T

data Time2Vec  = Time2Vec {
  w0 :: Tensor, -- 0 dim
  b0 :: Tensor, -- 0 dim
  w :: Tensor, -- 1 dim
  b :: Tensor -- 1 dim
}

-- t2v :: Float -> (Tensor, Tensor)
-- t2v t Time2Vec{..} = (mulScalar w0 t + b, T.sin (mulScalar w t _b))

model = undefined
