{-# LANGUAGE ScopedTypeVariables #-}
module Torch.Data.Metrics where

import Data.List (genericLength)
import Data.Function (on)


#ifdef CUDA
import Torch.Cuda.Double
import qualified Torch.Cuda.Long as Long
#else
import Torch.Double
import qualified Torch.Long as Long
#endif


catAccuracy
  :: forall c sz
  . (Eq c, Enum c) -- , sz ~ FromEnum (MaxBound c), KnownDim sz, KnownNat sz)
  => [(Int, c)] --  [(Tensor '[FromEnum (MaxBound c)], c)]
  -> Double
catAccuracy xs = filter issame xs // xs
  where
    (//) = (/) `on` genericLength
    issame (p, y) = toEnum p == y


