-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Models.Internal
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
--
-- Helper functions which might end up migrating to the -indef codebase
-------------------------------------------------------------------------------
{-# LANGUAGE ScopedTypeVariables #-}
module Torch.Models.Internal where

import Data.Function ((&))
import GHC.Generics
import Prelude as P
import Data.Singletons.Prelude hiding (type (*), All)
import Numeric.Dimensions

#ifdef CUDA
import Torch.Cuda.Double as Torch
import Torch.Cuda.Double.NN.Linear (Linear(..))
import qualified Torch.Cuda.Double.NN.Conv2d as NN
#else
import Torch.Double as Torch
import Torch.Double.NN.Linear (Linear(..))
import qualified Torch.Double.NN.Conv2d as NN
#endif


-- Layer initialization: These depend on random functions which are not unified and, thus,
-- it's a little trickier to fold these back into their respective NN modules.

-- | initialize a new linear layer
newLinear :: forall o i . All KnownDim '[i,o] => IO (Linear i o)
newLinear = fmap Linear . newLayerWithBias $ dimVal (dim :: Dim i)

-- | initialize a new conv2d layer
newConv2d :: forall o i kH kW . All KnownDim '[i,o,kH,kW] => IO (Conv2d i o '(kH,kW))
newConv2d = fmap Conv2d . newLayerWithBias $
  dimVal (dim :: Dim i) * dimVal (dim :: Dim kH) * dimVal (dim :: Dim kW)


-- | uniform random initialization
newLayerWithBias :: All Dimensions '[d,d'] => Word -> IO (Tensor d, Tensor d')
newLayerWithBias n = do
#ifdef CUDA
  (,) <$> uniform (-stdv) stdv
      <*> uniform (-stdv) stdv
#else
  g <- newRNG
  let Just pair = ord2Tuple (-stdv, stdv)
  manualSeed g 10
  (,) <$> uniform g pair
      <*> uniform g pair
#endif
  where
    stdv :: Double
    stdv = 1 / P.sqrt (fromIntegral n)

