{-# LANGUAGE CPP #-}
{-# LANGUAGE FlexibleContexts #-}
module Criterion where

#ifdef CUDA
import Torch.Cuda.Double
#else
import Torch.Double
#endif

import Data.Function ((&))
import Numeric.Backprop

crossEntropy
  :: (Reifies s W, All KnownDim '[b, p])
  => IndexTensor '[b]            -- THIndexTensor *target,
  -> BVar s (Tensor '[b, p])     -- THTensor *input,
  -> BVar s (Tensor '[1])        -- THTensor *output,
crossEntropy ys inp
  = logSoftMaxN (dim :: Dim 1) inp
  & classNLLCriterion ys


