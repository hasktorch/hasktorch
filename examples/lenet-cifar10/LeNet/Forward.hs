{-# LANGUAGE CPP #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DataKinds #-}
module LeNet.Forward where

import Utils

import Numeric.Backprop
import Data.Maybe

#ifdef CUDA
import Torch.Cuda.Double (Tensor, HsReal, maxIndex1d)
import qualified Torch.Cuda.Long as Long
#else
import Torch.Double (Tensor, HsReal, maxIndex1d)
import qualified Torch.Long as Long
#endif

import Torch.Models.Vision.LeNet
import Torch.Data.Loaders.Cifar10 (Category)


infer
  :: LeNet 3 5
  -> Tensor '[3, 32, 32]
  -> Category
infer net

  -- cast from Integer to 'Torch.Data.Loaders.Cifar10.Category'
  = toEnum . fromIntegral

  -- Unbox the LongTensor '[1] to get 'Integer'
  . getindex

  -- argmax the output Tensor '[10] distriubtion. Returns LongTensor '[1]
  . maxIndex1d

#ifndef DEBUG
  -- take an input tensor and run 'lenet' with the model (undefined is the
  -- learning rate, which we can ignore)
  . evalBP2 (lenet undefined) net

#else
  . foo

 where
  foo x
    -- take an input tensor and run 'lenet' with the model (undefined is the
    -- learning rate, which we can ignore)
    = unsafePerformIO $ do
        let x' = evalBP2 (lenet undefined) net x
        pure x'
#endif



