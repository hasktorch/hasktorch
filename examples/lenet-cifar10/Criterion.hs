{-# LANGUAGE CPP #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
module Criterion where

#ifdef CUDA
import Torch.Cuda.Double
import qualified Torch.Cuda.Long as Long
import qualified Torch.Cuda.Double.Dynamic.NN.Criterion as Dynamic
import qualified Torch.Cuda.Double.Dynamic as Dynamic
#else
import Torch.Double
import qualified Torch.Long as Long
import qualified Torch.Double.Dynamic.NN.Criterion as Dynamic
import qualified Torch.Double.Dynamic as Dynamic
#endif

import System.IO.Unsafe
import Data.Maybe (fromMaybe)
import Control.Monad ((>=>), (<=<))
import Data.Function ((&))
import Numeric.Backprop

crossEntropy
  :: (Reifies s W, All KnownDim '[b, p])
  => IndexTensor '[b]            -- THIndexTensor *target,
  -> BVar s (Tensor '[b, p])     -- THTensor *input,
  -> BVar s (Tensor '[1])        -- THTensor *output,
crossEntropy ys = liftOp1 . op1 $ \inp -> unsafePerformIO $ do
  (out, getgrad) <- crossEntropyIO ys inp
  pure (out, unsafePerformIO . getgrad)

crossEntropyIO
  :: (All KnownDim '[b, p])
  => IndexTensor '[b]            -- THIndexTensor *target,
  -> Tensor '[b, p]              -- THTensor *input,
  -> IO (Tensor '[1],
         Tensor '[1] -> IO (Tensor '[b, p]))
crossEntropyIO ys inp = do
  (lout, getLSMGrad) <- logSoftMaxBatchIO inp
  (nllout, getNLLGrad) <- classNLLIO ys lout
  pure (nllout, getNLLGrad >=> getLSMGrad)
  where
    verbose = True

logSoftMaxBatchIO
  :: forall b ps
  .  (All KnownDim '[b, ps])
  => Tensor '[b, ps]    -- ^ input
  -> IO (Tensor '[b, ps], Tensor '[b, ps] -> IO (Tensor '[b, ps]))   -- ^ output and gradient
logSoftMaxBatchIO = _logSoftMaxBatchIO (Just new) (Just new)

-- | run a threshold function againts two BVar variables
_logSoftMaxBatchIO
  :: forall b ps
  .  (All KnownDim '[b, ps])
  -- cachables
  => Maybe (Tensor '[b, ps])
  -> Maybe (Tensor '[b, ps])
  -> Tensor '[b, ps]    -- ^ input
  -> IO (Tensor '[b, ps], Tensor '[b, ps] -> IO (Tensor '[b, ps]))   -- ^ output and gradient
_logSoftMaxBatchIO mout mgin inp = do
  let out = fromMaybe new mout
  zero_ out
  updateOutput_ inp i out
  pure (out, \gout -> do
    let gin = fromMaybe new mgin
    zero_ gin
    updateGradInput_ inp gout out i gin
    pure gin)

 where
  i :: Dim 1
  i = dim

  updateOutput_ :: Tensor '[b, ps] -> Dim 1 -> Tensor '[b, ps] -> IO ()
  updateOutput_ inp i out =
    Dynamic._logSoftMax_updateOutput (asDynamic inp) (asDynamic out) (fromIntegral $ dimVal i)

  updateGradInput_
    :: Tensor '[b, ps]  -- input
    -> Tensor '[b, ps]  -- gradOutput
    -> Tensor '[b, ps]  -- output
    -> Dim 1            -- dimension

    -> Tensor '[b, ps]  -- gradInput
    -> IO ()
  updateGradInput_ inp gout out i gin =
    Dynamic._logSoftMax_updateGradInput
      (asDynamic inp)             -- input
      (asDynamic gout)            -- gradOutput
      (asDynamic gin)             -- gradInput
      (asDynamic out)             -- output
      (fromIntegral $ dimVal i)   -- dimension


criterion
  :: (ys -> out -> IO (loss, loss -> IO out))              -- ^ loss function
  -> (lr -> arch -> xs -> IO (out, out -> IO (arch, xs)))  -- ^ forward function with a learning rate
  -> lr
  -> arch
  -> ys
  -> xs
  -> IO (loss, arch)
criterion lossfn forward lr net ys xs = do
  (out, getArchGrad) <- forward lr net xs
  (loss, getLossGrad) <- lossfn ys out
  gnet <- fmap fst . (getArchGrad <=< getLossGrad) $ loss
  pure (loss, gnet)


