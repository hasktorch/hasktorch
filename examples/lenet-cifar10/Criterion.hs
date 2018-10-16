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
import Control.Monad ((>=>))
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



classNLLIO
  :: forall sz ps
  . (All KnownDim '[sz, ps])
  => IndexTensor '[sz]
  -> Tensor '[sz, ps] -> IO (Tensor '[1], Tensor '[1] -> IO (Tensor '[sz, ps]))
classNLLIO = _classNLLIO (Just new) (Just new) (Just new)

_classNLLIO
  :: forall sz ps
  . (All KnownDim '[sz, ps])
  -- optional cacheables
  => Maybe (Tensor '[1])
  -> Maybe (Tensor '[1])
  -> Maybe (Tensor '[sz, ps])

  -> IndexTensor '[sz]

  -> Tensor '[sz, ps]                                           --  \___ these constitue a closed cartesian category and
      -> IO (Tensor '[1], Tensor '[1] -> IO (Tensor '[sz, ps])) --  /    can be abstracted away into an autodiff lib.
_classNLLIO moutbuf mwbuf mginbuf target inp = do
  let out = fromMaybe new moutbuf
  zero_ out

  let total_weight = fromMaybe new mwbuf
  -- let total_weight = constant 1  -- https://github.com/torch/nn/commit/3585e827eb65d071272a4aa4fab567b0b1eeee54#diff-1aa6a505cf16ad0e59498ada8432afb5
  onesLike_ total_weight total_weight

  updateOutput_ inp target szAvg Nothing ix reduce (out, total_weight)

  pure (out, \gout -> do
    let gin = fromMaybe new mginbuf :: Tensor '[sz, ps]
    zero_ gin

    updateGradInput_ inp target gout szAvg Nothing total_weight ix reduce gin
    pure gin)
  where
    ix = (-100)
    reduce = True
    szAvg = True

    updateOutput_
      :: Tensor '[sz, ps]            -- THTensor *input,
      -> IndexTensor '[sz]           -- THIndexTensor *target,
      -> Bool                        -- bool sizeAverage,
      -> Maybe (Tensor '[sz, ps])    -- THTensor *weights,
      -> Integer                     -- int64_t ignore_index,
      -> Bool                        -- bool reduce
      -> (Tensor '[1], Tensor '[1])  -- THTensor *input, total_weight
      -> IO ()
    updateOutput_ inp tar szAvg mws ix reduce (out, total_weight) = do
      Dynamic._ClassNLLCriterion_updateOutput (asDynamic inp) (Long.longAsDynamic tar) (asDynamic out)
        szAvg (asDynamic <$> mws) (asDynamic total_weight) ix reduce

    updateGradInput_
      :: Tensor '[sz, ps]          -- THTensor *input,
      -> IndexTensor '[sz]         -- THIndexTensor *target,
      -> Tensor '[1]               -- THTensor *gradOutput,
      -> Bool                      -- bool sizeAverage,
      -> Maybe (Tensor '[sz, ps])  -- THTensor *weights,
      -> Tensor '[1]               -- THTensor *total_weight,
      -> Integer                   -- int64_t ignore_index,
      -> Bool                      -- bool reduce

      -> Tensor '[sz, ps]          -- gradient to update inplace
      -> IO ()
    updateGradInput_ inp tar gout szAvg mws total_weight ix reduce gin =
      Dynamic._ClassNLLCriterion_updateGradInput (asDynamic inp) (Long.longAsDynamic tar) (asDynamic gout) (asDynamic gin)
        szAvg (asDynamic <$> mws) (asDynamic total_weight) ix reduce


