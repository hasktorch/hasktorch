{-# LANGUAGE ScopedTypeVariables #-}

module Torch.GraduallyTyped.LearningRateSchedules where

-- | Single-cycle learning rate schedule.
-- See, for instance, https://arxiv.org/abs/1803.09820.
--
-- This is a simple schedule that is a stepwise linear interpolation
-- between the initial, maximum, and final learning rates.
-- The initial learning rate is zero.
singleCycleLearningRateSchedule ::
  -- | peak learning rate after warmup
  Double ->
  -- | learning rate at the end of the schedule
  Double ->
  -- | total number of epochs
  Int ->
  -- | number of warm-up epochs
  Int ->
  -- | number of cool-down epochs
  Int ->
  -- | current epoch
  Int ->
  -- | current learning rate
  Double
singleCycleLearningRateSchedule maxLearningRate finalLearningRate numEpochs numWarmupEpochs numCooldownEpochs epoch
  | epoch <= 0 = 0.0
  | 0 < epoch && epoch <= numWarmupEpochs =
    let a :: Double = fromIntegral epoch / fromIntegral numWarmupEpochs
     in a * maxLearningRate
  | numWarmupEpochs < epoch && epoch < numEpochs - numCooldownEpochs =
    let a :: Double =
          fromIntegral (numEpochs - numCooldownEpochs - epoch)
            / fromIntegral (numEpochs - numCooldownEpochs - numWarmupEpochs)
     in a * maxLearningRate + (1 - a) * finalLearningRate
  | otherwise = finalLearningRate
