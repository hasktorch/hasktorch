{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE TypeApplications #-}
{-# OPTIONS_GHC -Wno-missing-signatures #-}

module Main where

import Control.Monad (foldM_, when)
import Control.Monad.Indexed ((>>>=))
import Control.Monad.Indexed.State (IxStateT (..))
import Control.Monad.Indexed.Trans (IxMonadTrans (ilift))
import Data.Functor.Indexed ((<<$>>))
import Torch.GraduallyTyped

model state input = runIxStateT $ squeezeAll <<$>> IxStateT (forward state input)

groundTruth t = do
  weight <- sToTensor gradient layout device [42.0 :: Float, 64.0, 96.0] -- >>= sCheckedShape (SShape $ SNoName :&: SSize @3 :|: SNil) -- <<< uncomment to check shape
  let dataType = sGetDataType weight
  bias <- sFull (spec dataType) (3.14 :: Float)
  prod <- sCheckedDataType dataType t >>= (`matmul` weight)
  squeezeAll <$> prod `add` bias
  where
    gradient = SGradient SWithoutGradient
    layout = sGetLayout t
    device = sGetDevice t
    spec dataType = TensorSpec gradient layout device dataType (SShape SNil)

printParams GLinear {..} = do
  putStrLn $ "Parameters:\n" ++ show linearWeight
  putStrLn $ "Bias:\n" ++ show linearBias

main :: IO ()
main = do
  g0 <- defaultRNG
  let modelSpec = linearSpec SWithBias (SGradient SWithGradient) device dataType inputDim outputDim
  (initial, g1) <- initialize modelSpec g0
  printParams initial
  optim <- mkOptim initial

  foldM_
    ( \g i -> do
        (input, g') <- sRandn randSpec g
        target <- groundTruth input
        (loss, g'') <-
          stepWithGenerator
            optim
            modelSpec
            (\state -> runIxStateT $ IxStateT (model state input) >>>= ilift . (target `mseLoss`))
            g'
        when (i `mod` 100 == 0) $
          putStrLn $ "Iteration: " ++ show i ++ " | Loss: " ++ show loss
        pure g''
    )
    g1
    [1 :: Int .. numIters]

  trained <- getModel modelSpec optim
  printParams trained
  where
    mkOptim = mkAdam defaultAdamOptions {learningRate = 1e0}
    defaultRNG = sMkGenerator device 31415
    numIters = 2000
    batchDim = SNoName :&: SSize @4
    inputDim = SNoName :&: SSize @3
    outputDim = SNoName :&: SSize @1
    device = SDevice SCPU
    dataType = SDataType SFloat
    randSpec = TensorSpec (SGradient SWithoutGradient) (SLayout SDense) device dataType (SShape $ batchDim :|: inputDim :|: SNil)
