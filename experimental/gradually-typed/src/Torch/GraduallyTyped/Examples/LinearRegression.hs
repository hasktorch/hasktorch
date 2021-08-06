{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE TypeApplications #-}
{-# OPTIONS_GHC -Wno-missing-signatures #-}

module Torch.GraduallyTyped.Examples.LinearRegression where

import Control.Monad (foldM, when)
import Control.Monad.Indexed ((>>>=))
import Control.Monad.Indexed.State (IxStateT (..))
import Control.Monad.Indexed.Trans (IxMonadTrans (ilift))
import Data.Functor.Indexed ((<<$>>))
import Torch.GraduallyTyped

model state input =
  runIxStateT $
    squeezeAll <<$>> IxStateT (forward state input)

groundTruth t = do
  weight <-
    sToTensor gradient layout device [42.0 :: Float, 64.0, 96.0]
      >>= sCheckedShape (SShape $ SNoName :&: SSize @3 :|: SNil)
  let dataType = sGetDataType weight
  bias <- sFull (spec dataType) (3.14 :: Float)
  prod <- sCheckedDataType dataType t >>= (`matmul` weight)
  pure . squeezeAll $ prod `add` bias
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

  let spec = linearSpec hasBias gradient device dataType inputDim outputDim
  (initial, g1) <- initialize spec g0

  printParams initial

  optim <- optimizer initial

  _ <- foldLoop g1 numIters $ \g i -> do
    (input, g') <- sRandn tensorSpec g
    y <- groundTruth input
    (loss, g'') <-
      stepWithGenerator
        optim
        spec
        ( \state ->
            runIxStateT $
              IxStateT (model state input)
                >>>= ilift . (\y' -> y `mseLoss` y')
        )
        g'
    when (i `mod` 100 == 0) $ do
      putStrLn $ "Iteration: " ++ show i ++ " | Loss: " ++ show loss
    pure g''

  trained <- getModel spec optim

  printParams trained
  where
    optimizer = mkAdam defaultAdamOptions {learningRate = 1e0}
    defaultRNG = sMkGenerator device 31415
    numIters = 2000
    batchDim = SNoName :&: SSize @4
    inputDim = SNoName :&: SSize @3
    outputDim = SNoName :&: SSize @1

    foldLoop x count block = foldM block x [1 :: Int .. count]

    hasBias = SWithBias
    gradient = SGradient SWithGradient
    device = SDevice SCPU
    dataType = SDataType SFloat
    tensorSpec = TensorSpec gradient (SLayout SDense) device dataType (SShape $ batchDim :|: inputDim :|: SNil)
