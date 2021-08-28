{-# LANGUAGE BlockArguments #-}

module Main where

{- Optimizer implementations run on standard test functions -}

import Control.Monad (foldM, when)
import GHC.Generics
import TestFunctions
import Text.Printf (printf)
import Torch
import Prelude hiding (sqrt)

-- | show output after n iterations
showLog :: (Show a) => Int -> Int -> Int -> Tensor -> a -> IO ()
showLog n i maxIter lossValue state =
  when (i == 0 || mod i n == 0 || i == maxIter -1) do
    putStrLn
      ( "Iter: " ++ printf "%6d" i
          ++ " | Loss:"
          ++ printf "%05.4f" (asValue lossValue :: Float)
          ++ " | Parameters: "
          ++ show state
      )

-- | Optimize convex quadratic with specified optimizer
optConvQuad :: (Optimizer o) => Int -> o -> LearningRate -> IO ()
optConvQuad numIter optInit lr = do
  let dim = 2
      a = eye' dim dim
      b = zeros' [dim]
  paramInit <- sample $ ConvQuadSpec dim
  putStrLn ("Initial :" ++ show paramInit)
  trained <- foldLoop (paramInit, optInit) numIter $ \(paramState, optState) i -> do
    let lossValue = (lossConvQuad a b) paramState
    showLog 1000 i numIter lossValue paramState
    runStep paramState optState lossValue lr
  pure ()

-- | Optimize Rosenbrock function with specified optimizer
optRosen :: (Optimizer o) => Int -> o -> LearningRate -> IO ()
optRosen numIter optInit lr = do
  paramInit <- sample RosenSpec
  putStrLn ("Initial :" ++ show paramInit)
  trained <- foldLoop (paramInit, optInit) numIter $ \(paramState, optState) i -> do
    let lossValue = lossRosen paramState
    showLog 1000 i numIter lossValue paramState
    runStep paramState optState lossValue lr
  pure ()

-- | Optimize Ackley function with specified optimizer
optAckley :: (Optimizer o) => Int -> o -> LearningRate -> IO ()
optAckley numIter optInit lr = do
  paramInit <- sample AckleySpec
  putStrLn ("Initial :" ++ show paramInit)
  trained <- foldLoop (paramInit, optInit) numIter $ \(paramState, optState) i -> do
    let lossValue = lossAckley paramState
    showLog 1000 i numIter lossValue paramState
    runStep paramState optState lossValue lr
  pure ()

-- | Check global minimum point for Rosenbrock
checkGlobalMinRosen :: IO ()
checkGlobalMinRosen = do
  putStrLn "\nCheck Actual Global Minimum (at 1, 1):"
  print $ rosenbrock' (asTensor (1.0 :: Float)) (asTensor (1.0 :: Float))

-- | Check global minimum point for Convex Quadratic
checkGlobalMinConvQuad :: IO ()
checkGlobalMinConvQuad = do
  putStrLn "\nCheck Actual Global Minimum (at 0, 0):"
  let dim = 2
      a = eye' dim dim
      b = zeros' [dim]
  print $ convexQuadratic a b (zeros' [dim])

-- | Check global minimum point for Ackley
checkGlobalMinAckley :: IO ()
checkGlobalMinAckley = do
  putStrLn "\nCheck Actual Global Minimum (at 0, 0):"
  print $ ackley' (zeros' [2])

main :: IO ()
main = do
  let numIter = 20000

  -- Convex Quadratic w/ GD, GD+Momentum, Adam, Adagrad
  putStrLn "\nConvex Quadratic\n================"
  putStrLn "\nGD"
  optConvQuad numIter GD 5e-4
  putStrLn "\nGD + Momentum"
  optConvQuad numIter (GDM 0.9 [zeros' [2]]) 5e-4
  putStrLn "\nAdam"
  optConvQuad
    numIter
    Adam
      { beta1 = 0.9,
        beta2 = 0.999,
        m1 = [zeros' [1], zeros' [1]],
        m2 = [zeros' [1], zeros' [1]],
        iter = 0
      }
    5e-4
  putStrLn "\nAdagrad"
  optConvQuad numIter Adagrad {gsum = [zeros' [1]]} 5e-2
  checkGlobalMinConvQuad

  -- 2D Rosenbrock w/ GD, GD+Momentum, Adam, Adagrad
  putStrLn "\n2D Rosenbrock\n================"
  putStrLn "\nGD"
  optRosen numIter GD 5e-4
  putStrLn "\nGD + Momentum"
  optRosen numIter (GDM 0.9 [zeros' [1], zeros' [1]]) 5e-4
  putStrLn "\nAdam"
  optRosen
    numIter
    Adam
      { beta1 = 0.9,
        beta2 = 0.999,
        m1 = [zeros' [1], zeros' [1]],
        m2 = [zeros' [1], zeros' [1]],
        iter = 0
      }
    5e-4
  putStrLn "\nAdagrad"
  optRosen numIter Adagrad {gsum = [zeros' [1], zeros' [1]]} 5e-1
  checkGlobalMinRosen

  -- Ackley w/ GD, GD+Momentum, Adam, Adagrad
  putStrLn "\nAckley (Gradient methods fail)\n================"
  putStrLn "\nGD"
  optAckley numIter GD 5e-4
  putStrLn "\nGD + Momentum"
  optAckley numIter (GDM 0.9 [zeros' [1], zeros' [1]]) 5e-4
  putStrLn "\nAdam"
  optAckley
    numIter
    Adam
      { beta1 = 0.9,
        beta2 = 0.999,
        m1 = [zeros' [1], zeros' [1]],
        m2 = [zeros' [1], zeros' [1]],
        iter = 0
      }
    5e-4
  putStrLn "\nAdagrad"
  optAckley numIter Adagrad {gsum = [zeros' [1]]} 5e-2
  checkGlobalMinAckley
