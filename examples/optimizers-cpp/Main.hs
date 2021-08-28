{-# LANGUAGE BlockArguments #-}

module Main where

{- Optimizer implementations run on standard test functions -}

import Control.Monad (foldM, forM_, when)
import Data.Default.Class
import GHC.Generics
import TestFunctions
import Text.Printf (printf)
import Torch
import Torch.Optim.CppOptim
import Prelude hiding (sqrt)

-- import Data.IORef

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
optConvQuad :: (CppOptimizer opt) => Int -> opt -> IO ()
optConvQuad numIter optInit = do
  let dim = 2
      a = eye' dim dim
      b = zeros' [dim]
  paramInit <- sample $ ConvQuadSpec dim
  putStrLn ("Initial :" ++ show paramInit)
  optimizer <- initOptimizer optInit paramInit
  trained <- foldLoop (paramInit, optimizer) numIter $ \(paramState, optState) i -> do
    let lossValue = (lossConvQuad a b) paramState
    showLog 1000 i numIter lossValue paramState
    runStep paramState optState lossValue 5e-4
  pure ()

-- | Optimize Rosenbrock function with specified optimizer
optRosen :: (CppOptimizer opt) => Int -> opt -> IO ()
optRosen numIter optInit = do
  paramInit <- sample RosenSpec
  putStrLn ("Initial :" ++ show paramInit)
  optimizer <- initOptimizer optInit paramInit
  trained <- foldLoop (paramInit, optimizer) numIter $ \(paramState, optState) i -> do
    let lossValue = lossRosen paramState
    showLog 1000 i numIter lossValue paramState
    runStep paramState optState lossValue 5e-4
  pure ()

-- | Optimize Ackley function with specified optimizer
optAckley :: (CppOptimizer opt) => Int -> opt -> IO ()
optAckley numIter optInit = do
  paramInit <- sample AckleySpec
  putStrLn ("Initial :" ++ show paramInit)
  optimizer <- initOptimizer optInit paramInit
  trained <- foldLoop (paramInit, optimizer) numIter $ \(paramState, optState) i -> do
    let lossValue = lossAckley paramState
    showLog 1000 i numIter lossValue paramState
    runStep paramState optState lossValue 5e-4
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
      adamOpt =
        ( def
            { adamLr = 1e-4,
              adamBetas = (0.9, 0.999),
              adamEps = 1e-8,
              adamWeightDecay = 0,
              adamAmsgrad = False
            } ::
            AdamOptions
        )

  -- Convex Quadratic w/ GD, GD+Momentum, Adam
  putStrLn "\nConvex Quadratic\n================"
  putStrLn "\nGD"
  optConvQuad numIter (def :: SGDOptions)
  putStrLn "\nAdam"
  optConvQuad numIter adamOpt
  checkGlobalMinConvQuad

  -- 2D Rosenbrock w/ GD, GD+Momentum, Adam
  putStrLn "\n2D Rosenbrock\n================"
  putStrLn "\nGD"
  optRosen numIter (def :: SGDOptions)
  putStrLn "\nAdam"
  optRosen numIter adamOpt
  checkGlobalMinRosen

  -- Ackley w/ GD, GD+Momentum, Adam
  putStrLn "\nAckley (Gradient methods fail)\n================"
  putStrLn "\nGD"
  optAckley numIter (def :: SGDOptions)
  putStrLn "\nAdam"
  optAckley numIter adamOpt
  checkGlobalMinAckley
