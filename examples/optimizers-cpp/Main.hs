{-# LANGUAGE BlockArguments #-}

module Main where

{- Optimizer implementations run on standard test functions -}

import Control.Monad (foldM, when, forM_)
import GHC.Generics
import Prelude hiding (sqrt)
import Text.Printf (printf)
import Data.Default.Class

import Torch hiding (Optimizer(..), runStep, Adam(..))
import TestFunctions
import Torch.Optim.Internal

-- import Data.IORef

-- | show output after n iterations
showLog :: (Show a) => Int -> Int -> Int -> Tensor -> a -> IO ()
showLog n i maxIter lossValue state = 
    when (i == 0 || mod i n == 0 || i == maxIter-1) do
        putStrLn ("Iter: " ++ printf "%6d" i
            ++ " | Loss:" ++ printf "%05.4f" (asValue lossValue :: Float)
            ++ " | Parameters: " ++ show state)

-- | Optimize convex quadratic with specified optimizer
optConvQuad :: (Optimizer opt) => Int -> opt -> IO ()
optConvQuad numIter optInit = do
    let dim = 2
        a = eye' dim dim
        b = zeros' [dim]
    paramInit <- sample $ ConvQuadSpec dim
    putStrLn ("Initial :" ++ show paramInit)
    optimizer <- initOptimizer optInit paramInit
    forM_ [1..numIter] \i -> do
      step optimizer $ \paramState -> do
        let lossValue = (lossConvQuad a b) paramState
        showLog 1000 i numIter lossValue paramState
        return lossValue
    trained <- getParams optimizer :: IO ConvQuad
    -- ref <- newIORef 0
    -- steps numIter optInit paramInit $ \paramState -> do
    --     let lossValue = (lossConvQuad a b) paramState
    --     i <- atomicModifyIORef ref $ \v -> (v+1,v)
    --     showLog 1000 i numIter lossValue paramState
    --     return lossValue
    pure ()

-- | Optimize Rosenbrock function with specified optimizer
optRosen :: (Optimizer opt) => Int -> opt -> IO ()
optRosen numIter optInit = do
    paramInit <- sample RosenSpec
    putStrLn ("Initial :" ++ show paramInit)
    optimizer <- initOptimizer optInit paramInit
--    print paramInit
    forM_ [1..numIter] \i -> do
      step optimizer $ \paramState -> do
        let lossValue = lossRosen paramState
--        print paramState
--        print lossValue
        showLog 1000 i numIter lossValue paramState
        return lossValue
    trained <- getParams optimizer :: IO Rosen
    pure ()

-- | Optimize Ackley function with specified optimizer
optAckley :: (Optimizer opt) => Int -> opt -> IO ()
optAckley numIter optInit = do
    paramInit <- sample AckleySpec
    putStrLn ("Initial :" ++ show paramInit)
    optimizer <- initOptimizer optInit paramInit
    forM_ [1..numIter] \i -> do
      step optimizer $ \paramState -> do
        let lossValue = lossAckley paramState
        showLog 1000 i numIter lossValue paramState
        return lossValue
    trained <- getParams optimizer :: IO Ackley
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
        opt = (def { adamLr = 1e-4
                   , adamBetas = (0.9, 0.999)
                   , adamEps = 1e-8
                   , adamWeightDecay = 0
                   , adamAmsgrad = False
                   } :: AdamOptions)

    -- Convex Quadratic w/ GD, GD+Momentum, Adam
    putStrLn "\nConvex Quadratic\n================"
    putStrLn "\nAdam"
    optConvQuad numIter opt
    checkGlobalMinConvQuad

    -- 2D Rosenbrock w/ GD, GD+Momentum, Adam
    putStrLn "\n2D Rosenbrock\n================"
    putStrLn "\nAdam"
    optRosen numIter opt
    checkGlobalMinRosen

    -- Ackley w/ GD, GD+Momentum, Adam
    putStrLn "\nAckley (Gradient methods fail)\n================"
    putStrLn "\nAdam"
    optAckley numIter opt
    checkGlobalMinAckley
