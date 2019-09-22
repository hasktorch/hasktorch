{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE BlockArguments #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE FlexibleContexts #-}

module Main where

{- Optimizer implementations run on standard test functions -}

import Control.Monad (foldM, when)
import GHC.Generics
import Prelude hiding (sqrt)
import Text.Printf (printf)

import Torch.Tensor
import Torch.TensorFactories (eye', zeros')
import Torch.Functions
import Torch.Autograd
import Torch.NN

import TestFunctions
import Optimizers

-- | show output after n iterations
showLog :: (Show a) => Int -> Int -> Tensor -> a -> IO ()
showLog n i lossValue state = 
    when (mod i n == 0) do
        putStrLn ("Iter: " ++ printf "%4d" i 
            ++ " | Loss:" ++ printf "%.4f" (asValue lossValue :: Float)
            ++ " | Parameters: " ++ show state)

-- | produce flattened parameters and gradient for a single iteration
runIter
    :: (Show f, Parameterized f) =>
        f -> (f -> Tensor) -> Int -> IO ([Parameter], [Tensor])
runIter state loss i = do
    showLog 1000 i lossValue state
    pure (flatParameters, gradients)
    where
        lossValue = loss state
        flatParameters = flattenParameters state
        gradients = grad lossValue flatParameters

-- | foldM as a loop with action block as the last argument
foldLoop :: Monad m => a -> Int -> (a -> Int -> m a) -> m a
foldLoop x count block = foldM block x [1..count]

-- | Optimize Rosenbrock function with specified optimizer
optRosen :: (Optimizer o) => Int -> o -> IO ()
optRosen numIters optInit = do
    paramInit <- sample RosenSpec
    putStrLn ("Initial :" ++ show paramInit)
    trained <- foldLoop (paramInit, optInit) numIters $ \(paramState, optState) i -> do
        (flatParameters, gradients) <- runIter paramState lossRosen i
        let (result, newMemory) = step 5e-4 optState flatParameters gradients
        newFlatParam <- mapM makeIndependent result
        pure $ (replaceParameters paramState $ newFlatParam, newMemory)
    pure ()

-- | Optimize convex quadratic with specified optimizer
optConvQuad :: (Optimizer o) => Int -> o -> IO ()
optConvQuad numIters optInit = do
    let dim = 2
    paramInit <- sample $ CQSpec dim
    let a = eye' dim dim
        b = zeros' [dim]
    putStrLn ("Initial :" ++ show paramInit)
    trained <- foldLoop (paramInit, optInit) numIters $ \(paramState, optState) i -> do
        (flatParameters, gradients) <- runIter paramState (lossCQ a b) i
        let (result, optState') = step 5e-4 optState flatParameters gradients
        newFlatParam <- mapM makeIndependent result
        pure $ (replaceParameters paramState $ newFlatParam, optState')
    pure ()

-- | Check global minimum point for Rosenbrock
checkGlobalMinRosen = do
    putStrLn "\nCheck Actual Global Minimum (at 1, 1):"
    print $ rosenbrock' (asTensor (1.0 :: Float)) (asTensor (1.0 :: Float))

-- | Check global minimum point for Convex Quadratic
checkGlobalMinConvQuad = do
    putStrLn "\nCheck Actual Global Minimum (at 0, 0):"
    let dim = 2
        a = eye' dim dim
        b = zeros' [dim]
    print $ convexQuadratic a b (zeros' [dim])

main :: IO ()
main = do

    let iter = 6000

    putStrLn "\n2D Rosenbrock\n================"
    putStrLn "\nGD"
    optRosen iter GD
    putStrLn "\nGD + Momentum"
    optRosen iter (GDM 0.9 [zeros' [1], zeros' [1]])
    putStrLn "\nAdam"
    optRosen iter Adam { 
        beta1=0.9, beta2=0.999,
        m1=[zeros' [1], zeros' [1]], 
        m2=[zeros' [1], zeros' [1]],
        iter=0 }
    checkGlobalMinRosen

    putStrLn "\nConvex Quadratic\n================"
    putStrLn "\nGD"
    optConvQuad iter GD
    putStrLn "\nGD + Momentum"
    optConvQuad iter (GDM 0.9 [zeros' [2]])
    putStrLn "\nAdam"
    optConvQuad iter Adam {
        beta1=0.9, beta2=0.999,
        m1=[zeros' [1], zeros' [1]], 
        m2=[zeros' [1], zeros' [1]],
        iter=0
    }
    checkGlobalMinConvQuad

    putStrLn "\nDone"

