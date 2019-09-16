{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE BlockArguments #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Main where

{- Optimizer implementations run on standard test functions -}

import Control.Monad (foldM, when)
import GHC.Generics
import Text.Printf (printf)

import Torch.Tensor
import Torch.TensorFactories (eye', ones', rand', randn', zeros')
import Torch.Functions
import Torch.Autograd
import Torch.NN hiding (sgd)

import TestFunctions

-- Optimizers

gd :: Tensor -> [Parameter] -> [Tensor] -> [Tensor]
gd lr parameters gradients = zipWith step depParameters gradients
  where
    step p dp = p - (lr * dp)
    depParameters = fmap toDependent parameters

-- gradient descent with momentum
--     lr        beta      memory      parameters     gradients
gdm :: Tensor -> Tensor -> [Tensor] -> [Parameter] -> [Tensor] -> [(Tensor, Tensor)]
gdm lr beta gradMemory parameters gradients = zipWith3 step depParameters gradients gradMemory
  where
    z' dp z = beta * z + dp
    step p dp z = let newZ = z' dp z in (p - lr * newZ, newZ)
    depParameters = fmap toDependent parameters

showLog :: (Show a) => Int -> Tensor -> a -> IO ()
showLog i lossValue state = 
    when (mod i 1000 == 0) do
        putStrLn ("Iter: " ++ printf "%4d" i 
            ++ " | Loss:" ++ printf "%.4f" (asValue lossValue :: Float)
            ++ " | Parameters: " ++ show state)

testGD_Rosen :: Int -> IO ()
testGD_Rosen numIters = do
    init <- sample $ RosenSpec
    putStrLn ("Initial :" ++ show init)
    trained <- foldLoop init numIters $ \state i -> do
        let lossValue = lossRosen state
        showLog i lossValue state
        let flatParameters = flattenParameters (state :: Coord)
        let gradients = grad lossValue flatParameters
        newFlatParam <- mapM makeIndependent $ gd 1e-3 flatParameters gradients
        pure $ replaceParameters state $ newFlatParam
    pure ()
    where
        foldLoop x count block = foldM block x [1..count]

testGD_CQ :: Int -> IO ()
testGD_CQ numIters = do
    let dim = 2
    (init :: CQ) <- sample $ CQSpec dim
    let a = eye' dim dim
    let b = zeros' [dim]
    putStrLn ("Initial :" ++ show init)
    trained <- foldLoop init numIters $ \state i -> do
        let lossValue = lossCQ a b state
        showLog i lossValue state
        let flatParameters = flattenParameters (state :: CQ)
        let gradients = grad lossValue flatParameters
        newFlatParam <- mapM makeIndependent $ gd 1e-3 flatParameters gradients
        pure $ replaceParameters state $ newFlatParam
    pure ()
    where
        foldLoop x count block = foldM block x [1..count]


testGDM_Rosen :: Int -> IO ()
testGDM_Rosen numIters = do
    init <- sample $ RosenSpec
    let memory = [zeros' [1], zeros' [1]]
    putStrLn ("Initial :" ++ show init)
    trained <- foldLoop (init, memory) numIters $ \(state, memory) i -> do
        let lossValue = lossRosen state
        showLog i lossValue state
        let flatParameters = flattenParameters (state :: Coord)
        let gradients = grad lossValue flatParameters
        let result =  gdm 1e-3 0.9 memory flatParameters gradients
        newFlatParam <- mapM (makeIndependent . fst) result
        pure (replaceParameters state $ newFlatParam, fmap snd result)
    pure ()
    where
        foldLoop x count block = foldM block x [1..count]

testGDM_CQ :: Int -> IO ()
testGDM_CQ numIters = do
    let dim = 2
    init <- sample $ CQSpec dim
    let a = eye' dim dim
    let b = zeros' [dim]
    let z = [zeros' [dim]] -- momentum
    putStrLn ("Initial :" ++ show init)
    trained <- foldLoop (init, z) numIters $ \(state, z) i -> do
        let lossValue = lossCQ a b state
        showLog i lossValue state
        let flatParameters = flattenParameters (state :: CQ)
        let gradients = grad lossValue flatParameters
        let result = gdm 1e-3 0.9 z flatParameters gradients
        newFlatParam <- mapM (makeIndependent . fst) result
        pure $ (replaceParameters state $ newFlatParam, fmap snd result)
    pure ()
    where
        foldLoop x count block = foldM block x [1..count]

checkRosen = do
    putStrLn "Check Actual Global Minimum (at 1, 1):"
    print $ rosenbrock' (asTensor (1.0 :: Float)) (asTensor (1.0 :: Float))

checkCQ = do
    putStrLn "Check Actual Global Minimum (at 0, 0):"
    let dim = 2
    let (a, b) = (eye' dim dim, zeros' [dim])
    print $ convexQuadratic a b (zeros' [dim])


main :: IO ()
main = do

    let iter = 6000

    testGD_Rosen iter
    checkRosen

    testGDM_Rosen iter
    checkRosen

    testGD_CQ iter
    checkCQ

    testGDM_CQ iter
    checkCQ

    putStrLn "Done"

