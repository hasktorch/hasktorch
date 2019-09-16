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
import Prelude hiding (sqrt)
import Text.Printf (printf)

import Torch.Tensor
import Torch.TensorFactories (eye', ones', rand', randn', zeros')
import Torch.Functions
import Torch.Autograd
import Torch.NN hiding (sgd)

import TestFunctions

-- Optimizers

type LearningRate = Tensor
type Gradient = [Tensor]

gd :: LearningRate -> [Parameter] -> Gradient -> [Tensor]
gd lr parameters gradients = zipWith step depParameters gradients
  where
    step p dp = p - (lr * dp)
    depParameters = fmap toDependent parameters

-- gradient descent with momentum
--     lr              beta      memory      parameters     gradients
gdm :: LearningRate -> Tensor -> [Tensor] -> [Parameter] -> Gradient -> [(Tensor, Tensor)]
gdm lr beta gradMemory parameters gradients = (zipWith3 step) depParameters gradients gradMemory
  where
    z' dp z = beta * z + dp
    step p dp z = let newZ = z' dp z in (p - lr * newZ, newZ)
    depParameters = fmap toDependent parameters

data Adam = Adam { 
    m1 :: [Tensor], -- 1st moment
    m2 :: [Tensor], -- 2nd moment
    modelParam :: [Tensor] 
    } deriving Show

adam :: LearningRate -> Float -> Float -> Adam -> Gradient -> Int -> Adam
adam lr beta1 beta2 Adam{..} gradients iter = Adam m1' m2' w
    where
        -- 1st & 2nd moments
        f1 m1 dp = cmul m1 beta1 + cmul dp (1 - beta1)
        f2 m2 dp = cmul m2 beta2 + cmul (dp * dp) (1 - beta2)
        m1' = zipWith f1 m1 gradients
        m2' = zipWith f2 m2 gradients
        -- averages of moments
        a beta m = cdiv m (1 - beta^(iter + 1))
        a1 = fmap (a beta1) m1'
        a2 = fmap (a beta2) m2'
        -- parameter
        eps = 1e-6
        fw wprev avg1 avg2 = wprev - lr * avg1 / (sqrt avg2 + eps)
        w = zipWith3 fw modelParam a1 a2

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
        let result =  gdm 1e-4 0.9 memory flatParameters gradients
        newFlatParam <- mapM (makeIndependent . fst) result
        pure (replaceParameters state $ newFlatParam, fmap snd result)
    pure ()
    where
        foldLoop x count block = foldM block x [1..count]

testAdam_Rosen :: Int -> IO ()
testAdam_Rosen numIters = do
    init <- sample $ RosenSpec
    let adamInit = Adam {
        m1=[zeros' [1], zeros' [1]],
        m2=[zeros' [1], zeros' [1]],
        modelParam=[zeros' [1], zeros' [1]]
        }
    putStrLn ("Initial :" ++ show init)
    trained <- foldLoop (init, adamInit) numIters $ \(state, adamState) i -> do
        let lossValue = lossRosen state
        showLog i lossValue state
        let flatParameters = flattenParameters (state :: Coord)
        let gradients = grad lossValue flatParameters
        let params = fmap toDependent flatParameters
        let adamState' = adam 1e-3 0.9 0.999 adamState gradients i
        newFlatParam <- mapM makeIndependent (modelParam adamState')
        pure (replaceParameters state $ newFlatParam, adamState')
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

testAdam_CQ :: Int -> IO ()
testAdam_CQ numIters = do
    let dim = 2
    init <- sample $ CQSpec dim
    let a = eye' dim dim
    let b = zeros' [dim]
    let adamInit = Adam {
        m1=[zeros' [1], zeros' [1]],
        m2=[zeros' [1], zeros' [1]],
        modelParam=[zeros' [1], zeros' [1]]
        }
    putStrLn ("Initial :" ++ show init)
    trained <- foldLoop (init, adamInit) numIters $ \(state, adamState) i -> do
        let lossValue = lossCQ a b state
        showLog i lossValue state
        let flatParameters = flattenParameters (state :: CQ)
        let gradients = grad lossValue flatParameters
        let params = fmap toDependent flatParameters
        let adamState' = adam 1e-3 0.9 0.999 adamState gradients i
        newFlatParam <- mapM makeIndependent (modelParam adamState')
        pure (replaceParameters state $ newFlatParam, adamState')
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

    putStrLn "2D Rosenbrock\n================"
    putStrLn "GD"
    testGD_Rosen iter
    putStrLn "GD + Momentum"
    testGDM_Rosen iter
    putStrLn "Adam"
    testAdam_Rosen iter
    checkRosen

    putStrLn "Convex Quadratic\n================"
    putStrLn "GD"
    testGD_CQ iter
    putStrLn "GD + Momentum"
    testGDM_CQ iter
    putStrLn "Adam"
    testAdam_CQ iter
    checkCQ

    putStrLn "Done"

