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
import Torch.TensorFactories (eye', zeros')
import Torch.Functions
import Torch.Autograd
import Torch.NN

import TestFunctions

-- Optimizers

type LearningRate = Tensor
type Gradient = [Tensor]

-- gradient descent step
gd :: LearningRate -> [Parameter] -> Gradient -> [Tensor]
gd lr parameters gradients = zipWith step depParameters gradients
  where
    step p dp = p - (lr * dp)
    depParameters = fmap toDependent parameters

-- gradient descent with momentum step
--     lr              beta      memory      parameters     gradients
gdm :: LearningRate -> Float -> [Tensor] -> [Parameter] -> Gradient -> [(Tensor, Tensor)]
gdm lr beta gradMemory parameters gradients = (zipWith3 step) depParameters gradients gradMemory
  where
    z' dp z = cmul z beta + dp
    step p dp z = let newZ = z' dp z in (p - lr * newZ, newZ)
    depParameters = fmap toDependent parameters

-- Adam

-- | State representation for Adam Optimizer
data Adam = Adam { 
    m1 :: [Tensor], -- 1st moment
    m2 :: [Tensor], -- 2nd moment
    modelParam :: [Tensor] 
    } deriving Show

-- | Adap step
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
        -- parameter update
        eps = 1e-15
        fw wprev avg1 avg2 = wprev - lr * avg1 / (sqrt avg2 + eps)
        w = zipWith3 fw modelParam a1 a2

-- | show output after n iterations
showLog :: (Show a) => Int -> Int -> Tensor -> a -> IO ()
showLog n i lossValue state = 
    when (mod i n == 0) do
        putStrLn ("Iter: " ++ printf "%4d" i 
            ++ " | Loss:" ++ printf "%.4f" (asValue lossValue :: Float)
            ++ " | Parameters: " ++ show state)

-- | produce flattened parameters and gradient for a single iteration
runIter state loss i = do
    showLog 1000 i lossValue state
    pure (flattenParameters state, grad lossValue flatParameters)
    where
        lossValue = loss state
        flatParameters = flattenParameters state
        gradients = grad lossValue flatParameters

-- | foldM as a loop with action block as the last argument
foldLoop :: Monad m => a -> Int -> (a -> Int -> m a) -> m a
foldLoop x count block = foldM block x [1..count]

-- | Rosenbrock function gradient descent
testRosenGD :: Int -> IO ()
testRosenGD numIters = do
    init <- sample $ RosenSpec
    putStrLn ("Initial :" ++ show init)
    trained <- foldLoop init numIters $ \state i -> do
        (flatParameters, gradients) <- runIter state lossRosen i
        newFlatParam <- mapM makeIndependent $ gd 5e-4 flatParameters gradients
        pure $ replaceParameters state $ newFlatParam
    pure ()

-- | Rosenbrock function gradient descent + momentum
testRosenGDM :: Int -> IO ()
testRosenGDM numIters = do
    init <- sample $ RosenSpec
    let memory = [zeros' [1], zeros' [1]]
    putStrLn ("Initial :" ++ show init)
    trained <- foldLoop (init, memory) numIters $ \(state, memory) i -> do
        (flatParameters, gradients) <- runIter state lossRosen i
        let result =  gdm 5e-4 0.9 memory flatParameters gradients
        newFlatParam <- mapM (makeIndependent . fst) result
        pure (replaceParameters state $ newFlatParam, fmap snd result)
    pure ()

-- | Rosenbrock function Adam
testRosenAdam :: Int -> IO ()
testRosenAdam numIters = do
    init <- sample $ RosenSpec
    let adamInit = Adam {
        m1=[zeros' [1], zeros' [1]],
        m2=[zeros' [1], zeros' [1]],
        modelParam=[zeros' [1], zeros' [1]]
        }
    putStrLn ("Initial :" ++ show init)
    trained <- foldLoop (init, adamInit) numIters $ \(state, adamState) i -> do
        (flatParameters, gradients) <- runIter state lossRosen i
        let params = fmap toDependent flatParameters
        let adamState' = adam 5e-4 0.9 0.999 adamState gradients i
        newFlatParam <- mapM makeIndependent (modelParam adamState')
        pure (replaceParameters state $ newFlatParam, adamState')
    pure ()

-- | Convex Quadratic Gradient Descent
testConvQuadGD :: Int -> IO ()
testConvQuadGD numIters = do
    let dim = 2
    (init :: CQ) <- sample $ CQSpec dim
    let a = eye' dim dim
    let b = zeros' [dim]
    putStrLn ("Initial :" ++ show init)
    trained <- foldLoop init numIters $ \state i -> do
        (flatParameters, gradients) <- runIter state (lossCQ a b) i
        newFlatParam <- mapM makeIndependent $ gd 5e-4 flatParameters gradients
        pure $ replaceParameters state $ newFlatParam
    pure ()

-- | Convex Quadratic Gradient Descent + Momentum
testConvQuadGDM :: Int -> IO ()
testConvQuadGDM numIters = do
    let dim = 2
    init <- sample $ CQSpec dim
    let a = eye' dim dim
        b = zeros' [dim]
        z = [zeros' [dim]] -- momentum
    putStrLn ("Initial :" ++ show init)
    trained <- foldLoop (init, z) numIters $ \(state, z) i -> do
        (flatParameters, gradients) <- runIter state (lossCQ a b) i
        let result = gdm 5e-4 0.9 z flatParameters gradients
        newFlatParam <- mapM (makeIndependent . fst) result
        pure $ (replaceParameters state $ newFlatParam, fmap snd result)
    pure ()

-- | Convex Quadratic Adam
testConvQuadAdam :: Int -> IO ()
testConvQuadAdam numIters = do
    let dim = 2
    init <- sample $ CQSpec dim
    let a = eye' dim dim
        b = zeros' [dim]
        adamInit = Adam {
            m1=[zeros' [1], zeros' [1]],
            m2=[zeros' [1], zeros' [1]],
            modelParam=[zeros' [1], zeros' [1]]
            }
    putStrLn ("Initial :" ++ show init)
    trained <- foldLoop (init, adamInit) numIters $ \(state, adamState) i -> do
        (flatParameters, gradients) <- runIter state (lossCQ a b) i
        let params = fmap toDependent flatParameters
        let adamState' = adam 5e-4 0.9 0.999 adamState gradients i
        newFlatParam <- mapM makeIndependent (modelParam adamState')
        pure (replaceParameters state $ newFlatParam, adamState')
    pure ()

-- | Check global minimum point for Rosenbrock
checkRosen = do
    putStrLn "\nCheck Actual Global Minimum (at 1, 1):"
    print $ rosenbrock' (asTensor (1.0 :: Float)) (asTensor (1.0 :: Float))

-- | Check global minimum point for Convex Quadratic
checkCQ = do
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
    testRosenGD iter
    putStrLn "\nGD + Momentum"
    testRosenGDM iter
    putStrLn "\nAdam"
    testRosenAdam iter
    checkRosen

    putStrLn "\nConvex Quadratic\n================"
    putStrLn "\nGD"
    testConvQuadGD iter
    putStrLn "\nGD + Momentum"
    testConvQuadGDM iter
    putStrLn "\nAdam"
    testConvQuadAdam iter
    checkCQ

    putStrLn "\nDone"

