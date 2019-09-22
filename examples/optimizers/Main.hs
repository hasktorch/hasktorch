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

class Optimizer a where
    step :: LearningRate -> a -> [Parameter] -> Gradient -> ([Tensor], a)

-- Gradient Descent

data GD = GD deriving Show

instance Optimizer GD where
    step lr dummy parameters gradients = (gd lr parameters gradients, dummy) 

-- | Gradient descent step
gd :: LearningRate -> [Parameter] -> Gradient -> [Tensor]
gd lr parameters gradients = zipWith step depParameters gradients
  where
    step p dp = p - (lr * dp)
    depParameters = fmap toDependent parameters

-- Gradient Descent with Momentum

data GDM = GDM { beta :: Float, memory :: [Tensor] } deriving Show

instance Optimizer GDM where
    step lr state parameters gradients = gdm lr state parameters gradients

-- gradient descent with momentum step
gdm 
    :: LearningRate -- ^ learning rate
    -> GDM -- ^ beta & memory
    -> [Parameter] -- ^ parameters
    -> Gradient --gradients
    -> ([Tensor], GDM)
gdm lr GDM{..} parameters gradients = (fmap fst runStep, GDM beta (fmap snd runStep))
  where
    z' dp z = mulScalar z beta + dp
    step p dp z = let newZ = z' dp z in (p - lr * newZ, newZ)
    depParameters = fmap toDependent parameters
    runStep = (zipWith3 step) depParameters gradients memory

-- Adam

-- | State representation for Adam Optimizer
data Adam = Adam { 
    beta1 :: Float,
    beta2 :: Float,
    m1 :: [Tensor], -- 1st moment
    m2 :: [Tensor], -- 2nd moment
    iter :: Int -- iteration
    } deriving Show

instance Optimizer Adam where
    step lr state parameters gradients = adam lr state parameters gradients

-- | Adap step
adam :: LearningRate -> Adam -> [Parameter] -> Gradient -> ([Tensor], Adam)
adam lr Adam{..} parameters gradients = (w, Adam beta1 beta2 m1' m2' (iter+1))
    where
        -- 1st & 2nd moments
        f1 m1 dp = mulScalar m1 beta1 + mulScalar dp (1 - beta1)
        f2 m2 dp = mulScalar m2 beta2 + mulScalar (dp * dp) (1 - beta2)
        m1' = zipWith f1 m1 gradients
        m2' = zipWith f2 m2 gradients
        -- averages of moments
        a beta m = divScalar m (1 - beta^(iter + 1))
        a1 = fmap (a beta1) m1'
        a2 = fmap (a beta2) m2'
        -- parameter update
        eps = 1e-15
        fw wprev avg1 avg2 = wprev - lr * avg1 / (sqrt avg2 + eps)
        parameters' = fmap toDependent parameters
        w = zipWith3 fw parameters' a1 a2

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
    -- TODO - incorporate optimizer
    trained <- foldLoop init numIters $ \state i -> do
        (flatParameters, gradients) <- runIter state lossRosen i
        newFlatParam <- mapM makeIndependent $ gd 5e-4 flatParameters gradients
        pure $ replaceParameters state $ newFlatParam
    pure ()

-- | Rosenbrock function gradient descent + momentum
testRosenGDM :: Int -> IO ()
testRosenGDM numIters = do
    init <- sample $ RosenSpec
    let memory = GDM 0.9 [zeros' [1], zeros' [1]]
    putStrLn ("Initial :" ++ show init)
    trained <- foldLoop (init, memory) numIters $ \(state, memory) i -> do
        (flatParameters, gradients) <- runIter state lossRosen i
        let (result, newMemory) =  gdm 5e-4 memory flatParameters gradients
        newFlatParam <- mapM makeIndependent result
        pure (replaceParameters state $ newFlatParam, newMemory)
    pure ()

-- | Rosenbrock function Adam
testRosenAdam :: Int -> IO ()
testRosenAdam numIters = do
    init <- sample $ RosenSpec
    let adamInit = Adam {
        beta1=0.9,
        beta2=0.999,
        m1=[zeros' [1], zeros' [1]],
        m2=[zeros' [1], zeros' [1]],
        iter=0
        }
    let modelParam=[zeros' [1], zeros' [1]]
    putStrLn ("Initial :" ++ show init)
    trained <- foldLoop (init, adamInit) numIters $ \(state, adamState) i -> do
        (flatParameters, gradients) <- runIter state lossRosen i
        let params = fmap toDependent flatParameters
        let (result, adamState') = adam 5e-4 adamState flatParameters gradients 
        newFlatParam <- mapM makeIndependent result
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
    let state = GDM 0.9 [zeros' [dim]]
    putStrLn ("Initial :" ++ show init)
    trained <- foldLoop (init, state) numIters $ \(paramState, optState) i -> do
        (flatParameters, gradients) <- runIter paramState (lossCQ a b) i
        let (result, optState') = gdm 5e-4 optState flatParameters gradients
        newFlatParam <- mapM makeIndependent result
        pure $ (replaceParameters paramState $ newFlatParam, optState')
    pure ()

-- | Convex Quadratic Adam
testConvQuadAdam :: Int -> IO ()
testConvQuadAdam numIters = do
    let dim = 2
    init <- sample $ CQSpec dim
    let a = eye' dim dim
        b = zeros' [dim]
    let adamInit = Adam {
        beta1=0.9,
        beta2=0.999,
        m1=[zeros' [1], zeros' [1]],
        m2=[zeros' [1], zeros' [1]],
        iter=0
    }
    let modelParam=[zeros' [1], zeros' [1]]
    putStrLn ("Initial :" ++ show init)
    trained <- foldLoop (init, adamInit) numIters $ \(paramState, adamState) i -> do
        (flatParameters, gradients) <- runIter paramState (lossCQ a b) i
        let params = fmap toDependent flatParameters
        let (result, adamState') = adam 5e-4 adamState flatParameters gradients
        newFlatParam <- mapM makeIndependent result 
        pure (replaceParameters paramState $ newFlatParam, adamState')
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

