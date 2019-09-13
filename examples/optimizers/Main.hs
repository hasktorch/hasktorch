{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE BlockArguments #-}
{-# LANGUAGE ScopedTypeVariables #-}

{- Optimizer Implementations on math functions -}

import Control.Monad (foldM, when)
import GHC.Generics
import Text.Printf (printf)

import Torch.Tensor
import Torch.TensorFactories (eye', ones', rand', randn', zeros')
import Torch.Functions
import Torch.Autograd
import Torch.NN hiding (sgd)

-- 2d Rosenbrock

data RosenSpec = RosenSpec deriving (Show, Eq)
data Coord = Coord { x :: Parameter, y :: Parameter } deriving (Show, Generic)

instance Randomizable RosenSpec Coord where
  sample RosenSpec = do
      x <- makeIndependent =<< randn' [1]
      y <- makeIndependent =<< randn' [1]
      pure $ Coord x y

instance Parameterized Coord
instance Parameterized [Coord]

rosenbrock2d :: Float -> Float -> Tensor -> Tensor -> Tensor
rosenbrock2d a b x y = square (cadd ((-1.0) * x ) a) + cmul (square (y - x*x)) b
    where square c = pow c (2 :: Int)

rosenbrock' :: Tensor -> Tensor -> Tensor
rosenbrock' = rosenbrock2d 1.0 100.0

lossRosen :: Coord -> Tensor
lossRosen  Coord{..} = rosenbrock' (toDependent x) (toDependent y)

-- Convex Quadratic

data CQSpec = CQSpec { n :: Int }
data CQ = CQ { pos :: Parameter } deriving (Show, Generic)

instance Randomizable CQSpec CQ where
  sample (CQSpec n) = do
        x <- makeIndependent =<<randn' [n]
        pure $ CQ x

instance Parameterized CQ
instance Parameterized [CQ]

convexQuadratic :: CQ -> Tensor -> Tensor -> Tensor
convexQuadratic (CQ mat) b w =
    cmul (matmul (matmul (transpose2D w) mat') w) (0.5 :: Float) - matmul (transpose2D b) mat'
    where mat' = toDependent mat

-- convexQuadratic' = convexQuadratic (CQ (eye' 3 3))  

lossCQ :: CQ-> Tensor -> Tensor -> Tensor
lossCQ (CQ pos) = convexQuadratic (CQ pos)

-- Convex Quadratic

-- Optimizers

gd :: Tensor -> [Parameter] -> [Tensor] -> [Tensor]
gd lr parameters gradients = zipWith step depParameters gradients
  where
    step p dp = p - (lr * dp)
    depParameters = fmap toDependent parameters

gdMomentum :: Tensor -> Tensor -> [Tensor] -> [Parameter] -> [Tensor] -> [(Tensor, Tensor)]
gdMomentum lr beta gradMemory parameters gradients = zipWith3 step depParameters gradients gradMemory
  where
    z' dp z = beta * z + dp
    step p dp z = let newZ = z' dp z in (p - lr * newZ, newZ)
    depParameters = fmap toDependent parameters

showParam (Coord x y) = show (extract x :: Float, extract y :: Float)
  where
    extract :: TensorLike a => Parameter -> a
    extract p = asValue $ toDependent p

testGD_Rosen :: Int -> IO ()
testGD_Rosen numIters = do
    init <- sample $ RosenSpec
    putStrLn ("Initial :" ++ showParam init)
    trained <- foldLoop init numIters $ \state i -> do
        let lossValue = lossRosen state
        when (mod i 100 == 0) do
            putStrLn ("Iter: " ++ printf "%4d" i 
                ++ " | Loss:" ++ printf "%.4f" (asValue lossValue :: Float)
                ++ " | Parameters: " ++ showParam state)
        let flatParameters = flattenParameters (state :: Coord)
        let gradients = grad lossValue flatParameters
        newFlatParam <- mapM makeIndependent $ gd 2e-3 flatParameters gradients
        pure $ replaceParameters state $ newFlatParam
    pure ()
    where
        foldLoop x count block = foldM block x [1..count]

testGD_CQ :: Int -> IO ()
testGD_CQ numIters = do
    init <- sample $ CQSpec 1 
    putStrLn ("Initial :" ++ show init)
    trained <- foldLoop init numIters $ \state i -> do
        let lossValue = lossCQ state undefined undefined -- TOOD
        when (mod i 100 == 0) do
            putStrLn ("Iter: " ++ printf "%4d" i 
                ++ " | Loss:" ++ printf "%.4f" (asValue lossValue :: Float)
                ++ " | Parameters: " ++ show state)
        let flatParameters = flattenParameters (state :: CQ)
        let gradients = grad lossValue flatParameters
        newFlatParam <- mapM makeIndependent $ gd 2e-3 flatParameters gradients
        pure $ replaceParameters state $ newFlatParam
    pure ()
    where
        foldLoop x count block = foldM block x [1..count]

main :: IO ()
main = do
    testGD_Rosen  5000
    putStrLn "Check Actual Global Minimum (at 1, 1):"
    print $ rosenbrock' (asTensor (1.0 :: Float)) (asTensor (1.0 :: Float))
    testGD_CQ 5000
    putStrLn "Done"

