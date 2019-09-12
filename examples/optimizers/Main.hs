{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE BlockArguments #-}

{- Optimizer Implementations on math functions -}

import Control.Monad (foldM, when)
import GHC.Generics
import Text.Printf (printf)

import Torch.Tensor
import Torch.TensorFactories (eye', ones', rand', randn', zeros')
import Torch.Functions
import Torch.Autograd
import Torch.NN hiding (sgd)

data Coord = Coord { x :: Parameter, y :: Parameter } deriving (Show, Generic)

-- note - only works for n=1 for now
data CoordSpec = CoordSpec { n :: Int } deriving (Show, Eq)

instance Randomizable CoordSpec Coord where
  sample (CoordSpec n) = do
      x <- makeIndependent =<< randn' [n]
      y <- makeIndependent =<< randn' [n]
      -- x <- makeIndependent (ones' [n] + 0.5 * ones' [n]) -- check with fixed values
      -- y <- makeIndependent (ones' [n] + 0.5 * ones' [n])
      pure $ Coord x y

instance Parameterized Coord

instance Parameterized [Coord]

rosenbrock :: Float -> Float -> Tensor -> Tensor -> Tensor
rosenbrock a b x y = square (cadd ((-1.0) * x ) a) + cmul (square (y - x*x)) b
    where square c = pow c (2 :: Int)

rosenbrock' :: Tensor -> Tensor -> Tensor
rosenbrock' = rosenbrock 1.0 100.0

loss :: Coord -> Tensor
loss Coord{..} = rosenbrock' (toDependent x) (toDependent y)

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

testGD :: Int -> IO ()
testGD numIters = do
    init <- sample $ CoordSpec {n=1}
    putStrLn ("Initial :" ++ showParam init)
    trained <- foldLoop init numIters $ \state i -> do
        let lossValue = loss state
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

main :: IO ()
main = do
    testGD 5000
    putStrLn "Check Actual Global Minimum (at 1, 1):"
    print $ rosenbrock' (asTensor (1.0 :: Float)) (asTensor (1.0 :: Float))
    putStrLn "Done"

