{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleInstances #-}

module TestFunctions where

import GHC.Generics

import Torch.Tensor
import Torch.TensorFactories (eye', ones', rand', randn', zeros')
import Torch.Functions
import Torch.Autograd
import Torch.NN hiding (sgd)

-- 2D Rosenbrock

data RosenSpec = RosenSpec deriving (Show, Eq)
data Coord = Coord { x :: Parameter, y :: Parameter } deriving (Generic)

instance Show Coord where
    show (Coord x y) = show (extract x :: Float, extract y :: Float)
        where
            extract :: TensorLike a => Parameter -> a
            extract p = asValue $ toDependent p

instance Randomizable RosenSpec Coord where
  sample RosenSpec = do
      x <- makeIndependent =<< randn' [1]
      y <- makeIndependent =<< randn' [1]
      pure $ Coord x y

instance Parameterized Coord
instance Parameterized [Coord]

rosenbrock2d :: Float -> Float -> Tensor -> Tensor -> Tensor
rosenbrock2d a b x y = square (addScalar ((-1.0) * x ) a) + mulScalar (square (y - x*x)) b
    where square c = pow c (2 :: Int)

rosenbrock' :: Tensor -> Tensor -> Tensor
rosenbrock' = rosenbrock2d 1.0 100.0

lossRosen :: Coord -> Tensor
lossRosen  Coord{..} = rosenbrock' (toDependent x) (toDependent y)

-- Convex Quadratic

data CQSpec = CQSpec { n :: Int }
data CQ = CQ { w :: Parameter } deriving (Show, Generic)

instance Randomizable CQSpec CQ where
  sample (CQSpec n) = do
        w <- makeIndependent =<<randn' [n]
        pure $ CQ w

instance Parameterized CQ
instance Parameterized [CQ]

convexQuadratic :: Tensor -> Tensor -> Tensor -> Tensor
convexQuadratic a b w =
    mulScalar (dot w (mv a w)) (0.5 :: Float) - dot b w

lossCQ :: Tensor -> Tensor -> CQ -> Tensor
lossCQ a b (CQ w) = convexQuadratic a b w'
    where w' = toDependent w
