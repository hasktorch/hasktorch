{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}

module TestFunctions where

import GHC.Generics
import Torch.Autograd
import Torch.Functional
import Torch.NN hiding (sgd)
import Torch.Tensor
import Torch.TensorFactories (eye', ones', randIO', randnIO', zeros')
import Prelude hiding (cos, exp, sqrt)
import qualified Prelude as P

-- Convex Quadratic

data ConvQuadSpec = ConvQuadSpec {n :: Int}

data ConvQuad = ConvQuad {w :: Parameter} deriving (Show, Generic)

instance Randomizable ConvQuadSpec ConvQuad where
  sample (ConvQuadSpec n) = do
    w <- makeIndependent =<< randnIO' [n]
    pure $ ConvQuad w

instance Parameterized ConvQuad

convexQuadratic :: Tensor -> Tensor -> Tensor -> Tensor
convexQuadratic a b w =
  mulScalar (0.5 :: Float) (dot w (mv a w)) - dot b w

lossConvQuad :: Tensor -> Tensor -> ConvQuad -> Tensor
lossConvQuad a b (ConvQuad w) = convexQuadratic a b w'
  where
    w' = toDependent w

-- 2D Rosenbrock

data RosenSpec = RosenSpec deriving (Show, Eq)

data Rosen = Rosen {x :: Parameter, y :: Parameter} deriving (Generic)

instance Show Rosen where
  show (Rosen x y) = show (extract x :: Float, extract y :: Float)
    where
      extract :: TensorLike a => Parameter -> a
      extract p = asValue $ toDependent p

instance Randomizable RosenSpec Rosen where
  sample RosenSpec = do
    x <- makeIndependent =<< randnIO' [1]
    y <- makeIndependent =<< randnIO' [1]
    pure $ Rosen x y

-- instance Parameterized Rosen

instance Parameterized Rosen where
  -- flattenParameters :: f -> [Parameter]
  flattenParameters (Rosen x y) = [x, y]

rosenbrock2d :: Float -> Float -> Tensor -> Tensor -> Tensor
rosenbrock2d a b x y = square (addScalar a ((-1.0) * x)) + mulScalar b (square (y - x * x))
  where
    square c = pow (2 :: Int) c

rosenbrock' :: Tensor -> Tensor -> Tensor
rosenbrock' = rosenbrock2d 1.0 100.0

lossRosen :: Rosen -> Tensor
lossRosen Rosen {..} = rosenbrock' (toDependent x) (toDependent y)

-- Ackley function

data AckleySpec = AckleySpec deriving (Show, Eq)

data Ackley = Ackley {pos :: Parameter} deriving (Show, Generic)

instance Randomizable AckleySpec Ackley where
  sample AckleySpec = do
    pos <- makeIndependent =<< randnIO' [2]
    pure $ Ackley pos

instance Parameterized Ackley

ackley :: Float -> Float -> Float -> Tensor -> Tensor
ackley a b c x =
  mulScalar (- a) (exp (- b' * (sqrt $ (sumAll (x * x)) / d)))
    - exp (1.0 / d * sumAll (cos (mulScalar c x)))
    + (asTensor $ a + P.exp 1.0)
  where
    b' = asTensor b
    c' = asTensor c
    d = asTensor . product $ shape x

ackley' = ackley 20.0 0.2 (2 * pi :: Float)

lossAckley :: Ackley -> Tensor
lossAckley (Ackley x) = ackley' x'
  where
    x' = toDependent x
