{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE DeriveGeneric #-}

module TestFunctions where

import GHC.Generics

import Prelude hiding (exp, cos, sqrt)
import qualified Prelude as P

import Torch.Tensor
import Torch.TensorFactories (eye', ones', rand', randn', zeros')
import Torch.Functional
import Torch.Autograd
import Torch.NN hiding (sgd)

-- Convex Quadratic

data ConvQuadSpec = ConvQuadSpec { n :: Int }
data ConvQuad = ConvQuad { w :: Parameter } deriving (Show, Generic)

instance Randomizable ConvQuadSpec ConvQuad where
  sample (ConvQuadSpec n) = do
        w <- makeIndependent =<<randn' [n]
        pure $ ConvQuad w

instance Parameterized ConvQuad

convexQuadratic :: Tensor -> Tensor -> Tensor -> Tensor
convexQuadratic a b w =
    mulScalar (dot w (mv a w)) (0.5 :: Float) - dot b w

lossConvQuad :: Tensor -> Tensor -> ConvQuad -> Tensor
lossConvQuad a b (ConvQuad w) = convexQuadratic a b w'
    where w' = toDependent w

-- 2D Rosenbrock

data RosenSpec = RosenSpec deriving (Show, Eq)
data Rosen = Rosen { x :: Parameter, y :: Parameter } deriving (Generic)

instance Show Rosen where
    show (Rosen x y) = show (extract x :: Float, extract y :: Float)
        where
            extract :: TensorLike a => Parameter -> a
            extract p = asValue $ toDependent p

instance Randomizable RosenSpec Rosen where
  sample RosenSpec = do
      x <- makeIndependent =<< randn' [1]
      y <- makeIndependent =<< randn' [1]
      pure $ Rosen x y

-- instance Parameterized Rosen

instance Parameterized Rosen where
  -- flattenParameters :: f -> [Parameter]
  flattenParameters (Rosen x y) = [x, y]

rosenbrock2d :: Float -> Float -> Tensor -> Tensor -> Tensor
rosenbrock2d a b x y = square (addScalar ((-1.0) * x ) a) + mulScalar (square (y - x*x)) b
    where square c = pow c (2 :: Int)

rosenbrock' :: Tensor -> Tensor -> Tensor
rosenbrock' = rosenbrock2d 1.0 100.0

lossRosen :: Rosen -> Tensor
lossRosen  Rosen{..} = rosenbrock' (toDependent x) (toDependent y)

-- Ackley function

data AckleySpec = AckleySpec deriving (Show, Eq)
data Ackley = Ackley { pos :: Parameter } deriving (Show, Generic)

instance Randomizable AckleySpec Ackley where
  sample AckleySpec = do
      pos <- makeIndependent =<< randn' [2]
      pure $ Ackley pos

instance Parameterized Ackley

ackley :: Float -> Float -> Float -> Tensor -> Tensor
ackley a b c x = 
    mulScalar (exp (-b' * (sqrt $ (sumAll (x * x)) / d))) (-a)
    - exp (1.0 / d * sumAll (cos (mulScalar x c))) 
    + (asTensor $ a + P.exp 1.0)
    where
        b' = asTensor b
        c' = asTensor c
        d = asTensor . product $ shape x

ackley' = ackley 20.0 0.2 (2*pi :: Float)

lossAckley :: Ackley -> Tensor
lossAckley (Ackley x) = ackley' x'
    where x' = toDependent x
