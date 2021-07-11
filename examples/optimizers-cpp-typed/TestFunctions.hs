{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE UndecidableInstances #-}

module TestFunctions where

import GHC.Generics
import GHC.TypeNats
import Torch.Typed
import Torch.Typed.Autograd
import Torch.Typed.Factories (eyeSquare, ones, rand, randn, zeros)
import Torch.Typed.Functional
import Torch.Typed.NN hiding (sgd)
import Prelude hiding (cos, exp, sqrt)
import qualified Prelude as P

-- Convex Quadratic

data ConvQuadSpec (dev :: (DeviceType, Nat)) (dtype :: DType) (n :: Nat)
  = ConvQuadSpec

data ConvQuad dev dtype n = ConvQuad {w :: Parameter dev dtype '[n]}
  deriving (Show, Generic)

instance
  ( RandDTypeIsValid dev dtype,
    KnownNat n,
    KnownDType dtype,
    KnownDevice dev
  ) =>
  Randomizable (ConvQuadSpec dev dtype n) (ConvQuad dev dtype n)
  where
  sample ConvQuadSpec = do
    w <- makeIndependent =<< randn
    pure $ ConvQuad w

instance Parameterized (ConvQuad dev dtype n)

convexQuadratic ::
  forall dev dtype n.
  ( DotDTypeIsValid dev dtype,
    KnownDevice dev,
    KnownDType dtype,
    KnownNat n
  ) =>
  Tensor dev dtype '[n, n] ->
  Tensor dev dtype '[n] ->
  Tensor dev dtype '[n] ->
  Tensor dev dtype '[]
convexQuadratic a b w =
  mulScalar (0.5 :: Float) (dot w (mv a w)) - dot b w

lossConvQuad ::
  (DotDTypeIsValid dev dtype, KnownDevice dev, KnownDType dtype, KnownNat n) =>
  Tensor dev dtype '[n, n] ->
  Tensor dev dtype '[n] ->
  ConvQuad dev dtype n ->
  Tensor dev dtype '[]
lossConvQuad a b (ConvQuad w) = convexQuadratic a b w'
  where
    w' = toDependent w

-- 2D Rosenbrock

data RosenSpec (dev :: (DeviceType, Nat)) (dtype :: DType)
  = RosenSpec
  deriving (Show, Eq)

data Rosen dev dtype = Rosen
  { x :: Parameter dev dtype '[],
    y :: Parameter dev dtype '[]
  }
  deriving (Generic)

instance Show (Rosen dev 'Float) where
  show (Rosen x y) = show (toFloat $ toDependent x, toFloat $ toDependent y)

instance
  (RandDTypeIsValid dev dtype, KnownDType dtype, KnownDevice dev) =>
  Randomizable (RosenSpec dev dtype) (Rosen dev dtype)
  where
  sample RosenSpec = do
    x <- makeIndependent =<< randn
    y <- makeIndependent =<< randn
    pure $ Rosen x y

-- instance Parameterized Rosen

instance Parameterized (Rosen dev dtype)

rosenbrock2d ::
  KnownDevice dev =>
  Float ->
  Float ->
  Tensor dev 'Float '[] ->
  Tensor dev 'Float '[] ->
  Tensor dev 'Float '[]
rosenbrock2d a b x y =
  square (addScalar a ((-1.0) * x))
    + mulScalar b (square (y - x * x))
  where
    square c = powScalar (2 :: Float) c

rosenbrock' ::
  KnownDevice dev =>
  Tensor dev 'Float '[] ->
  Tensor dev 'Float '[] ->
  Tensor dev 'Float '[]
rosenbrock' = rosenbrock2d 1.0 100.0

lossRosen :: KnownDevice dev => Rosen dev 'Float -> Tensor dev 'Float '[]
lossRosen Rosen {..} = rosenbrock' (toDependent x) (toDependent y)

--
-- -- Ackley function
--
-- data AckleySpec = AckleySpec deriving (Show, Eq)
-- data Ackley = Ackley { pos :: Parameter } deriving (Show, Generic)
--
-- instance Randomizable AckleySpec Ackley where
--   sample AckleySpec = do
--       pos <- makeIndependent =<< randnIO' [2]
--       pure $ Ackley pos
--
-- instance Parameterized Ackley
--
-- ackley :: Float -> Float -> Float -> Tensor -> Tensor
-- ackley a b c x =
--     mulScalar (-a) (exp (-b' * (sqrt $ (sumAll (x * x)) / d)))
--     - exp (1.0 / d * sumAll (cos (mulScalar c x)))
--     + (asTensor $ a + P.exp 1.0)
--     where
--         b' = asTensor b
--         c' = asTensor c
--         d = asTensor . product $ shape x
--
-- ackley' = ackley 20.0 0.2 (2*pi :: Float)
--
-- lossAckley :: Ackley -> Tensor
-- lossAckley (Ackley x) = ackley' x'
--     where x' = toDependent x
