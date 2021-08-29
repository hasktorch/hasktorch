{-# LANGUAGE DataKinds #-}

module Torch.Distributions.Constraints
  ( Constraint,
    dependent,
    boolean,
    integerInterval,
    integerLessThan,
    integerGreaterThan,
    integerLessThanEq,
    integerGreaterThanEq,
    real,
    greaterThan,
    greaterThanEq,
    lessThan,
    lessThanEq,
    interval,
    halfOpenInterval,
    simplex,
    nonNegativeInteger,
    positiveInteger,
    positive,
    unitInterval,
  )
where

import qualified Torch.Functional as F
import qualified Torch.Functional.Internal as I
import Torch.Scalar
import qualified Torch.Tensor as D
import qualified Torch.TensorFactories as D

type Constraint = D.Tensor -> D.Tensor

dependent :: Constraint
dependent _tensor = error "Cannot determine validity of dependent constraint"

boolean :: Constraint
boolean tensor = (tensor `F.eq` D.zerosLike tensor) `I.logical_or` (tensor `F.eq` D.onesLike tensor)

integerInterval :: Int -> Int -> Constraint
integerInterval lower_bound upper_bound tensor = (tensor `F.ge` fullLike' lower_bound tensor) `I.logical_and` (tensor `F.le` fullLike' upper_bound tensor)

integerLessThan :: Int -> Constraint
integerLessThan upper_bound tensor = tensor `F.lt` fullLike' upper_bound tensor

integerGreaterThan :: Int -> Constraint
integerGreaterThan lower_bound tensor = tensor `F.gt` fullLike' lower_bound tensor

integerLessThanEq :: Int -> Constraint
integerLessThanEq upper_bound tensor = tensor `F.le` fullLike' upper_bound tensor

integerGreaterThanEq :: Int -> Constraint
integerGreaterThanEq lower_bound tensor = tensor `F.ge` fullLike' lower_bound tensor

real :: Constraint
real = I.isfinite

greaterThan :: Float -> Constraint
greaterThan lower_bound tensor = tensor `F.gt` fullLike' lower_bound tensor

greaterThanEq :: Float -> Constraint
greaterThanEq lower_bound tensor = tensor `F.ge` fullLike' lower_bound tensor

lessThan :: Float -> Constraint
lessThan upper_bound tensor = tensor `F.lt` fullLike' upper_bound tensor

lessThanEq :: Float -> Constraint
lessThanEq upper_bound tensor = tensor `F.le` fullLike' upper_bound tensor

interval :: Float -> Float -> Constraint
interval lower_bound upper_bound tensor = (tensor `F.ge` fullLike' lower_bound tensor) `I.logical_and` (tensor `F.le` fullLike' upper_bound tensor)

halfOpenInterval :: Float -> Float -> Constraint
halfOpenInterval lower_bound upper_bound tensor = (tensor `F.ge` fullLike' lower_bound tensor) `I.logical_and` (tensor `F.lt` fullLike' upper_bound tensor)

simplex :: Constraint
simplex tensor = F.allDim (F.Dim $ -1) False (greaterThanEq 0.0 tensor) `I.logical_and` (lessThan 1e-6 $ F.abs $ summed `F.sub` D.onesLike summed)
  where
    summed = F.sumDim (F.Dim $ -1) F.RemoveDim (D.dtype tensor) tensor

-- TODO: lowerTriangular
-- TODO: lowerCholesky
-- TODO: positiveDefinite
-- TODO: realVector
-- TODO: cat
-- TODO: stack

nonNegativeInteger :: Constraint
nonNegativeInteger = integerGreaterThanEq 0

positiveInteger :: Constraint
positiveInteger = integerGreaterThanEq 1

positive :: Constraint
positive = greaterThan 0.0

unitInterval :: Constraint
unitInterval = interval 0.0 1.0

fullLike' :: (Scalar a) => a -> D.Tensor -> D.Tensor
fullLike' i t = F.mulScalar i $ D.onesLike t
