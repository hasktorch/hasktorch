{-# LANGUAGE DataKinds #-}

module Torch.Distributions.Constraints (
    Constraint,
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
) where

import qualified Torch.Functional.Internal as I
import qualified Torch.Tensor as D
import qualified Torch.TensorFactories as D
import qualified Torch.Functional as F
import Torch.Scalar

type Constraint = D.Tensor -> D.Tensor

dependent :: Constraint
dependent _tensor = error "Cannot determine validity of dependent constraint"

boolean :: Constraint
boolean tensor = (tensor `F.eq` d_zerosLike tensor) `logical_or` (tensor `F.eq` D.onesLike tensor)
    where
        d_zerosLike = fullLike' (0.0 :: Float)
        logical_or = F.add

integerInterval :: Int -> Int -> Constraint
integerInterval lower_bound upper_bound tensor = (tensor `F.ge` fullLike' lower_bound tensor) `logical_and` (tensor `F.le` fullLike' upper_bound tensor)
        where logical_and = F.mul

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
interval lower_bound upper_bound tensor = (tensor `F.ge` fullLike' lower_bound tensor) `logical_and` (tensor `F.le` fullLike' upper_bound tensor)
        where logical_and = F.mul

halfOpenInterval :: Float -> Float -> Constraint
halfOpenInterval lower_bound upper_bound tensor = (tensor `F.ge` fullLike' lower_bound tensor) `logical_and` (tensor `F.lt` fullLike' upper_bound tensor)
        where logical_and = F.mul

simplex :: Constraint
simplex tensor = F.allDim (greaterThanEq 0.0 tensor) (-1) False `logical_and` (lessThan 1e-6 $ F.abs $ summed `F.sub` D.onesLike summed)
        where
            logical_and = F.mul
            summed = F.sumDim (-1) tensor

-- TODO: lowerTriangular
-- TODO: lowerCholesky
-- TODO: positiveDefinite
-- TODO: realVector
-- TODO: cat
-- TODO: stack

nonNegativeInteger :: Constraint
nonNegativeInteger = integerGreaterThanEq 0
positiveInteger    :: Constraint
positiveInteger    = integerGreaterThanEq 1
positive           :: Constraint
positive           = greaterThan 0.0
unitInterval       :: Constraint
unitInterval       = interval 0.0 1.0

fullLike' :: (Scalar a) => a -> D.Tensor -> D.Tensor
fullLike' i t = F.mulScalar (D.onesLike t) i
