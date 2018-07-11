-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Types.Numeric
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
--
-- ADTs to manage runtime boundaries
-------------------------------------------------------------------------------
module Torch.Types.Numeric where


-- | Datatype to represent the open unit interval: @0 < x < 1@. Any 'OpenUnit' inhabitant
-- must satisfy being in the interval.
--
-- FIXME: replace with numhask.
newtype OpenUnit x = OpenUnit x
  deriving (Eq, Ord, Show)

-- | Get a value from the open unit interval.
openUnitValue :: OpenUnit x -> x
openUnitValue (OpenUnit x) = x

-- | smart constructor to place a number in the open unit interval.
openUnit :: (Ord x, Num x) => x -> Maybe (OpenUnit x)
openUnit x
  | x >= 1 || x <= 0 = Nothing
  | otherwise = Just (OpenUnit x)

-- | Datatype to represent the closed unit interval: @0 =< x =< 1@. Any 'ClosedUnit'
-- inhabitant must satisfy being in the interval.
--
-- FIXME: replace with numhask.
newtype ClosedUnit x = ClosedUnit x
  deriving (Eq, Ord, Show)

-- | Get a value from the closed unit interval.
closedUnitValue :: ClosedUnit x -> x
closedUnitValue (ClosedUnit x) = x

-- | smart constructor to place a number in the closed unit interval.
closedUnit :: (Ord x, Num x) => x -> Maybe (ClosedUnit x)
closedUnit x
  | x >= 1 || x <= 0 = Nothing
  | otherwise = Just (ClosedUnit x)

-- | Datatype to represent an ordered pair of numbers, @(a, b)@, where @a <= b@.
--
-- FIXME: replace with numhask.
newtype Ord2Tuple x = Ord2Tuple (x, x)
  deriving (Eq, Show)

-- | Get the values of an ordered tuple.
ord2TupleValues :: Ord2Tuple x -> (x, x)
ord2TupleValues (Ord2Tuple x) = x

-- | smart constructor to place two values in an ordered tuple.
ord2Tuple :: (Ord x, Num x) => (x, x) -> Maybe (Ord2Tuple x)
ord2Tuple x@(a, b)
  | a <= b = Just (Ord2Tuple x)
  | otherwise = Nothing

-- | Datatype to represent a generic positive number: @0 =< x@.
--
-- FIXME: replace with numhask.
newtype Positive x = Positive x
  deriving (Eq, Ord, Show)

-- | Get a value from the positive bound.
positiveValue :: Positive x -> x
positiveValue (Positive x) = x

-- | smart constructor to place a number in a positive bound.
positive :: (Ord x, Num x) => x -> Maybe (Positive x)
positive x
  | x < 0 = Nothing
  | otherwise = Just (Positive x)


-- | Datatype to represent a number that is not zero: @0 /= x@.
--
-- FIXME: replace with numhask.
newtype NonZero x = NonZero x
  deriving (Eq, Ord, Show)

-- | Get a value from the positive bound.
nonZeroValue :: NonZero x -> x
nonZeroValue (NonZero x) = x

-- | smart constructor to place a number in a positive bound.
nonZero :: (Ord x, Num x) => x -> Maybe (NonZero x)
nonZero x
  | x < 0 = Nothing
  | otherwise = Just (NonZero x)


