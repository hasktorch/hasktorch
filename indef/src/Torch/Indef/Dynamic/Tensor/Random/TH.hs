-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Indef.Dynamic.Tensor.Random.TH
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
--
-- Torch provides accurate mathematical random generation, based on
-- <http://www.math.sci.hiroshima-u.ac.jp/%7Em-mat/MT/emt.html Mersenne Twister>
-- random number generator.
--
-- FIXME: verify these are all correct -- I am working off of
-- <https://github.com/torch/torch7/blob/aed31711c6b8846b8337a263a7f9f998697994e7/doc/random.md
-- random.md> from torch/torch7.
-------------------------------------------------------------------------------
module Torch.Indef.Dynamic.Tensor.Random.TH
  ( _random                 , random
  , _clampedRandom          , clampedRandom
  , _cappedRandom           , cappedRandom
  , _geometric              , geometric
  , _bernoulli              , bernoulli
  , _bernoulli_FloatTensor  , bernoulli_FloatTensor
  , _bernoulli_DoubleTensor , bernoulli_DoubleTensor
  , _uniform                , uniform
  , _normal                 , normal
  , _normal_means           , normal_means
  , _normal_stddevs         , normal_stddevs
  , _normal_means_stddevs   , normal_means_stddevs
  , _exponential            , exponential
  , _standard_gamma         , standard_gamma
  , _cauchy                 , cauchy
  , _logNormal              , logNormal
  , _multinomial
  , _multinomialAliasSetup
  , _multinomialAliasDraw

  , OpenUnit, openUnit, openUnitValue
  , ClosedUnit, closedUnit, closedUnitValue
  , Positive, positive, positiveValue
  , Ord2Tuple, ord2Tuple, ord2TupleValues
  ) where

import GHC.Word
import Numeric.Dimensions

import Torch.Indef.Types
import Torch.Indef.Dynamic.Tensor
import qualified Torch.Sig.Tensor.Random.TH as Sig
import qualified Torch.Sig.Types as Sig
import qualified Torch.Sig.Types.Global as Sig
import qualified Torch.Types.TH as TH

-- | Returns a tensor of specified size with random numbers from [1,2^mantissa].
random :: Dims (d::[Nat]) -> Generator -> IO Dynamic
random d g = withInplace (`_random` g) d

-- | Returns a tensor of specified size with random numbers from @[minBound,maxBound]@.
clampedRandom
  :: Dims (d::[Nat])   -- ^ size of tensor to create
  -> Generator         -- ^ generator
  -> Ord2Tuple Integer -- ^ (minbound, maxBound)
  -> IO Dynamic
clampedRandom d g bs = flip withInplace d $ \t -> _clampedRandom t g bs

-- | Returns a tensor of specified size with random numbers from @[0,maxBound]@.
cappedRandom
  :: Dims (d::[Nat])  -- ^ size of tensor to create
  -> Generator        -- ^ generator
  -> Word64           -- ^ maxbound
  -> IO Dynamic
cappedRandom d g a = flip withInplace d $ \t -> _cappedRandom t g a

-- | Returns a random tensor according to a geometric distribution
-- @p(i) = (1-p) * p^(i-1)@. @p@ must satisfy @0 < p < 1@.
geometric
  :: Dims (d::[Nat])      -- ^ size of tensor to create
  -> Generator            -- ^ generator
  -> OpenUnit HsAccReal   -- ^ @p@, where @0 < p < 1@
  -> IO Dynamic
geometric d g p = flip withInplace d $ \t -> _geometric t g (openUnitValue p)

-- | Returns a tensor filled with elements which are 1 with probability @p@ and
-- 0 with probability @1-p@. @p@ must satisfy @0 <= p <= 1@.
bernoulli
  :: Dims (d::[Nat])      -- ^ size of tensor to create
  -> Generator            -- ^ generator
  -> ClosedUnit HsAccReal -- ^ @p@, where @0 <= p <= 1@
  -> IO Dynamic
bernoulli d g a = flip withInplace d $ \t -> _bernoulli t g (closedUnitValue a)

-- | Undocumented in Lua. It is assumed that this returns a tensor filled with
-- elements which are 1 with probability @p@ and 0 with probability @1-p@,
-- where @p@ comes from the 'TH.FloatDynamic' tensor. All @p@s must satisfy
-- @0 <= p <= 1@. It is uncertain if the output dimensions and the
-- 'TH.FloatDynamic' tensor dimensions need to match.
bernoulli_FloatTensor
  :: Dims (d::[Nat])      -- ^ size of tensor to create
  -> Generator            -- ^ generator
  -> TH.FloatDynamic      -- ^ float tensor of @p@ values, where @0 <= p <= 1@
  -> IO Dynamic
bernoulli_FloatTensor d g a = flip withInplace d $ \t -> _bernoulli_FloatTensor t g a

-- | Undocumented in Lua. It is assumed that this returns a tensor filled with
-- elements which are 1 with probability @p@ and 0 with probability @1-p@,
-- where @p@ comes from the 'TH.DoubleDynamic' tensor. All @p@s must satisfy
-- @0 <= p <= 1@. It is uncertain if the output dimensions and the
-- 'TH.DoubleDynamic' tensor dimensions need to match.
bernoulli_DoubleTensor
  :: Dims (d::[Nat])      -- ^ size of tensor to create
  -> Generator            -- ^ generator
  -> TH.DoubleDynamic     -- ^ double tensor of @p@ values, where @0 <= p <= 1@
  -> IO Dynamic
bernoulli_DoubleTensor d g a = flip withInplace d $ \t -> _bernoulli_DoubleTensor t g a

-- | Returns a tensor filled with values according to uniform distribution on @[a,b)@.
uniform
  :: Dims (d::[Nat])      -- ^ size of tensor to create
  -> Generator            -- ^ generator
  -> Ord2Tuple HsAccReal  -- ^ tuple of @(a, b)@ representing the @[a,b)@ interval.
  -> IO Dynamic
uniform d g tup = flip withInplace d $ \t -> _uniform t g tup

-- | Returns a tensor filled with values according to a normal distribution with
-- the given mean and standard deviation stdv. stdv must be positive.
normal
  :: Dims (d::[Nat])      -- ^ size of tensor to create
  -> Generator            -- ^ generator
  -> HsAccReal            -- ^ mean
  -> Positive HsAccReal   -- ^ standard deviation.
  -> IO Dynamic
normal d g a b = flip withInplace d $ \t -> _normal t g a b

-- | Same as 'normal', taking a tensor of means to use instead of a scalar.
--
-- FIXME: It is uncertain if the output dimensions and mean tensor dimensions
-- need to match.
normal_means
  :: Dims (d::[Nat])      -- ^ size of tensor to create
  -> Generator            -- ^ generator
  -> Dynamic              -- ^ tensor of means
  -> Positive HsAccReal   -- ^ standard deviation.
  -> IO Dynamic
normal_means d g m b = flip withInplace d $ \t -> _normal_means t g m b

-- | Same as 'normal', taking a tensor of standard deviations to use instead of
-- a scalar. All standard deviations must be positive.
--
-- FIXME: It is uncertain if the output dimensions and stddv tensor dimensions
-- need to match.
normal_stddevs
  :: Dims (d::[Nat])      -- ^ size of tensor to create
  -> Generator            -- ^ generator
  -> HsAccReal            -- ^ mean
  -> Dynamic              -- ^ tensor of standard deviations
  -> IO Dynamic
normal_stddevs d g a s = flip withInplace d $ \t -> _normal_stddevs t g a s

-- | Same as 'normal', taking a tensor of standard deviations and tensor of means
-- to use instead of a scalar values. All standard deviations must be positive.
--
-- FIXME: It is uncertain if all of the tensor dimensions need to match.
normal_means_stddevs
  :: Dims (d::[Nat])      -- ^ size of tensor to create
  -> Generator            -- ^ generator
  -> Dynamic              -- ^ tensor of means
  -> Dynamic              -- ^ tensor of standard deviations
  -> IO Dynamic
normal_means_stddevs d g m s = flip withInplace d $ \t -> _normal_means_stddevs t g m s

-- | Returns a tensor filled with values according to the exponential distribution
-- @p(x) = lambda * exp(-lambda * x)@.
exponential
  :: Dims (d::[Nat])      -- ^ size of tensor to create
  -> Generator            -- ^ generator
  -> HsAccReal            -- ^ lambda
  -> IO Dynamic
exponential d g a = flip withInplace d $ \t -> _exponential t g a

-- |  Draw samples from a standard Gamma distribution.
--
-- PyTorch's @standard_gamma@ function mostly references Numpy's. Documentation can be found here:
-- <https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.standard_gamma.html numpy.random.standard_gamma>
--
-- I (stites) am not sure at the moment if the tensor argument is a tensor of
-- parameters (in which case, can we replace it with something safer?), or a datasource.
--
-- FIXME(stites): This is an undocumented feature as far as I can tell. Someone
-- should update this with a more thorough investigation.
standard_gamma
  :: Dims (d::[Nat])      -- ^ size of tensor to create
  -> Generator            -- ^ generator
  -> Dynamic
  -> IO Dynamic
standard_gamma d g a = flip withInplace d $ \t -> _standard_gamma t g a

-- | Returns a tensor filled with values according to the Cauchy distribution
-- @p(x) = sigma/(pi*(sigma^2 + (x-median)^2))@
cauchy
  :: Dims (d::[Nat])      -- ^ size of tensor to create
  -> Generator            -- ^ generator
  -> HsAccReal            -- ^ median
  -> HsAccReal            -- ^ sigma
  -> IO Dynamic
cauchy d g a b = flip withInplace d $ \t -> _cauchy t g a b

-- | Returns a tensor filled with values according to the log-normal
-- distribution with the given mean and standard deviation stdv. mean and stdv
-- are the corresponding mean and standard deviation of the underlying normal
-- distribution, and not of the returned distribution.
--
-- stdv must be positive.
logNormal
  :: Dims (d::[Nat])      -- ^ size of tensor to create
  -> Generator            -- ^ generator
  -> HsAccReal            -- ^ mean
  -> Positive HsAccReal   -- ^ standard deviation.
  -> IO Dynamic
logNormal d g a b = flip withInplace d $ \t -> _logNormal t g a b

-- | call C-level @random@
_random t g = tenGen t g Sig.c_random

-- | call C-level @clampedRandom@
_clampedRandom r g tup =
  tenGen r g $ shuffle3'2 Sig.c_clampedRandom (fromIntegral a) (fromIntegral b)
  where
    (a, b) = ord2TupleValues tup

-- | call C-level @cappedRandom@
_cappedRandom r g a = tenGen r g $ shuffle3 Sig.c_cappedRandom (fromIntegral a)

-- | call C-level @geometric@
_geometric r g a = tenGen r g $ shuffle3 Sig.c_geometric (hs2cAccReal a)

-- | call C-level @bernoulli@
_bernoulli r g a = tenGen r g $ shuffle3 Sig.c_bernoulli (hs2cAccReal a)

-- | call C-level @bernoulli_FloatTensor@
_bernoulli_FloatTensor r g a = tenGenTen r g (snd $ TH.floatDynamicState a) Sig.c_bernoulli_FloatTensor

-- | call C-level @bernoulli_DoubleTensor@
_bernoulli_DoubleTensor r g a = tenGenTen r g (snd $ TH.doubleDynamicState a) Sig.c_bernoulli_DoubleTensor

-- | call C-level @uniform@
_uniform :: Dynamic -> Generator -> Ord2Tuple HsAccReal -> IO ()
_uniform r g tup = tenGen r g $ shuffle3'2 Sig.c_uniform (hs2cAccReal a) (hs2cAccReal b)
  where
    (a, b) = ord2TupleValues tup

-- | call C-level @normal@
_normal :: Dynamic -> Generator -> HsAccReal -> Positive HsAccReal -> IO ()
_normal r g a b = tenGen r g $ shuffle3'2 Sig.c_normal (hs2cAccReal a) (hs2cAccReal $ positiveValue b)

-- | call C-level @normal_means@
_normal_means :: Dynamic -> Generator -> Dynamic -> Positive HsAccReal -> IO ()
_normal_means r g m v = runManaged . joinIO $ Sig.c_normal_means
  <$> manage' Sig.dynamicStateRef r
  <*> manage' Sig.ctensor r
  <*> manage' Sig.rng g
  <*> manage' Sig.ctensor m
  <*> pure (Sig.hs2cAccReal $ positiveValue v)

-- | call C-level @normal_stddevs@
_normal_stddevs :: Dynamic -> Generator -> HsAccReal -> Dynamic -> IO ()
_normal_stddevs r g v m = runManaged . joinIO $ Sig.c_normal_stddevs
  <$> manage' Sig.dynamicStateRef r
  <*> manage' Sig.ctensor r
  <*> manage' Sig.rng g
  <*> pure (Sig.hs2cAccReal v)
  <*> manage' Sig.ctensor m

-- | call C-level @normal_means_stddevs@
_normal_means_stddevs :: Dynamic -> Generator -> Dynamic -> Dynamic -> IO ()
_normal_means_stddevs r g a b = runManaged . joinIO $ Sig.c_normal_means_stddevs
  <$> manage' Sig.dynamicStateRef r
  <*> manage' Sig.ctensor r
  <*> manage' Sig.rng g
  <*> manage' Sig.ctensor a
  <*> manage' Sig.ctensor b

-- | call C-level @exponential@
_exponential :: Dynamic -> Generator -> HsAccReal -> IO ()
_exponential r g v = runManaged . joinIO $ Sig.c_exponential
  <$> manage' Sig.dynamicStateRef r
  <*> manage' Sig.ctensor r
  <*> manage' Sig.rng g
  <*> pure (Sig.hs2cAccReal v)

-- | call C-level @standard_gamma@
_standard_gamma :: Dynamic -> Generator -> Dynamic -> IO ()
_standard_gamma r g m = runManaged . joinIO $ Sig.c_standard_gamma
  <$> manage' Sig.dynamicStateRef r
  <*> manage' Sig.ctensor r
  <*> manage' Sig.rng g
  <*> manage' Sig.ctensor m

-- | call C-level @cauchy@
_cauchy :: Dynamic -> Generator -> HsAccReal -> HsAccReal -> IO ()
_cauchy r g a b = runManaged . joinIO $ Sig.c_cauchy
  <$> manage' Sig.dynamicStateRef r
  <*> manage' Sig.ctensor r
  <*> manage' Sig.rng g
  <*> pure (Sig.hs2cAccReal a)
  <*> pure (Sig.hs2cAccReal b)

-- | call C-level @logNormal@
_logNormal :: Dynamic -> Generator -> HsAccReal -> Positive HsAccReal -> IO ()
_logNormal r g a b = runManaged . joinIO $ Sig.c_logNormal
  <$> manage' Sig.dynamicStateRef r
  <*> manage' Sig.ctensor r
  <*> manage' Sig.rng g
  <*> pure (Sig.hs2cAccReal a)
  <*> pure (Sig.hs2cAccReal $ positiveValue b)

-- | call C-level @multinomial@
_multinomial :: LongDynamic -> Generator -> Dynamic -> Int -> Int -> IO ()
_multinomial r g t a b = runManaged . joinIO $ Sig.c_multinomial
  <$> manage' (fst . Sig.longDynamicState) r
  <*> manage' (snd . Sig.longDynamicState) r
  <*> manage' Sig.rng g
  <*> manage' Sig.ctensor t
  <*> pure (fromIntegral a)
  <*> pure (fromIntegral b)

-- | call C-level @multinomialAliasSetup@
_multinomialAliasSetup :: Dynamic -> LongDynamic -> Dynamic -> IO ()
_multinomialAliasSetup r l t = runManaged . joinIO $ Sig.c_multinomialAliasSetup
  <$> manage' (Sig.dynamicStateRef) r
  <*> manage' (Sig.ctensor) r
  <*> manage' (snd . Sig.longDynamicState) l
  <*> manage' (Sig.ctensor) t

-- | call C-level @multinomialAliasDraw@
_multinomialAliasDraw  :: LongDynamic -> Generator -> LongDynamic -> Dynamic -> IO ()
_multinomialAliasDraw r g l t = runManaged . joinIO $ Sig.c_multinomialAliasDraw
  <$> manage' (fst . Sig.longDynamicState) r
  <*> manage' (snd . Sig.longDynamicState) r
  <*> manage' (Sig.rng) g
  <*> manage' (snd . Sig.longDynamicState) l
  <*> manage' (Sig.ctensor) t

-- ========================================================================= --
-- * ADTs to manage runtime boundaries

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


-- ========================================================================= --
-- helper functions

tenGen
  :: Dynamic
  -> Generator
  -> (Ptr CState -> Ptr CTensor -> Ptr CGenerator -> IO x)
  -> IO x
tenGen r g fn =
  withDynamicState r $ \s' r' ->
    withGen g (fn s' r')

tenGenTen
  :: Dynamic
  -> Generator
  -> ForeignPtr a
  -> (Ptr CState -> Ptr CTensor -> Ptr CGenerator -> Ptr a -> IO x)
  -> IO x
tenGenTen r g t fn = tenGen r g $ \s' r' g' ->
    withForeignPtr t (fn s' r' g')

