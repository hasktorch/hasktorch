-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Indef.Static.Tensor.Random.TH
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
-------------------------------------------------------------------------------
{-# LANGUAGE ScopedTypeVariables #-}
module Torch.Indef.Static.Tensor.Random.TH
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

  , OpenUnit   , Dynamic.openUnit   , Dynamic.openUnitValue
  , ClosedUnit , Dynamic.closedUnit , Dynamic.closedUnitValue
  , Positive   , Dynamic.positive   , Dynamic.positiveValue
  , Ord2Tuple  , Dynamic.ord2Tuple  , Dynamic.ord2TupleValues

  , multivariate_normal
  ) where


import Numeric.Dimensions
import Control.Monad
import GHC.Word
import Torch.Indef.Types
import Torch.Indef.Static.Tensor
import Torch.Indef.Static.Tensor.Math
import Torch.Indef.Static.Tensor.Math.Pointwise
import Torch.Indef.Static.Tensor.Math.Blas
import Torch.Indef.Dynamic.Tensor.Random.TH (Ord2Tuple, Positive, ClosedUnit, OpenUnit)
import qualified Torch.Indef.Dynamic.Tensor.Random.TH as Dynamic
import qualified Torch.Types.TH as TH
import qualified Torch.Sig.Tensor.Random.TH as Sig
import qualified Torch.Sig.Types as Sig
import qualified Torch.Sig.Types.Global as Sig

-- | Static call to 'Dynamic._random'
_random :: Dimensions d => Tensor d -> Generator -> IO ()
_random r = Dynamic._random (asDynamic r)

-- | Static call to 'Dynamic._clampedRandom'
_clampedRandom :: Dimensions d => Tensor d -> Generator -> Ord2Tuple Integer -> IO ()
_clampedRandom r = Dynamic._clampedRandom (asDynamic r)

-- | Static call to 'Dynamic._cappedRandom'
_cappedRandom :: Dimensions d => Tensor d -> Generator -> Integer -> IO ()
_cappedRandom r = Dynamic._cappedRandom (asDynamic r)

-- | Static call to 'Dynamic._geometric'
_geometric :: Dimensions d => Tensor d -> Generator -> HsAccReal -> IO ()
_geometric r = Dynamic._geometric (asDynamic r)

-- | Static call to 'Dynamic._bernoulli'
_bernoulli :: Dimensions d => Tensor d -> Generator -> HsAccReal -> IO ()
_bernoulli r = Dynamic._bernoulli (asDynamic r)

-- | Static call to 'Dynamic._bernoulli_FloatTensor'
_bernoulli_FloatTensor :: Dimensions d => Tensor d -> Generator -> TH.FloatTensor d -> IO ()
_bernoulli_FloatTensor r g f = Dynamic._bernoulli_FloatTensor (asDynamic r) g (TH.floatAsDynamic f)

-- | Static call to 'Dynamic._bernoulli_DoubleTensor'
_bernoulli_DoubleTensor :: Dimensions d => Tensor d -> Generator -> TH.DoubleTensor d -> IO ()
_bernoulli_DoubleTensor r g d = Dynamic._bernoulli_DoubleTensor (asDynamic r) g (TH.doubleAsDynamic d)

-- | Static call to 'Dynamic._uniform'
_uniform :: Dimensions d => Tensor d -> Generator -> Ord2Tuple HsAccReal -> IO ()
_uniform r = Dynamic._uniform (asDynamic r)

-- | Static call to 'Dynamic._normal'
_normal :: Dimensions d => Tensor d -> Generator -> HsAccReal -> Positive HsAccReal -> IO ()
_normal r = Dynamic._normal (asDynamic r)

-- | Static call to 'Dynamic._normal_means'
_normal_means :: Dimensions d => Tensor d -> Generator -> Tensor d -> Positive HsAccReal -> IO ()
_normal_means r g a = Dynamic._normal_means (asDynamic r) g (asDynamic a)

-- | Static call to 'Dynamic._normal_stddevs'
_normal_stddevs :: Dimensions d => Tensor d -> Generator -> HsAccReal -> Tensor d -> IO ()
_normal_stddevs r g v a = Dynamic._normal_stddevs (asDynamic r) g v (asDynamic a)

-- | Static call to 'Dynamic._normal_means_stddevs'
_normal_means_stddevs :: Dimensions d => Tensor d -> Generator -> Tensor d -> Tensor d -> IO ()
_normal_means_stddevs r g a b = Dynamic._normal_means_stddevs (asDynamic r) g (asDynamic a) (asDynamic b)

-- | Static call to 'Dynamic._exponential'
_exponential :: Dimensions d => Tensor d -> Generator -> HsAccReal -> IO ()
_exponential r = Dynamic._exponential (asDynamic r)

-- | Static call to 'Dynamic._standard_gamma'
_standard_gamma :: Dimensions d => Tensor d -> Generator -> Tensor d -> IO ()
_standard_gamma r g t = Dynamic._standard_gamma (asDynamic r) g (asDynamic t)

-- | Static call to 'Dynamic._cauchy'
_cauchy :: Dimensions d => Tensor d -> Generator -> HsAccReal -> HsAccReal -> IO ()
_cauchy r = Dynamic._cauchy (asDynamic r)

-- | Static call to 'Dynamic._logNormal'
_logNormal :: Dimensions d => Tensor d -> Generator -> HsAccReal -> Positive HsAccReal -> IO ()
_logNormal r = Dynamic._logNormal (asDynamic r)

-- | Static call to 'Dynamic._multinomial'
_multinomial :: Dimensions d => IndexTensor d -> Generator -> Tensor d -> Int -> Int -> IO ()
_multinomial r g t = Dynamic._multinomial (longAsDynamic r) g (asDynamic t)

-- | Static call to 'Dynamic._multinomialAliasSetup'
_multinomialAliasSetup :: Dimensions d => Tensor d -> IndexTensor d -> Tensor d -> IO ()
_multinomialAliasSetup r g t = Dynamic._multinomialAliasSetup (asDynamic r) (longAsDynamic g) (asDynamic t)

-- | Static call to 'Dynamic._multinomialAliasDraw'
_multinomialAliasDraw :: Dimensions d => IndexTensor d -> Generator -> IndexTensor d -> Tensor d -> IO ()
_multinomialAliasDraw r g a b = Dynamic._multinomialAliasDraw (longAsDynamic r) g (longAsDynamic a) (asDynamic b)

-- | Static call to 'Dynamic.random'
random :: forall d . Dimensions d => Generator -> IO (Tensor d)
random g = asStatic <$> Dynamic.random (dims :: Dims d) g

-- | Static call to 'Dynamic.clampedRandom'
clampedRandom :: forall d . (Dimensions d) => Generator -> Ord2Tuple Integer -> IO (Tensor d)
clampedRandom g a = asStatic <$> Dynamic.clampedRandom (dims :: Dims d) g a

-- | Static call to 'Dynamic.cappedRandom'
cappedRandom :: forall d . (Dimensions d) => Generator -> Word64 -> IO (Tensor d)
cappedRandom g a = asStatic <$> Dynamic.cappedRandom (dims :: Dims d) g a

-- | Static call to 'Dynamic.geometric'
geometric :: forall d . (Dimensions d) => Generator -> OpenUnit HsAccReal -> IO (Tensor d)
geometric g a = asStatic <$> Dynamic.geometric (dims :: Dims d) g a

-- | Static call to 'Dynamic.bernoulli'
bernoulli :: forall d . (Dimensions d) => Generator -> ClosedUnit HsAccReal -> IO (Tensor d)
bernoulli g a = asStatic <$> Dynamic.bernoulli (dims :: Dims d) g a

-- | Static call to 'Dynamic.bernoulli_FloatTensor'
bernoulli_FloatTensor :: forall d . (Dimensions d) => Generator -> TH.FloatTensor d -> IO (Tensor d)
bernoulli_FloatTensor g a = asStatic <$> Dynamic.bernoulli_FloatTensor (dims :: Dims d) g (TH.floatAsDynamic a)

-- | Static call to 'Dynamic.bernoulli_DoubleTensor'
bernoulli_DoubleTensor :: forall d . (Dimensions d) => Generator -> TH.DoubleTensor d -> IO (Tensor d)
bernoulli_DoubleTensor g a = asStatic <$> Dynamic.bernoulli_DoubleTensor (dims :: Dims d) g (TH.doubleAsDynamic a)

-- | Static call to 'Dynamic.uniform'
uniform :: forall d . (Dimensions d) => Generator -> Ord2Tuple HsAccReal -> IO (Tensor d)
uniform g a = asStatic <$> Dynamic.uniform (dims :: Dims d) g a

-- | Static call to 'Dynamic.normal'
normal :: forall d . (Dimensions d) => Generator -> HsAccReal -> Positive HsAccReal -> IO (Tensor d)
normal g a b = asStatic <$> Dynamic.normal (dims :: Dims d) g a b

-- | Static call to 'Dynamic.normal_means'
normal_means :: forall d . (Dimensions d) => Generator -> Tensor d -> Positive HsAccReal -> IO (Tensor d)
normal_means g m b = asStatic <$> Dynamic.normal_means (dims :: Dims d) g (asDynamic m) b

-- | Static call to 'Dynamic.normal_stddevs'
normal_stddevs :: forall d . (Dimensions d) => Generator -> HsAccReal -> Tensor d -> IO (Tensor d)
normal_stddevs g a s = asStatic <$> Dynamic.normal_stddevs (dims :: Dims d) g a (asDynamic s)

-- | Static call to 'Dynamic.normal_means_stddevs'
normal_means_stddevs :: forall d . (Dimensions d) => Generator -> Tensor d -> Tensor d -> IO (Tensor d)
normal_means_stddevs g m s = asStatic <$> Dynamic.normal_means_stddevs (dims :: Dims d) g (asDynamic m) (asDynamic s)

-- | Static call to 'Dynamic.exponential'
exponential :: forall d . (Dimensions d) => Generator -> HsAccReal -> IO (Tensor d)
exponential g a = asStatic <$> Dynamic.exponential (dims :: Dims d) g a

-- | Static call to 'Dynamic.standard_gamma'
standard_gamma :: forall d . (Dimensions d) => Generator -> Tensor d -> IO (Tensor d)
standard_gamma g a = asStatic <$> Dynamic.standard_gamma (dims :: Dims d) g (asDynamic a)

-- | Static call to 'Dynamic.cauchy'
cauchy :: forall d . (Dimensions d) => Generator -> HsAccReal -> HsAccReal -> IO (Tensor d)
cauchy g a b = asStatic <$> Dynamic.cauchy (dims :: Dims d) g a b

-- | Static call to 'Dynamic.logNormal'
logNormal :: forall d . (Dimensions d) => Generator -> HsAccReal -> Positive HsAccReal -> IO (Tensor d)
logNormal g a b = asStatic <$> Dynamic.logNormal (dims :: Dims d) g a b

-- ========================================================================= --
-- * Custom functions

-- | find the multivariate normal distribution given @mu@, an @eigenvector@ and @eigenvalues@.
multivariate_normal
  :: forall n p . (All KnownDim '[n,p])
  => Generator
  -> Tensor '[p]       -- ^ mu
  -> Tensor '[p, p]    -- ^ eigenvec
  -> Tensor '[p]       -- ^ eigenval
  -> IO (Tensor '[n, p])
multivariate_normal g mu eigvec eigval = go (transpose2d eigvec)
  <$> diag1d eigval
  <*> pure (expand2d mu)
  <*> normal g 0 p1
 where
  Just p1 = Dynamic.positive 1

  go :: Tensor '[p, p] -> Tensor '[p, p] -> Tensor '[n, p] -> Tensor '[p, n] -> Tensor '[n, p]
  go evec' eval' offset samps = transpose2d (y !*! samps) ^+^ offset
    where
      x = evec' !*! eval'
      y = x !*! eigvec


