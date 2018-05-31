{-# LANGUAGE ScopedTypeVariables #-}
module Torch.Indef.Static.Tensor.Random.TH where

import Control.Monad
import Torch.Indef.Types
import Torch.Dimensions
import Torch.Indef.Static.Tensor
import Torch.Indef.Static.Tensor.Math
import Torch.Indef.Static.Tensor.Math.Pointwise
import Torch.Indef.Static.Tensor.Math.Blas
import qualified Torch.Indef.Dynamic.Tensor.Random.TH as Dynamic
import qualified Torch.Types.TH as TH
import qualified Torch.Sig.Tensor.Random.TH as Sig
import qualified Torch.Sig.Types as Sig
import qualified Torch.Sig.Types.Global as Sig

_random :: Dimensions d => Tensor d -> Generator -> IO ()
_random r = Dynamic._random (asDynamic r)

_clampedRandom :: Dimensions d => Tensor d -> Generator -> Integer -> Integer -> IO ()
_clampedRandom r = Dynamic._clampedRandom (asDynamic r)

_cappedRandom :: Dimensions d => Tensor d -> Generator -> Integer -> IO ()
_cappedRandom r = Dynamic._cappedRandom (asDynamic r)

_geometric :: Dimensions d => Tensor d -> Generator -> HsAccReal -> IO ()
_geometric r = Dynamic._geometric (asDynamic r)

_bernoulli :: Dimensions d => Tensor d -> Generator -> HsAccReal -> IO ()
_bernoulli r = Dynamic._bernoulli (asDynamic r)

_bernoulli_FloatTensor :: Dimensions d => Tensor d -> Generator -> TH.FloatTensor d -> IO ()
_bernoulli_FloatTensor r g f = Dynamic._bernoulli_FloatTensor (asDynamic r) g (TH.floatAsDynamic f)

_bernoulli_DoubleTensor :: Dimensions d => Tensor d -> Generator -> TH.DoubleTensor d -> IO ()
_bernoulli_DoubleTensor r g d = Dynamic._bernoulli_DoubleTensor (asDynamic r) g (TH.doubleAsDynamic d)

_uniform :: Dimensions d => Tensor d -> Generator -> HsAccReal -> HsAccReal -> IO ()
_uniform r = Dynamic._uniform (asDynamic r)

_normal :: Dimensions d => Tensor d -> Generator -> HsAccReal -> HsAccReal -> IO ()
_normal r = Dynamic._normal (asDynamic r)

_normal_means :: Dimensions d => Tensor d -> Generator -> Tensor d -> HsAccReal -> IO ()
_normal_means r g a = Dynamic._normal_means (asDynamic r) g (asDynamic a)

_normal_stddevs :: Dimensions d => Tensor d -> Generator -> HsAccReal -> Tensor d -> IO ()
_normal_stddevs r g v a = Dynamic._normal_stddevs (asDynamic r) g v (asDynamic a)

_normal_means_stddevs :: Dimensions d => Tensor d -> Generator -> Tensor d -> Tensor d -> IO ()
_normal_means_stddevs r g a b = Dynamic._normal_means_stddevs (asDynamic r) g (asDynamic a) (asDynamic b)

_exponential :: Dimensions d => Tensor d -> Generator -> HsAccReal -> IO ()
_exponential r = Dynamic._exponential (asDynamic r)

_standard_gamma :: Dimensions d => Tensor d -> Generator -> Tensor d -> IO ()
_standard_gamma r g t = Dynamic._standard_gamma (asDynamic r) g (asDynamic t)

_cauchy :: Dimensions d => Tensor d -> Generator -> HsAccReal -> HsAccReal -> IO ()
_cauchy r = Dynamic._cauchy (asDynamic r)

_logNormal :: Dimensions d => Tensor d -> Generator -> HsAccReal -> HsAccReal -> IO ()
_logNormal r = Dynamic._logNormal (asDynamic r)

_multinomial :: Dimensions d => IndexTensor d -> Generator -> Tensor d -> Int -> Int -> IO ()
_multinomial r g t = Dynamic._multinomial (longAsDynamic r) g (asDynamic t)

_multinomialAliasSetup :: Dimensions d => Tensor d -> IndexTensor d -> Tensor d -> IO ()
_multinomialAliasSetup r g t = Dynamic._multinomialAliasSetup (asDynamic r) (longAsDynamic g) (asDynamic t)

_multinomialAliasDraw :: Dimensions d => IndexTensor d -> Generator -> IndexTensor d -> Tensor d -> IO ()
_multinomialAliasDraw r g a b = Dynamic._multinomialAliasDraw (longAsDynamic r) g (longAsDynamic a) (asDynamic b)

-- ========================================================================= --
-- Custom functions
-- ========================================================================= --
multivariate_normal
  :: forall n p . (KnownNatDim2 n p)
  => Generator -> Tensor '[p] -> Tensor '[p, p] -> Tensor '[p] -> IO (Tensor '[n, p])
multivariate_normal g mu eigvec eigval = join $ go
  <$> newTranspose2d eigvec
  <*> diag1d eigval
  <*> expand2d mu
  <*> normal g 0 1
 where
  go :: Tensor '[p, p] -> Tensor '[p, p] -> Tensor '[n, p] -> Tensor '[p, n] -> IO (Tensor '[n, p])
  go evec' eval' offset samps = (^+^ offset) <$> newTranspose2d (y !*! samps)
    where
      x = evec' !*! eval'
      y = x !*! eigvec

random :: (Dimensions d) => Generator -> IO (Tensor d)
random g = withEmpty (`_random` g)

clampedRandom :: (Dimensions d) => Generator -> Integer -> Integer -> IO (Tensor d)
clampedRandom g a b = withEmpty $ \t -> _clampedRandom t g a b

cappedRandom :: (Dimensions d) => Generator -> Integer -> IO (Tensor d)
cappedRandom g a = withEmpty $ \t -> _cappedRandom t g a

geometric :: (Dimensions d) => Generator -> HsAccReal -> IO (Tensor d)
geometric g a = withEmpty $ \t -> _geometric t g a

bernoulli :: (Dimensions d) => Generator -> HsAccReal -> IO (Tensor d)
bernoulli g a = withEmpty $ \t -> _bernoulli t g a

bernoulli_FloatTensor :: (Dimensions d) => Generator -> TH.FloatTensor d -> IO (Tensor d)
bernoulli_FloatTensor g a = withEmpty $ \t -> _bernoulli_FloatTensor t g a

bernoulli_DoubleTensor :: (Dimensions d) => Generator -> TH.DoubleTensor d -> IO (Tensor d)
bernoulli_DoubleTensor g a = withEmpty $ \t -> _bernoulli_DoubleTensor t g a

uniform :: (Dimensions d) => Generator -> HsAccReal -> HsAccReal -> IO (Tensor d)
uniform g a b = withEmpty $ \t -> _uniform t g a b

normal :: (Dimensions d) => Generator -> HsAccReal -> HsAccReal -> IO (Tensor d)
normal g a b = withEmpty $ \t -> _normal t g a b

normal_means :: (Dimensions d) => Generator -> Tensor d -> HsAccReal -> IO (Tensor d)
normal_means g m b = withEmpty $ \t -> _normal_means t g m b

normal_stddevs :: (Dimensions d) => Generator -> HsAccReal -> Tensor d -> IO (Tensor d)
normal_stddevs g a s = withEmpty $ \t -> _normal_stddevs t g a s

normal_means_stddevs :: (Dimensions d) => Generator -> Tensor d -> Tensor d -> IO (Tensor d)
normal_means_stddevs g m s = withEmpty $ \t -> _normal_means_stddevs t g m s

exponential :: (Dimensions d) => Generator -> HsAccReal -> IO (Tensor d)
exponential g a = withEmpty $ \t -> _exponential t g a

standard_gamma :: (Dimensions d) => Generator -> Tensor d -> IO (Tensor d)
standard_gamma g a = withEmpty $ \t -> _standard_gamma t g a

cauchy :: (Dimensions d) => Generator -> HsAccReal -> HsAccReal -> IO (Tensor d)
cauchy g a b = withEmpty $ \t -> _cauchy t g a b

logNormal :: (Dimensions d) => Generator -> HsAccReal -> HsAccReal -> IO (Tensor d)
logNormal g a b = withEmpty $ \t -> _logNormal t g a b

