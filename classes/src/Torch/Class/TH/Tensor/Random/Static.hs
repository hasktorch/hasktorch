{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE FlexibleContexts #-}
module Torch.Class.TH.Tensor.Random.Static where

import Control.Monad
import Torch.Class.Types
import Torch.Dimensions
import Torch.Class.Tensor.Static
import Torch.Class.Tensor.Math.Static
import Torch.Class.Tensor.Math.Pointwise.Static
import Torch.Class.Tensor.Math.Blas.Static
import qualified Torch.Types.TH as TH

-- ========================================================================= --
-- Custom functions
-- ========================================================================= --
multivariate_normal
  :: forall t n p . (KnownNatDim2 n p)
  => (Num (HsReal (t '[p, p])))
  => (Num (HsReal (t '[p, n])))
  => (Num (HsReal (t '[n, p])))
  => (Num (HsAccReal (t '[p, n])))
  => TensorMathPointwise t
  => (IsTensor t, THTensorRandom t, TensorMathBlas t)
  => Generator (t '[p, n]) -> t '[p] -> t '[p, p] -> t '[p] -> IO (t '[n, p])
multivariate_normal g mu eigvec eigval = join $ go
  <$> newTranspose2d eigvec
  <*> diag1d eigval
  <*> expand2d mu
  <*> normal g 0 1
 where
  go :: t '[p, p] -> t '[p, p] -> t '[n, p] -> t '[p, n] -> IO (t '[n, p])
  go evec' eval' offset samps = (^+^ offset) <$> newTranspose2d (y !*! samps)
    where
      x = evec' !*! eval'
      y = x !*! eigvec

-- ========================================================================= --
-- Typeclass definition
-- ========================================================================= --

class THTensorRandom t where
  _random                     :: Dimensions d => t d -> Generator (t d) -> IO ()
  _clampedRandom              :: Dimensions d => t d -> Generator (t d) -> Integer -> Integer -> IO ()
  _cappedRandom               :: Dimensions d => t d -> Generator (t d) -> Integer -> IO ()
  _geometric                  :: Dimensions d => t d -> Generator (t d) -> HsAccReal (t d) -> IO ()
  _bernoulli                  :: Dimensions d => t d -> Generator (t d) -> HsAccReal (t d) -> IO ()
  _bernoulli_FloatTensor      :: Dimensions d => t d -> Generator (t d) -> TH.FloatTensor d -> IO ()
  _bernoulli_DoubleTensor     :: Dimensions d => t d -> Generator (t d) -> TH.DoubleTensor d -> IO ()

  _uniform                    :: Dimensions d => t d -> Generator (t d) -> HsAccReal (t d) -> HsAccReal (t d) -> IO ()
  _normal                     :: Dimensions d => t d -> Generator (t d) -> HsAccReal (t d) -> HsAccReal (t d) -> IO ()
  _normal_means               :: Dimensions d => t d -> Generator (t d) -> t d -> HsAccReal (t d) -> IO ()
  _normal_stddevs             :: Dimensions d => t d -> Generator (t d) -> HsAccReal (t d) -> t d -> IO ()
  _normal_means_stddevs       :: Dimensions d => t d -> Generator (t d) -> t d -> t d -> IO ()
  _exponential                :: Dimensions d => t d -> Generator (t d) -> HsAccReal (t d) -> IO ()
  _standard_gamma             :: Dimensions d => t d -> Generator (t d) -> t d -> IO ()
  _cauchy                     :: Dimensions d => t d -> Generator (t d) -> HsAccReal (t d) -> HsAccReal (t d) -> IO ()
  _logNormal                  :: Dimensions d => t d -> Generator (t d) -> HsAccReal (t d) -> HsAccReal (t d) -> IO ()

  _multinomial                :: Dimensions d => IndexTensor t d -> Generator (t d) -> t d -> Int -> Int -> IO ()
  _multinomialAliasSetup      :: Dimensions d => t d -> IndexTensor t d -> t d -> IO ()
  _multinomialAliasDraw       :: Dimensions d => IndexTensor t d -> Generator (t d) -> IndexTensor t d -> t d -> IO ()

random :: (IsTensor t, THTensorRandom t, Dimensions d) => Generator (t d) -> IO (t d)
random g = withEmpty (`_random` g)

clampedRandom :: (IsTensor t, THTensorRandom t, Dimensions d) => Generator (t d) -> Integer -> Integer -> IO (t d)
clampedRandom g a b = withEmpty $ \t -> _clampedRandom t g a b

cappedRandom :: (IsTensor t, THTensorRandom t, Dimensions d) => Generator (t d) -> Integer -> IO (t d)
cappedRandom g a = withEmpty $ \t -> _cappedRandom t g a

geometric :: (IsTensor t, THTensorRandom t, Dimensions d) => Generator (t d) -> HsAccReal (t d) -> IO (t d)
geometric g a = withEmpty $ \t -> _geometric t g a

bernoulli :: (IsTensor t, THTensorRandom t, Dimensions d) => Generator (t d) -> HsAccReal (t d) -> IO (t d)
bernoulli g a = withEmpty $ \t -> _bernoulli t g a

bernoulli_FloatTensor :: (IsTensor t, THTensorRandom t, Dimensions d) => Generator (t d) -> TH.FloatTensor d -> IO (t d)
bernoulli_FloatTensor g a = withEmpty $ \t -> _bernoulli_FloatTensor t g a

bernoulli_DoubleTensor :: (IsTensor t, THTensorRandom t, Dimensions d) => Generator (t d) -> TH.DoubleTensor d -> IO (t d)
bernoulli_DoubleTensor g a = withEmpty $ \t -> _bernoulli_DoubleTensor t g a

uniform :: (IsTensor t, THTensorRandom t, Dimensions d) => Generator (t d) -> HsAccReal (t d) -> HsAccReal (t d) -> IO (t d)
uniform g a b = withEmpty $ \t -> _uniform t g a b

normal :: (IsTensor t, THTensorRandom t, Dimensions d) => Generator (t d) -> HsAccReal (t d) -> HsAccReal (t d) -> IO (t d)
normal g a b = withEmpty $ \t -> _normal t g a b

normal_means :: (IsTensor t, THTensorRandom t, Dimensions d) => Generator (t d) -> t d -> HsAccReal (t d) -> IO (t d)
normal_means g m b = withEmpty $ \t -> _normal_means t g m b

normal_stddevs :: (IsTensor t, THTensorRandom t, Dimensions d) => Generator (t d) -> HsAccReal (t d) -> t d -> IO (t d)
normal_stddevs g a s = withEmpty $ \t -> _normal_stddevs t g a s

normal_means_stddevs :: (IsTensor t, THTensorRandom t, Dimensions d) => Generator (t d) -> t d -> t d -> IO (t d)
normal_means_stddevs g m s = withEmpty $ \t -> _normal_means_stddevs t g m s

exponential :: (IsTensor t, THTensorRandom t, Dimensions d) => Generator (t d) -> HsAccReal (t d) -> IO (t d)
exponential g a = withEmpty $ \t -> _exponential t g a

standard_gamma :: (IsTensor t, THTensorRandom t, Dimensions d) => Generator (t d) -> t d -> IO (t d)
standard_gamma g a = withEmpty $ \t -> _standard_gamma t g a

cauchy :: (IsTensor t, THTensorRandom t, Dimensions d) => Generator (t d) -> HsAccReal (t d) -> HsAccReal (t d) -> IO (t d)
cauchy g a b = withEmpty $ \t -> _cauchy t g a b

logNormal :: (IsTensor t, THTensorRandom t, Dimensions d) => Generator (t d) -> HsAccReal (t d) -> HsAccReal (t d) -> IO (t d)
logNormal g a b = withEmpty $ \t -> _logNormal t g a b

