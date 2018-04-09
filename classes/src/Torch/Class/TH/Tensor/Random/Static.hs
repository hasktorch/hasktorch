{-# LANGUAGE ScopedTypeVariables #-}
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
  => (IsTensor t, THTensorRandom t)
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
  random_                     :: Dimensions d => t d -> Generator (t d) -> IO ()
  clampedRandom_              :: Dimensions d => t d -> Generator (t d) -> Integer -> Integer -> IO ()
  cappedRandom_               :: Dimensions d => t d -> Generator (t d) -> Integer -> IO ()
  geometric_                  :: Dimensions d => t d -> Generator (t d) -> HsAccReal (t d) -> IO ()
  bernoulli_                  :: Dimensions d => t d -> Generator (t d) -> HsAccReal (t d) -> IO ()
  bernoulli_FloatTensor_      :: Dimensions d => t d -> Generator (t d) -> TH.FloatTensor d -> IO ()
  bernoulli_DoubleTensor_     :: Dimensions d => t d -> Generator (t d) -> TH.DoubleTensor d -> IO ()

  uniform_                    :: Dimensions d => t d -> Generator (t d) -> HsAccReal (t d) -> HsAccReal (t d) -> IO ()
  normal_                     :: Dimensions d => t d -> Generator (t d) -> HsAccReal (t d) -> HsAccReal (t d) -> IO ()
  normal_means_               :: Dimensions d => t d -> Generator (t d) -> t d -> HsAccReal (t d) -> IO ()
  normal_stddevs_             :: Dimensions d => t d -> Generator (t d) -> HsAccReal (t d) -> t d -> IO ()
  normal_means_stddevs_       :: Dimensions d => t d -> Generator (t d) -> t d -> t d -> IO ()
  exponential_                :: Dimensions d => t d -> Generator (t d) -> HsAccReal (t d) -> IO ()
  standard_gamma_             :: Dimensions d => t d -> Generator (t d) -> t d -> IO ()
  cauchy_                     :: Dimensions d => t d -> Generator (t d) -> HsAccReal (t d) -> HsAccReal (t d) -> IO ()
  logNormal_                  :: Dimensions d => t d -> Generator (t d) -> HsAccReal (t d) -> HsAccReal (t d) -> IO ()

  multinomial_                :: Dimensions d => IndexTensor (t d) d -> Generator (t d) -> t d -> Int -> Int -> IO ()
  multinomialAliasSetup_      :: Dimensions d => t d -> IndexTensor (t d) d -> t d -> IO ()
  multinomialAliasDraw_       :: Dimensions d => IndexTensor (t d) d -> Generator (t d) -> IndexTensor (t d) d -> t d -> IO ()

random :: (IsTensor t, THTensorRandom t, Dimensions d) => Generator (t d) -> IO (t d)
random g = withEmpty (`random_` g)

clampedRandom :: (IsTensor t, THTensorRandom t, Dimensions d) => Generator (t d) -> Integer -> Integer -> IO (t d)
clampedRandom g a b = withEmpty $ \t -> clampedRandom_ t g a b

cappedRandom :: (IsTensor t, THTensorRandom t, Dimensions d) => Generator (t d) -> Integer -> IO (t d)
cappedRandom g a = withEmpty $ \t -> cappedRandom_ t g a

geometric :: (IsTensor t, THTensorRandom t, Dimensions d) => Generator (t d) -> HsAccReal (t d) -> IO (t d)
geometric g a = withEmpty $ \t -> geometric_ t g a

bernoulli :: (IsTensor t, THTensorRandom t, Dimensions d) => Generator (t d) -> HsAccReal (t d) -> IO (t d)
bernoulli g a = withEmpty $ \t -> bernoulli_ t g a

bernoulli_FloatTensor :: (IsTensor t, THTensorRandom t, Dimensions d) => Generator (t d) -> TH.FloatTensor d -> IO (t d)
bernoulli_FloatTensor g a = withEmpty $ \t -> bernoulli_FloatTensor_ t g a

bernoulli_DoubleTensor :: (IsTensor t, THTensorRandom t, Dimensions d) => Generator (t d) -> TH.DoubleTensor d -> IO (t d)
bernoulli_DoubleTensor g a = withEmpty $ \t -> bernoulli_DoubleTensor_ t g a

uniform :: (IsTensor t, THTensorRandom t, Dimensions d) => Generator (t d) -> HsAccReal (t d) -> HsAccReal (t d) -> IO (t d)
uniform g a b = withEmpty $ \t -> uniform_ t g a b

normal :: (IsTensor t, THTensorRandom t, Dimensions d) => Generator (t d) -> HsAccReal (t d) -> HsAccReal (t d) -> IO (t d)
normal g a b = withEmpty $ \t -> normal_ t g a b

normal_means :: (IsTensor t, THTensorRandom t, Dimensions d) => Generator (t d) -> t d -> HsAccReal (t d) -> IO (t d)
normal_means g m b = withEmpty $ \t -> normal_means_ t g m b

normal_stddevs :: (IsTensor t, THTensorRandom t, Dimensions d) => Generator (t d) -> HsAccReal (t d) -> t d -> IO (t d)
normal_stddevs g a s = withEmpty $ \t -> normal_stddevs_ t g a s

normal_means_stddevs :: (IsTensor t, THTensorRandom t, Dimensions d) => Generator (t d) -> t d -> t d -> IO (t d)
normal_means_stddevs g m s = withEmpty $ \t -> normal_means_stddevs_ t g m s

exponential :: (IsTensor t, THTensorRandom t, Dimensions d) => Generator (t d) -> HsAccReal (t d) -> IO (t d)
exponential g a = withEmpty $ \t -> exponential_ t g a

standard_gamma :: (IsTensor t, THTensorRandom t, Dimensions d) => Generator (t d) -> t d -> IO (t d)
standard_gamma g a = withEmpty $ \t -> standard_gamma_ t g a

cauchy :: (IsTensor t, THTensorRandom t, Dimensions d) => Generator (t d) -> HsAccReal (t d) -> HsAccReal (t d) -> IO (t d)
cauchy g a b = withEmpty $ \t -> cauchy_ t g a b

logNormal :: (IsTensor t, THTensorRandom t, Dimensions d) => Generator (t d) -> HsAccReal (t d) -> HsAccReal (t d) -> IO (t d)
logNormal g a b = withEmpty $ \t -> logNormal_ t g a b

