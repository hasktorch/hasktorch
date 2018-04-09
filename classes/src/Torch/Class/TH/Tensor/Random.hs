module Torch.Class.TH.Tensor.Random where

import Torch.Class.Types
import Torch.Dimensions
import Torch.Class.Tensor
import qualified Torch.Types.TH as TH

class THTensorRandom t where
  random_                     :: t -> Generator t -> IO ()
  clampedRandom_              :: t -> Generator t -> Integer -> Integer -> IO ()
  cappedRandom_               :: t -> Generator t -> Integer -> IO ()
  geometric_                  :: t -> Generator t -> HsAccReal t -> IO ()
  bernoulli_                  :: t -> Generator t -> HsAccReal t -> IO ()
  bernoulli_FloatTensor_      :: t -> Generator t -> TH.FloatDynamic -> IO ()
  bernoulli_DoubleTensor_     :: t -> Generator t -> TH.DoubleDynamic -> IO ()

  uniform_                    :: t -> Generator t -> HsAccReal t -> HsAccReal t -> IO ()
  normal_                     :: t -> Generator t -> HsAccReal t -> HsAccReal t -> IO ()
  normal_means_               :: t -> Generator t -> t -> HsAccReal t -> IO ()
  normal_stddevs_             :: t -> Generator t -> HsAccReal t -> t -> IO ()
  normal_means_stddevs_       :: t -> Generator t -> t -> t -> IO ()
  exponential_                :: t -> Generator t -> HsAccReal t -> IO ()
  standard_gamma_             :: t -> Generator t -> t -> IO ()
  cauchy_                     :: t -> Generator t -> HsAccReal t -> HsAccReal t -> IO ()
  logNormal_                  :: t -> Generator t -> HsAccReal t -> HsAccReal t -> IO ()

  multinomial_                :: IndexDynamic t -> Generator t -> t -> Int -> Int -> IO ()
  multinomialAliasSetup_      :: t -> IndexDynamic t -> t -> IO ()
  multinomialAliasDraw_       :: IndexDynamic t -> Generator t -> IndexDynamic t -> t -> IO ()

random :: (Tensor t, THTensorRandom t) => Dim (d::[Nat]) -> Generator t -> IO t
random d g = withInplace (`random_` g) d

clampedRandom :: (Tensor t, THTensorRandom t) => Dim (d::[Nat]) -> Generator t -> Integer -> Integer -> IO t
clampedRandom d g a b = flip withInplace d $ \t -> clampedRandom_ t g a b

cappedRandom :: (Tensor t, THTensorRandom t) => Dim (d::[Nat]) -> Generator t -> Integer -> IO t
cappedRandom d g a = flip withInplace d $ \t -> cappedRandom_ t g a

geometric :: (Tensor t, THTensorRandom t) => Dim (d::[Nat]) -> Generator t -> HsAccReal t -> IO t
geometric d g a = flip withInplace d $ \t -> geometric_ t g a

bernoulli :: (Tensor t, THTensorRandom t) => Dim (d::[Nat]) -> Generator t -> HsAccReal t -> IO t
bernoulli d g a = flip withInplace d $ \t -> bernoulli_ t g a

bernoulli_FloatTensor :: (Tensor t, THTensorRandom t) => Dim (d::[Nat]) -> Generator t -> TH.FloatDynamic -> IO t
bernoulli_FloatTensor d g a = flip withInplace d $ \t -> bernoulli_FloatTensor_ t g a

bernoulli_DoubleTensor :: (Tensor t, THTensorRandom t) => Dim (d::[Nat]) -> Generator t -> TH.DoubleDynamic -> IO t
bernoulli_DoubleTensor d g a = flip withInplace d $ \t -> bernoulli_DoubleTensor_ t g a

uniform :: (Tensor t, THTensorRandom t) => Dim (d::[Nat]) -> Generator t -> HsAccReal t -> HsAccReal t -> IO t
uniform d g a b = flip withInplace d $ \t -> uniform_ t g a b
uniform' (SomeDims d) = uniform d

normal :: (Tensor t, THTensorRandom t) => Dim (d::[Nat]) -> Generator t -> HsAccReal t -> HsAccReal t -> IO t
normal d g a b = flip withInplace d $ \t -> normal_ t g a b
normal' (SomeDims d) = normal d

normal_means :: (Tensor t, THTensorRandom t) => Dim (d::[Nat]) -> Generator t -> t -> HsAccReal t -> IO t
normal_means d g m b = flip withInplace d $ \t -> normal_means_ t g m b

normal_stddevs :: (Tensor t, THTensorRandom t) => Dim (d::[Nat]) -> Generator t -> HsAccReal t -> t -> IO t
normal_stddevs d g a s = flip withInplace d $ \t -> normal_stddevs_ t g a s

normal_means_stddevs :: (Tensor t, THTensorRandom t) => Dim (d::[Nat]) -> Generator t -> t -> t -> IO t
normal_means_stddevs d g m s = flip withInplace d $ \t -> normal_means_stddevs_ t g m s

exponential :: (Tensor t, THTensorRandom t) => Dim (d::[Nat]) -> Generator t -> HsAccReal t -> IO t
exponential d g a = flip withInplace d $ \t -> exponential_ t g a

standard_gamma :: (Tensor t, THTensorRandom t) => Dim (d::[Nat]) -> Generator t -> t -> IO t
standard_gamma d g a = flip withInplace d $ \t -> standard_gamma_ t g a

cauchy :: (Tensor t, THTensorRandom t) => Dim (d::[Nat]) -> Generator t -> HsAccReal t -> HsAccReal t -> IO t
cauchy d g a b = flip withInplace d $ \t -> cauchy_ t g a b

logNormal :: (Tensor t, THTensorRandom t) => Dim (d::[Nat]) -> Generator t -> HsAccReal t -> HsAccReal t -> IO t
logNormal d g a b = flip withInplace d $ \t -> logNormal_ t g a b

