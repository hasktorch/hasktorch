module Torch.Class.TH.Tensor.Random where

import Torch.Class.Types
import Torch.Dimensions
import Torch.Class.Tensor
import qualified Torch.Types.TH as TH

class THTensorRandom t where
  _random                     :: t -> Generator t -> IO ()
  _clampedRandom              :: t -> Generator t -> Integer -> Integer -> IO ()
  _cappedRandom               :: t -> Generator t -> Integer -> IO ()
  _geometric                  :: t -> Generator t -> HsAccReal t -> IO ()
  _bernoulli                  :: t -> Generator t -> HsAccReal t -> IO ()
  _bernoulli_FloatTensor      :: t -> Generator t -> TH.FloatDynamic -> IO ()
  _bernoulli_DoubleTensor     :: t -> Generator t -> TH.DoubleDynamic -> IO ()

  _uniform                    :: t -> Generator t -> HsAccReal t -> HsAccReal t -> IO ()
  _normal                     :: t -> Generator t -> HsAccReal t -> HsAccReal t -> IO ()
  _normal_means               :: t -> Generator t -> t -> HsAccReal t -> IO ()
  _normal_stddevs             :: t -> Generator t -> HsAccReal t -> t -> IO ()
  _normal_means_stddevs       :: t -> Generator t -> t -> t -> IO ()
  _exponential                :: t -> Generator t -> HsAccReal t -> IO ()
  _standard_gamma             :: t -> Generator t -> t -> IO ()
  _cauchy                     :: t -> Generator t -> HsAccReal t -> HsAccReal t -> IO ()
  _logNormal                  :: t -> Generator t -> HsAccReal t -> HsAccReal t -> IO ()

  _multinomial                :: IndexDynamic t -> Generator t -> t -> Int -> Int -> IO ()
  _multinomialAliasSetup      :: t -> IndexDynamic t -> t -> IO ()
  _multinomialAliasDraw       :: IndexDynamic t -> Generator t -> IndexDynamic t -> t -> IO ()

random :: (IsTensor t, THTensorRandom t) => Dim (d::[Nat]) -> Generator t -> IO t
random d g = withInplace (`_random` g) d

clampedRandom :: (IsTensor t, THTensorRandom t) => Dim (d::[Nat]) -> Generator t -> Integer -> Integer -> IO t
clampedRandom d g a b = flip withInplace d $ \t -> _clampedRandom t g a b

cappedRandom :: (IsTensor t, THTensorRandom t) => Dim (d::[Nat]) -> Generator t -> Integer -> IO t
cappedRandom d g a = flip withInplace d $ \t -> _cappedRandom t g a

geometric :: (IsTensor t, THTensorRandom t) => Dim (d::[Nat]) -> Generator t -> HsAccReal t -> IO t
geometric d g a = flip withInplace d $ \t -> _geometric t g a

bernoulli :: (IsTensor t, THTensorRandom t) => Dim (d::[Nat]) -> Generator t -> HsAccReal t -> IO t
bernoulli d g a = flip withInplace d $ \t -> _bernoulli t g a

bernoulli_FloatTensor :: (IsTensor t, THTensorRandom t) => Dim (d::[Nat]) -> Generator t -> TH.FloatDynamic -> IO t
bernoulli_FloatTensor d g a = flip withInplace d $ \t -> _bernoulli_FloatTensor t g a

bernoulli_DoubleTensor :: (IsTensor t, THTensorRandom t) => Dim (d::[Nat]) -> Generator t -> TH.DoubleDynamic -> IO t
bernoulli_DoubleTensor d g a = flip withInplace d $ \t -> _bernoulli_DoubleTensor t g a

uniform :: (IsTensor t, THTensorRandom t) => Dim (d::[Nat]) -> Generator t -> HsAccReal t -> HsAccReal t -> IO t
uniform d g a b = flip withInplace d $ \t -> _uniform t g a b
uniform' (SomeDims d) = uniform d

normal :: (IsTensor t, THTensorRandom t) => Dim (d::[Nat]) -> Generator t -> HsAccReal t -> HsAccReal t -> IO t
normal d g a b = flip withInplace d $ \t -> _normal t g a b
normal' (SomeDims d) = normal d

normal_means :: (IsTensor t, THTensorRandom t) => Dim (d::[Nat]) -> Generator t -> t -> HsAccReal t -> IO t
normal_means d g m b = flip withInplace d $ \t -> _normal_means t g m b

normal_stddevs :: (IsTensor t, THTensorRandom t) => Dim (d::[Nat]) -> Generator t -> HsAccReal t -> t -> IO t
normal_stddevs d g a s = flip withInplace d $ \t -> _normal_stddevs t g a s

normal_means_stddevs :: (IsTensor t, THTensorRandom t) => Dim (d::[Nat]) -> Generator t -> t -> t -> IO t
normal_means_stddevs d g m s = flip withInplace d $ \t -> _normal_means_stddevs t g m s

exponential :: (IsTensor t, THTensorRandom t) => Dim (d::[Nat]) -> Generator t -> HsAccReal t -> IO t
exponential d g a = flip withInplace d $ \t -> _exponential t g a

standard_gamma :: (IsTensor t, THTensorRandom t) => Dim (d::[Nat]) -> Generator t -> t -> IO t
standard_gamma d g a = flip withInplace d $ \t -> _standard_gamma t g a

cauchy :: (IsTensor t, THTensorRandom t) => Dim (d::[Nat]) -> Generator t -> HsAccReal t -> HsAccReal t -> IO t
cauchy d g a b = flip withInplace d $ \t -> _cauchy t g a b

logNormal :: (IsTensor t, THTensorRandom t) => Dim (d::[Nat]) -> Generator t -> HsAccReal t -> HsAccReal t -> IO t
logNormal d g a b = flip withInplace d $ \t -> _logNormal t g a b


