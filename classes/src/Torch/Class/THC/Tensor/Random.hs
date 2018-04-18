module Torch.Class.THC.Tensor.Random where

import Torch.Class.Types
import Torch.Class.Tensor
import Torch.Dimensions
import qualified Torch.Types.TH as TH

class THCTensorRandom t where
  _random :: t -> IO ()
  _clampedRandom :: t -> Integer -> Integer -> IO ()
  _cappedRandom :: t -> Integer -> IO ()
  _bernoulli :: t -> HsAccReal t -> IO ()
  _bernoulli_DoubleTensor :: t -> t -> IO ()
  _geometric :: t -> HsAccReal t -> IO ()

  _uniform :: t -> HsAccReal t -> HsAccReal t -> IO ()
  _normal :: t -> HsAccReal t -> HsAccReal t -> IO ()
  _normal_means :: t -> t -> HsAccReal t -> IO ()
  _normal_stddevs :: t -> HsAccReal t -> t -> IO ()
  _normal_means_stddevs :: t -> t -> t -> IO ()
  _logNormal :: t -> HsAccReal t -> HsAccReal t -> IO ()
  _exponential :: t -> HsAccReal t -> IO ()
  _cauchy :: t -> HsAccReal t -> HsAccReal t -> IO ()

  _multinomial :: IndexDynamic t -> t -> Int -> Int -> IO ()
  _multinomialAliasSetup :: t -> IndexDynamic t -> t -> IO ()
  _multinomialAliasDraw :: IndexDynamic t -> IndexDynamic t -> t -> IO ()

  _rand :: t -> TH.LongStorage -> IO ()
  _randn :: t -> TH.LongStorage -> IO ()

{-
random :: (Tensor io t, TensorRandom t) => Dim (d::[Nat]) -> Generator t -> io t
random d g = inplace (`_random` g) d

clampedRandom :: (Tensor io t, TensorRandom t) => Dim (d::[Nat]) -> Generator t -> Int64 -> Int64 -> io t
clampedRandom d g a b = flip inplace d $ \t -> _clampedRandom t g a b

cappedRandom :: (Tensor io t, TensorRandom t) => Dim (d::[Nat]) -> Generator t -> Int64 -> io t
cappedRandom d g a = flip inplace d $ \t -> _cappedRandom t g a

geometric :: (Tensor io t, TensorRandom t) => Dim (d::[Nat]) -> Generator t -> Double -> io t
geometric d g a = flip inplace d $ \t -> _geometric t g a

bernoulli :: (Tensor io t, TensorRandom t) => Dim (d::[Nat]) -> Generator t -> Double -> io t
bernoulli d g a = flip inplace d $ \t -> _bernoulli t g a

bernoulli_FloatTensor :: (Tensor io t, TensorRandom t) => Dim (d::[Nat]) -> Generator t -> F.DynTensor -> io t
bernoulli_FloatTensor d g a = flip inplace d $ \t -> _bernoulli_FloatTensor t g a

bernoulli_DoubleTensor :: (Tensor io t, TensorRandom t) => Dim (d::[Nat]) -> Generator t -> D.DynTensor -> io t
bernoulli_DoubleTensor d g a = flip inplace d $ \t -> _bernoulli_DoubleTensor t g a

class TensorRandomFloating t where
  _uniform               :: t -> Generator t -> HsAccReal t -> HsAccReal t -> io ()
  _normal                :: t -> Generator t -> HsAccReal t -> HsAccReal t -> io ()
  _normal_means          :: t -> Generator t -> t -> HsAccReal t -> io ()
  _normal_stddevs        :: t -> Generator t -> HsAccReal t -> t -> io ()
  _normal_means_stddevs  :: t -> Generator t -> t -> t -> io ()
  _exponential           :: t -> Generator t -> HsAccReal t -> io ()
  _standard_gamma        :: t -> Generator t -> t -> io ()
  _cauchy                :: t -> Generator t -> HsAccReal t -> HsAccReal t -> io ()
  _logNormal             :: t -> Generator t -> HsAccReal t -> HsAccReal t -> io ()
--  multinomial            :: Ptr CTHLongTensor -> Ptr CTHGenerator -> t -> Int32 -> Int32 -> io ()
--  multinomialAliasSetup  :: t -> Ptr CTHLongTensor -> t -> io ()
--  multinomialAliasDraw   :: Ptr CTHLongTensor -> Ptr CTHGenerator -> Ptr CTHLongTensor -> t -> io ()

uniform :: (Tensor io t, TensorRandomFloating t) => Dim (d::[Nat]) -> Generator t -> HsAccReal t -> HsAccReal t -> io t
uniform d g a b = flip inplace d $ \t -> _uniform t g a b
uniform' (SomeDims d) = uniform d

normal :: (Tensor io t, TensorRandomFloating t) => Dim (d::[Nat]) -> Generator t -> HsAccReal t -> HsAccReal t -> io t
normal d g a b = flip inplace d $ \t -> _normal t g a b
normal' (SomeDims d) = normal d

normal_means :: (Tensor io t, TensorRandomFloating t) => Dim (d::[Nat]) -> Generator t -> t -> HsAccReal t -> io t
normal_means d g m b = flip inplace d $ \t -> _normal_means t g m b

normal_stddevs :: (Tensor io t, TensorRandomFloating t) => Dim (d::[Nat]) -> Generator t -> HsAccReal t -> t -> io t
normal_stddevs d g a s = flip inplace d $ \t -> _normal_stddevs t g a s

normal_means_stddevs :: (Tensor io t, TensorRandomFloating t) => Dim (d::[Nat]) -> Generator t -> t -> t -> io t
normal_means_stddevs d g m s = flip inplace d $ \t -> _normal_means_stddevs t g m s

exponential :: (Tensor io t, TensorRandomFloating t) => Dim (d::[Nat]) -> Generator t -> HsAccReal t -> io t
exponential d g a = flip inplace d $ \t -> _exponential t g a

standard_gamma :: (Tensor io t, TensorRandomFloating t) => Dim (d::[Nat]) -> Generator t -> t -> io t
standard_gamma d g a = flip inplace d $ \t -> _standard_gamma t g a

cauchy :: (Tensor io t, TensorRandomFloating t) => Dim (d::[Nat]) -> Generator t -> HsAccReal t -> HsAccReal t -> io t
cauchy d g a b = flip inplace d $ \t -> _cauchy t g a b

logNormal :: (Tensor io t, TensorRandomFloating t) => Dim (d::[Nat]) -> Generator t -> HsAccReal t -> HsAccReal t -> io t
logNormal d g a b = flip inplace d $ \t -> _logNormal t g a b

-}
