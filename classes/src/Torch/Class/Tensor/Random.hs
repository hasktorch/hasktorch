{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
module Torch.Class.Tensor.Random where

import Torch.Types.TH
import Foreign
import GHC.Int
import GHC.TypeLits (Nat)
import Foreign.C.Types
import Torch.Class.Types
import Torch.Dimensions
import Torch.Class.Tensor
import qualified Torch.Types.TH.Float as F
import qualified Torch.Types.TH.Double as D

class TensorRandom t where
  random_                 :: t -> Generator t -> io ()
  clampedRandom_          :: t -> Generator t -> Int64 -> Int64 -> io ()
  cappedRandom_           :: t -> Generator t -> Int64 -> io ()
  geometric_              :: t -> Generator t -> Double -> io ()
  bernoulli_              :: t -> Generator t -> Double -> io ()
  bernoulli_FloatTensor_  :: t -> Generator t -> F.DynTensor -> io ()
  bernoulli_DoubleTensor_ :: t -> Generator t -> D.DynTensor -> io ()

random :: (Tensor io t, TensorRandom t) => Dim (d::[Nat]) -> Generator t -> io t
random d g = inplace (`random_` g) d

clampedRandom :: (Tensor io t, TensorRandom t) => Dim (d::[Nat]) -> Generator t -> Int64 -> Int64 -> io t
clampedRandom d g a b = flip inplace d $ \t -> clampedRandom_ t g a b

cappedRandom :: (Tensor io t, TensorRandom t) => Dim (d::[Nat]) -> Generator t -> Int64 -> io t
cappedRandom d g a = flip inplace d $ \t -> cappedRandom_ t g a

geometric :: (Tensor io t, TensorRandom t) => Dim (d::[Nat]) -> Generator t -> Double -> io t
geometric d g a = flip inplace d $ \t -> geometric_ t g a

bernoulli :: (Tensor io t, TensorRandom t) => Dim (d::[Nat]) -> Generator t -> Double -> io t
bernoulli d g a = flip inplace d $ \t -> bernoulli_ t g a

bernoulli_FloatTensor :: (Tensor io t, TensorRandom t) => Dim (d::[Nat]) -> Generator t -> F.DynTensor -> io t
bernoulli_FloatTensor d g a = flip inplace d $ \t -> bernoulli_FloatTensor_ t g a

bernoulli_DoubleTensor :: (Tensor io t, TensorRandom t) => Dim (d::[Nat]) -> Generator t -> D.DynTensor -> io t
bernoulli_DoubleTensor d g a = flip inplace d $ \t -> bernoulli_DoubleTensor_ t g a

class TensorRandomFloating t where
  uniform_               :: t -> Generator t -> HsAccReal t -> HsAccReal t -> io ()
  normal_                :: t -> Generator t -> HsAccReal t -> HsAccReal t -> io ()
  normal_means_          :: t -> Generator t -> t -> HsAccReal t -> io ()
  normal_stddevs_        :: t -> Generator t -> HsAccReal t -> t -> io ()
  normal_means_stddevs_  :: t -> Generator t -> t -> t -> io ()
  exponential_           :: t -> Generator t -> HsAccReal t -> io ()
  standard_gamma_        :: t -> Generator t -> t -> io ()
  cauchy_                :: t -> Generator t -> HsAccReal t -> HsAccReal t -> io ()
  logNormal_             :: t -> Generator t -> HsAccReal t -> HsAccReal t -> io ()
--  multinomial            :: Ptr CTHLongTensor -> Ptr CTHGenerator -> t -> Int32 -> Int32 -> io ()
--  multinomialAliasSetup  :: t -> Ptr CTHLongTensor -> t -> io ()
--  multinomialAliasDraw   :: Ptr CTHLongTensor -> Ptr CTHGenerator -> Ptr CTHLongTensor -> t -> io ()

uniform :: (Tensor io t, TensorRandomFloating t) => Dim (d::[Nat]) -> Generator t -> HsAccReal t -> HsAccReal t -> io t
uniform d g a b = flip inplace d $ \t -> uniform_ t g a b
uniform' (SomeDims d) = uniform d

normal :: (Tensor io t, TensorRandomFloating t) => Dim (d::[Nat]) -> Generator t -> HsAccReal t -> HsAccReal t -> io t
normal d g a b = flip inplace d $ \t -> normal_ t g a b
normal' (SomeDims d) = normal d

normal_means :: (Tensor io t, TensorRandomFloating t) => Dim (d::[Nat]) -> Generator t -> t -> HsAccReal t -> io t
normal_means d g m b = flip inplace d $ \t -> normal_means_ t g m b

normal_stddevs :: (Tensor io t, TensorRandomFloating t) => Dim (d::[Nat]) -> Generator t -> HsAccReal t -> t -> io t
normal_stddevs d g a s = flip inplace d $ \t -> normal_stddevs_ t g a s

normal_means_stddevs :: (Tensor io t, TensorRandomFloating t) => Dim (d::[Nat]) -> Generator t -> t -> t -> io t
normal_means_stddevs d g m s = flip inplace d $ \t -> normal_means_stddevs_ t g m s

exponential :: (Tensor io t, TensorRandomFloating t) => Dim (d::[Nat]) -> Generator t -> HsAccReal t -> io t
exponential d g a = flip inplace d $ \t -> exponential_ t g a

standard_gamma :: (Tensor io t, TensorRandomFloating t) => Dim (d::[Nat]) -> Generator t -> t -> io t
standard_gamma d g a = flip inplace d $ \t -> standard_gamma_ t g a

cauchy :: (Tensor io t, TensorRandomFloating t) => Dim (d::[Nat]) -> Generator t -> HsAccReal t -> HsAccReal t -> io t
cauchy d g a b = flip inplace d $ \t -> cauchy_ t g a b

logNormal :: (Tensor io t, TensorRandomFloating t) => Dim (d::[Nat]) -> Generator t -> HsAccReal t -> HsAccReal t -> io t
logNormal d g a b = flip inplace d $ \t -> logNormal_ t g a b


