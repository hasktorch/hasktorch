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
  random_                 :: t -> Generator t -> IO ()
  clampedRandom_          :: t -> Generator t -> Int64 -> Int64 -> IO ()
  cappedRandom_           :: t -> Generator t -> Int64 -> IO ()
  geometric_              :: t -> Generator t -> Double -> IO ()
  bernoulli_              :: t -> Generator t -> Double -> IO ()
  bernoulli_FloatTensor_  :: t -> Generator t -> F.DynTensor -> IO ()
  bernoulli_DoubleTensor_ :: t -> Generator t -> D.DynTensor -> IO ()

random :: (Tensor t, TensorRandom t) => Dim (d::[Nat]) -> Generator t -> IO t
random d g = inplace (`random_` g) d

clampedRandom :: (Tensor t, TensorRandom t) => Dim (d::[Nat]) -> Generator t -> Int64 -> Int64 -> IO t
clampedRandom d g a b = flip inplace d $ \t -> clampedRandom_ t g a b

cappedRandom :: (Tensor t, TensorRandom t) => Dim (d::[Nat]) -> Generator t -> Int64 -> IO t
cappedRandom d g a = flip inplace d $ \t -> cappedRandom_ t g a

geometric :: (Tensor t, TensorRandom t) => Dim (d::[Nat]) -> Generator t -> Double -> IO t
geometric d g a = flip inplace d $ \t -> geometric_ t g a

bernoulli :: (Tensor t, TensorRandom t) => Dim (d::[Nat]) -> Generator t -> Double -> IO t
bernoulli d g a = flip inplace d $ \t -> bernoulli_ t g a

bernoulli_FloatTensor :: (Tensor t, TensorRandom t) => Dim (d::[Nat]) -> Generator t -> F.DynTensor -> IO t
bernoulli_FloatTensor d g a = flip inplace d $ \t -> bernoulli_FloatTensor_ t g a

bernoulli_DoubleTensor :: (Tensor t, TensorRandom t) => Dim (d::[Nat]) -> Generator t -> D.DynTensor -> IO t
bernoulli_DoubleTensor d g a = flip inplace d $ \t -> bernoulli_DoubleTensor_ t g a

class TensorRandomFloating t where
  uniform_               :: t -> Generator t -> HsAccReal t -> HsAccReal t -> IO ()
  normal_                :: t -> Generator t -> HsAccReal t -> HsAccReal t -> IO ()
  normal_means_          :: t -> Generator t -> t -> HsAccReal t -> IO ()
  normal_stddevs_        :: t -> Generator t -> HsAccReal t -> t -> IO ()
  normal_means_stddevs_  :: t -> Generator t -> t -> t -> IO ()
  exponential_           :: t -> Generator t -> HsAccReal t -> IO ()
  standard_gamma_        :: t -> Generator t -> t -> IO ()
  cauchy_                :: t -> Generator t -> HsAccReal t -> HsAccReal t -> IO ()
  logNormal_             :: t -> Generator t -> HsAccReal t -> HsAccReal t -> IO ()
--  multinomial            :: Ptr CTHLongTensor -> Ptr CTHGenerator -> t -> Int32 -> Int32 -> IO ()
--  multinomialAliasSetup  :: t -> Ptr CTHLongTensor -> t -> IO ()
--  multinomialAliasDraw   :: Ptr CTHLongTensor -> Ptr CTHGenerator -> Ptr CTHLongTensor -> t -> IO ()

uniform :: (Tensor t, TensorRandomFloating t) => Dim (d::[Nat]) -> Generator t -> HsAccReal t -> HsAccReal t -> IO t
uniform d g a b = flip inplace d $ \t -> uniform_ t g a b
uniform' (SomeDims d) = uniform d

normal :: (Tensor t, TensorRandomFloating t) => Dim (d::[Nat]) -> Generator t -> HsAccReal t -> HsAccReal t -> IO t
normal d g a b = flip inplace d $ \t -> normal_ t g a b
normal' (SomeDims d) = normal d

normal_means :: (Tensor t, TensorRandomFloating t) => Dim (d::[Nat]) -> Generator t -> t -> HsAccReal t -> IO t
normal_means d g m b = flip inplace d $ \t -> normal_means_ t g m b

normal_stddevs :: (Tensor t, TensorRandomFloating t) => Dim (d::[Nat]) -> Generator t -> HsAccReal t -> t -> IO t
normal_stddevs d g a s = flip inplace d $ \t -> normal_stddevs_ t g a s

normal_means_stddevs :: (Tensor t, TensorRandomFloating t) => Dim (d::[Nat]) -> Generator t -> t -> t -> IO t
normal_means_stddevs d g m s = flip inplace d $ \t -> normal_means_stddevs_ t g m s

exponential :: (Tensor t, TensorRandomFloating t) => Dim (d::[Nat]) -> Generator t -> HsAccReal t -> IO t
exponential d g a = flip inplace d $ \t -> exponential_ t g a

standard_gamma :: (Tensor t, TensorRandomFloating t) => Dim (d::[Nat]) -> Generator t -> t -> IO t
standard_gamma d g a = flip inplace d $ \t -> standard_gamma_ t g a

cauchy :: (Tensor t, TensorRandomFloating t) => Dim (d::[Nat]) -> Generator t -> HsAccReal t -> HsAccReal t -> IO t
cauchy d g a b = flip inplace d $ \t -> cauchy_ t g a b

logNormal :: (Tensor t, TensorRandomFloating t) => Dim (d::[Nat]) -> Generator t -> HsAccReal t -> HsAccReal t -> IO t
logNormal d g a b = flip inplace d $ \t -> logNormal_ t g a b


