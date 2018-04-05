module Torch.Class.TH.Tensor.Random where

import Torch.Class.Types
import Torch.Dimensions
import qualified Torch.Types.TH as TH

class THTensorRandom t where
  random                     :: t -> Generator t -> IO ()
  clampedRandom              :: t -> Generator t -> Integer -> Integer -> IO ()
  cappedRandom               :: t -> Generator t -> Integer -> IO ()
  geometric                  :: t -> Generator t -> HsAccReal t -> IO ()
  bernoulli                  :: t -> Generator t -> HsAccReal t -> IO ()
  bernoulli_FloatTensor      :: t -> Generator t -> TH.FloatDynamic -> IO ()
  bernoulli_DoubleTensor     :: t -> Generator t -> TH.DoubleDynamic -> IO ()

  uniform                    :: t -> Generator t -> HsAccReal t -> HsAccReal t -> IO ()
  normal                     :: t -> Generator t -> HsAccReal t -> HsAccReal t -> IO ()
  normal_means               :: t -> Generator t -> t -> HsAccReal t -> IO ()
  normal_stddevs             :: t -> Generator t -> HsAccReal t -> t -> IO ()
  normal_means_stddevs       :: t -> Generator t -> t -> t -> IO ()
  exponential                :: t -> Generator t -> HsAccReal t -> IO ()
  standard_gamma             :: t -> Generator t -> t -> IO ()
  cauchy                     :: t -> Generator t -> HsAccReal t -> HsAccReal t -> IO ()
  logNormal                  :: t -> Generator t -> HsAccReal t -> HsAccReal t -> IO ()

-- c_multinomialAliasSetup :: Ptr CState -> Ptr CTensor -> Ptr CLongTensor -> Ptr CTensor -> IO ()
-- c_multinomialAliasDraw  :: Ptr CState -> Ptr CLongTensor -> Ptr CGenerator -> Ptr CLongTensor -> Ptr CTensor -> IO ()
  multinomial                :: IndexDynamic t -> Generator t -> t -> Int -> Int -> IO ()
  multinomialAliasSetup      :: t -> IndexDynamic t -> t -> IO ()
  multinomialAliasDraw       :: IndexDynamic t -> Generator t -> IndexDynamic t -> t -> IO ()
{-
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
--  multinomial            :: THLongTensor -> THGenerator -> t -> Int32 -> Int32 -> io ()
--  multinomialAliasSetup  :: t -> THLongTensor -> t -> io ()
--  multinomialAliasDraw   :: THLongTensor -> Ptr CTHGenerator -> Ptr CTHLongTensor -> t -> io ()

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

-}
