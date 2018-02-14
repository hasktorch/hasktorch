module Torch.Class.C.Tensor.Random where

import THTypes
import Foreign hiding (new)
import GHC.Int
import Foreign.C.Types
import Torch.Class.C.Internal
import Torch.Class.C.IsTensor
import THRandomTypes (Generator)
import qualified THFloatTypes as F
import qualified THDoubleTypes as D

class TensorRandom t where
  random_                 :: t -> Generator -> IO ()
  clampedRandom_          :: t -> Generator -> Int64 -> Int64 -> IO ()
  cappedRandom_           :: t -> Generator -> Int64 -> IO ()
  geometric_              :: t -> Generator -> Double -> IO ()
  bernoulli_              :: t -> Generator -> Double -> IO ()
  bernoulli_FloatTensor_  :: t -> Generator -> F.DynTensor -> IO ()
  bernoulli_DoubleTensor_ :: t -> Generator -> D.DynTensor -> IO ()

random :: (IsTensor t, TensorRandom t) => Generator -> IO t
random g = inplace (`random_` g)

clampedRandom :: (IsTensor t, TensorRandom t) => Generator -> Int64 -> Int64 -> IO t
clampedRandom g a b = inplace $ \t -> clampedRandom_ t g a b

cappedRandom :: (IsTensor t, TensorRandom t) => Generator -> Int64 -> IO t
cappedRandom g a = inplace $ \t -> cappedRandom_ t g a

geometric :: (IsTensor t, TensorRandom t) => Generator -> Double -> IO t
geometric g a = inplace $ \t -> geometric_ t g a

bernoulli :: (IsTensor t, TensorRandom t) => Generator -> Double -> IO t
bernoulli g a = inplace $ \t -> bernoulli_ t g a

bernoulli_FloatTensor :: (IsTensor t, TensorRandom t) => Generator -> F.DynTensor -> IO t
bernoulli_FloatTensor g a = inplace $ \t -> bernoulli_FloatTensor_ t g a

bernoulli_DoubleTensor :: (IsTensor t, TensorRandom t) => Generator -> D.DynTensor -> IO t
bernoulli_DoubleTensor g a = inplace $ \t -> bernoulli_DoubleTensor_ t g a

class TensorRandomFloating t where
  uniform_               :: t -> Generator -> HsAccReal t -> HsAccReal t -> IO ()
  normal_                :: t -> Generator -> HsAccReal t -> HsAccReal t -> IO ()
  normal_means_          :: t -> Generator -> t -> HsAccReal t -> IO ()
  normal_stddevs_        :: t -> Generator -> HsAccReal t -> t -> IO ()
  normal_means_stddevs_  :: t -> Generator -> t -> t -> IO ()
  exponential_           :: t -> Generator -> HsAccReal t -> IO ()
  standard_gamma_        :: t -> Generator -> t -> IO ()
  cauchy_                :: t -> Generator -> HsAccReal t -> HsAccReal t -> IO ()
  logNormal_             :: t -> Generator -> HsAccReal t -> HsAccReal t -> IO ()
--  multinomial            :: Ptr CTHLongTensor -> Ptr CTHGenerator -> t -> Int32 -> Int32 -> IO ()
--  multinomialAliasSetup  :: t -> Ptr CTHLongTensor -> t -> IO ()
--  multinomialAliasDraw   :: Ptr CTHLongTensor -> Ptr CTHGenerator -> Ptr CTHLongTensor -> t -> IO ()

uniform :: (IsTensor t, TensorRandomFloating t) => Generator -> HsAccReal t -> HsAccReal t -> IO t
uniform g a b = inplace $ \t -> uniform_ t g a b

normal :: (IsTensor t, TensorRandomFloating t) => Generator -> HsAccReal t -> HsAccReal t -> IO t
normal g a b = inplace $ \t -> normal_ t g a b

normal_means :: (IsTensor t, TensorRandomFloating t) => Generator -> t -> HsAccReal t -> IO t
normal_means g m b = inplace $ \t -> normal_means_ t g m b

normal_stddevs :: (IsTensor t, TensorRandomFloating t) => Generator -> HsAccReal t -> t -> IO t
normal_stddevs g a s = inplace $ \t -> normal_stddevs_ t g a s

normal_means_stddevs :: (IsTensor t, TensorRandomFloating t) => Generator -> t -> t -> IO t
normal_means_stddevs g m s = inplace $ \t -> normal_means_stddevs_ t g m s

exponential :: (IsTensor t, TensorRandomFloating t) => Generator -> HsAccReal t -> IO t
exponential g a = inplace $ \t -> exponential_ t g a

standard_gamma :: (IsTensor t, TensorRandomFloating t) => Generator -> t -> IO t
standard_gamma g a = inplace $ \t -> standard_gamma_ t g a

cauchy :: (IsTensor t, TensorRandomFloating t) => Generator -> HsAccReal t -> HsAccReal t -> IO t
cauchy g a b = inplace $ \t -> cauchy_ t g a b

logNormal :: (IsTensor t, TensorRandomFloating t) => Generator -> HsAccReal t -> HsAccReal t -> IO t
logNormal g a b = inplace $ \t -> logNormal_ t g a b


