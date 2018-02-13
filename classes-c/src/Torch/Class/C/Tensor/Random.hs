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
random g = new >>= \t -> random_ t g >> pure t

class TensorRandomFloating t where
  uniform                :: t -> Generator -> HsAccReal t -> HsAccReal t -> IO ()
  normal                 :: t -> Generator -> HsAccReal t -> HsAccReal t -> IO ()
  normal_means           :: t -> Generator -> t -> HsAccReal t -> IO ()
  normal_stddevs         :: t -> Generator -> HsAccReal t -> t -> IO ()
  normal_means_stddevs   :: t -> Generator -> t -> t -> IO ()
  exponential            :: t -> Generator -> HsAccReal t -> IO ()
  standard_gamma         :: t -> Generator -> t -> IO ()
  cauchy                 :: t -> Generator -> HsAccReal t -> HsAccReal t -> IO ()
  logNormal              :: t -> Generator -> HsAccReal t -> HsAccReal t -> IO ()
--  multinomial            :: Ptr CTHLongTensor -> Ptr CTHGenerator -> t -> Int32 -> Int32 -> IO ()
--  multinomialAliasSetup  :: t -> Ptr CTHLongTensor -> t -> IO ()
--  multinomialAliasDraw   :: Ptr CTHLongTensor -> Ptr CTHGenerator -> Ptr CTHLongTensor -> t -> IO ()

