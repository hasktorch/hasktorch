module Torch.Class.C.Tensor.Random where

import THTypes
import Foreign
import GHC.Int
import Foreign.C.Types
import Torch.Class.C.Internal

class TensorRandom t where
  random                 :: t -> Ptr CTHGenerator -> IO ()
  clampedRandom          :: t -> Ptr CTHGenerator -> Int64 -> Int64 -> IO ()
  cappedRandom           :: t -> Ptr CTHGenerator -> Int64 -> IO ()
  geometric              :: t -> Ptr CTHGenerator -> Double -> IO ()
  bernoulli              :: t -> Ptr CTHGenerator -> Double -> IO ()
  bernoulli_FloatTensor  :: t -> Ptr CTHGenerator -> Ptr CTHFloatTensor -> IO ()
  bernoulli_DoubleTensor :: t -> Ptr CTHGenerator -> Ptr CTHDoubleTensor -> IO ()
  -- bernoulli_Tensor       :: t -> Ptr CTHGenerator -> t -> IO ()

class TensorRandomFloating t where
  uniform                :: t -> Ptr CTHGenerator -> HsAccReal t -> HsAccReal t -> IO ()
  normal                 :: t -> Ptr CTHGenerator -> HsAccReal t -> HsAccReal t -> IO ()
  normal_means           :: t -> Ptr CTHGenerator -> t -> HsAccReal t -> IO ()
  normal_stddevs         :: t -> Ptr CTHGenerator -> HsAccReal t -> t -> IO ()
  normal_means_stddevs   :: t -> Ptr CTHGenerator -> t -> t -> IO ()
  exponential            :: t -> Ptr CTHGenerator -> HsAccReal t -> IO ()
  standard_gamma         :: t -> Ptr CTHGenerator -> t -> IO ()
  cauchy                 :: t -> Ptr CTHGenerator -> HsAccReal t -> HsAccReal t -> IO ()
  logNormal              :: t -> Ptr CTHGenerator -> HsAccReal t -> HsAccReal t -> IO ()
--  multinomial            :: Ptr CTHLongTensor -> Ptr CTHGenerator -> t -> Int32 -> Int32 -> IO ()
--  multinomialAliasSetup  :: t -> Ptr CTHLongTensor -> t -> IO ()
--  multinomialAliasDraw   :: Ptr CTHLongTensor -> Ptr CTHGenerator -> Ptr CTHLongTensor -> t -> IO ()

