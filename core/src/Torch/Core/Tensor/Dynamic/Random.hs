{-# LANGUAGE TypeSynonymInstances #-}
module Torch.Core.Tensor.Dynamic.Random
  ( TensorRandom(..)
  ) where

import Torch.Class.C.Internal
import GHC.Int
import qualified Torch.Class.C.Tensor.Random as CCall

import THTypes
import Foreign
import qualified Torch.Core.LongTensor.Dynamic   as L
import qualified Torch.Core.FloatTensor.Dynamic  as F
import qualified Torch.Core.ByteTensor.Dynamic   as B
-- import qualified Torch.Core.CharTensor.Dynamic   as C
import qualified Torch.Core.ShortTensor.Dynamic  as S
import qualified Torch.Core.IntTensor.Dynamic    as I
import qualified Torch.Core.DoubleTensor.Dynamic as D
-- import qualified Torch.Core.HalfTensor.Dynamic   as H

type FloatTensor = F.Tensor
type DoubleTensor = D.Tensor

class CCall.TensorRandom t => TensorRandom t where
  random                 :: t -> Ptr CTHGenerator -> IO ()
  random                 = CCall.random
  clampedRandom          :: t -> Ptr CTHGenerator -> Int64 -> Int64 -> IO ()
  clampedRandom          = CCall.clampedRandom
  cappedRandom           :: t -> Ptr CTHGenerator -> Int64 -> IO ()
  cappedRandom           = CCall.cappedRandom
  geometric              :: t -> Ptr CTHGenerator -> Double -> IO ()
  geometric              = CCall.geometric
  bernoulli              :: t -> Ptr CTHGenerator -> Double -> IO ()
  bernoulli              = CCall.bernoulli
  bernoulli_FloatTensor  :: t -> Ptr CTHGenerator -> FloatTensor -> IO ()
  bernoulli_FloatTensor t g f  = withForeignPtr (F.tensor f) $ \f' -> CCall.bernoulli_FloatTensor t g f'
  bernoulli_DoubleTensor :: t -> Ptr CTHGenerator -> DoubleTensor -> IO ()
  bernoulli_DoubleTensor t g d = withForeignPtr (D.tensor d) $ \d' -> CCall.bernoulli_DoubleTensor t g d'

instance TensorRandom B.Tensor where
instance TensorRandom S.Tensor where
instance TensorRandom I.Tensor where
instance TensorRandom L.Tensor where
instance TensorRandom F.Tensor where
instance TensorRandom D.Tensor where

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

