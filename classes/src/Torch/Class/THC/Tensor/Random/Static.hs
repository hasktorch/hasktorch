module Torch.Class.THC.Tensor.Random.Static where

import Torch.Class.Types
import Torch.Class.Tensor
import Torch.Dimensions
import qualified Torch.Types.TH as TH

class THCTensorRandom t where
  _random                 :: Dimensions d => t d -> IO ()
  _clampedRandom          :: Dimensions d => t d -> Integer -> Integer -> IO ()
  _cappedRandom           :: Dimensions d => t d -> Integer -> IO ()
  _bernoulli              :: Dimensions d => t d -> HsAccReal (t d) -> IO ()
  _geometric              :: Dimensions d => t d -> HsAccReal (t d) -> IO ()
  _bernoulli_DoubleTensor :: Dimensions d => t d -> t d -> IO ()
  _uniform                :: Dimensions d => t d -> HsAccReal (t d) -> HsAccReal (t d) -> IO ()
  _normal                 :: Dimensions d => t d -> HsAccReal (t d) -> HsAccReal (t d) -> IO ()
  _normal_means           :: Dimensions d => t d -> t d -> HsAccReal (t d) -> IO ()
  _normal_stddevs         :: Dimensions d => t d -> HsAccReal (t d) -> t d -> IO ()
  _normal_means_stddevs   :: Dimensions d => t d -> t d -> t d -> IO ()
  _logNormal              :: Dimensions d => t d -> HsAccReal (t d) -> HsAccReal (t d) -> IO ()
  _exponential            :: Dimensions d => t d -> HsAccReal (t d) -> IO ()
  _cauchy                 :: Dimensions d => t d -> HsAccReal (t d) -> HsAccReal (t d) -> IO ()

  _multinomial            :: Dimensions d => IndexTensor (t d) d -> t d -> Int -> Int -> IO ()
  _multinomialAliasSetup  :: Dimensions d => t d -> IndexTensor (t d) d -> t d -> IO ()
  _multinomialAliasDraw   :: Dimensions d => IndexTensor (t d) d -> IndexTensor (t d) d -> t d -> IO ()

  _rand                   :: Dimensions d => t d -> TH.LongStorage -> IO ()
  _randn                  :: Dimensions d => t d -> TH.LongStorage -> IO ()


