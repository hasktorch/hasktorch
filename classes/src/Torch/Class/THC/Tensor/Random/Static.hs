module Torch.Class.THC.Tensor.Random.Static where

import Torch.Class.Types
import Torch.Class.Tensor
import Torch.Dimensions
import qualified Torch.Types.TH as TH

class THCTensorRandom t where
  random                 :: Dimensions d => t d -> IO ()
  clampedRandom          :: Dimensions d => t d -> Integer -> Integer -> IO ()
  cappedRandom           :: Dimensions d => t d -> Integer -> IO ()
  bernoulli              :: Dimensions d => t d -> HsAccReal (t d) -> IO ()
  geometric              :: Dimensions d => t d -> HsAccReal (t d) -> IO ()
  bernoulli_DoubleTensor :: Dimensions d => t d -> t d -> IO ()
  uniform                :: Dimensions d => t d -> HsAccReal (t d) -> HsAccReal (t d) -> IO ()
  normal                 :: Dimensions d => t d -> HsAccReal (t d) -> HsAccReal (t d) -> IO ()
  normal_means           :: Dimensions d => t d -> t d -> HsAccReal (t d) -> IO ()
  normal_stddevs         :: Dimensions d => t d -> HsAccReal (t d) -> t d -> IO ()
  normal_means_stddevs   :: Dimensions d => t d -> t d -> t d -> IO ()
  logNormal              :: Dimensions d => t d -> HsAccReal (t d) -> HsAccReal (t d) -> IO ()
  exponential            :: Dimensions d => t d -> HsAccReal (t d) -> IO ()
  cauchy                 :: Dimensions d => t d -> HsAccReal (t d) -> HsAccReal (t d) -> IO ()

  multinomial            :: Dimensions d => IndexTensor (t d) d -> t d -> Int -> Int -> IO ()
  multinomialAliasSetup  :: Dimensions d => t d -> IndexTensor (t d) d -> t d -> IO ()
  multinomialAliasDraw   :: Dimensions d => IndexTensor (t d) d -> IndexTensor (t d) d -> t d -> IO ()

  rand                   :: Dimensions d => t d -> TH.LongStorage -> IO ()
  randn                  :: Dimensions d => t d -> TH.LongStorage -> IO ()


