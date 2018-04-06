module Torch.Class.TH.Tensor.Random.Static where

import Torch.Class.Types
import Torch.Dimensions
import qualified Torch.Types.TH as TH

class THTensorRandom t where
  random                     :: Dimensions d => t d -> Generator (t d) -> IO ()
  clampedRandom              :: Dimensions d => t d -> Generator (t d) -> Integer -> Integer -> IO ()
  cappedRandom               :: Dimensions d => t d -> Generator (t d) -> Integer -> IO ()
  geometric                  :: Dimensions d => t d -> Generator (t d) -> HsAccReal (t d) -> IO ()
  bernoulli                  :: Dimensions d => t d -> Generator (t d) -> HsAccReal (t d) -> IO ()
  bernoulli_FloatTensor      :: Dimensions d => t d -> Generator (t d) -> TH.FloatTensor d -> IO ()
  bernoulli_DoubleTensor     :: Dimensions d => t d -> Generator (t d) -> TH.DoubleTensor d -> IO ()

  uniform                    :: Dimensions d => t d -> Generator (t d) -> HsAccReal (t d) -> HsAccReal (t d) -> IO ()
  normal                     :: Dimensions d => t d -> Generator (t d) -> HsAccReal (t d) -> HsAccReal (t d) -> IO ()
  normal_means               :: Dimensions d => t d -> Generator (t d) -> t d -> HsAccReal (t d) -> IO ()
  normal_stddevs             :: Dimensions d => t d -> Generator (t d) -> HsAccReal (t d) -> t d -> IO ()
  normal_means_stddevs       :: Dimensions d => t d -> Generator (t d) -> t d -> t d -> IO ()
  exponential                :: Dimensions d => t d -> Generator (t d) -> HsAccReal (t d) -> IO ()
  standard_gamma             :: Dimensions d => t d -> Generator (t d) -> t d -> IO ()
  cauchy                     :: Dimensions d => t d -> Generator (t d) -> HsAccReal (t d) -> HsAccReal (t d) -> IO ()
  logNormal                  :: Dimensions d => t d -> Generator (t d) -> HsAccReal (t d) -> HsAccReal (t d) -> IO ()

  multinomial                :: Dimensions d => IndexTensor (t d) d -> Generator (t d) -> t d -> Int -> Int -> IO ()
  multinomialAliasSetup      :: Dimensions d => t d -> IndexTensor (t d) d -> t d -> IO ()
  multinomialAliasDraw       :: Dimensions d => IndexTensor (t d) d -> Generator (t d) -> IndexTensor (t d) d -> t d -> IO ()
