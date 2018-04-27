module Torch.Class.THC.Tensor.Random.Static where

import Torch.Class.Types
-- import Torch.Class.Tensor
import Torch.Class.Tensor.Static
import Torch.Dimensions
import qualified Torch.Types.TH as TH

class THCTensorRandom t where
  _random                 :: t d -> IO ()
  _clampedRandom          :: t d -> Integer -> Integer -> IO ()
  _cappedRandom           :: t d -> Integer -> IO ()
  _bernoulli              :: t d -> HsAccReal (t d) -> IO ()
  _geometric              :: t d -> HsAccReal (t d) -> IO ()
  _bernoulli_DoubleTensor :: t d -> t d -> IO ()
  _uniform                :: t d -> HsAccReal (t d) -> HsAccReal (t d) -> IO ()
  _normal                 :: t d -> HsAccReal (t d) -> HsAccReal (t d) -> IO ()
  _normal_means           :: t d -> t d -> HsAccReal (t d) -> IO ()
  _normal_stddevs         :: t d -> HsAccReal (t d) -> t d -> IO ()
  _normal_means_stddevs   :: t d -> t d -> t d -> IO ()
  _logNormal              :: t d -> HsAccReal (t d) -> HsAccReal (t d) -> IO ()
  _exponential            :: t d -> HsAccReal (t d) -> IO ()
  _cauchy                 :: t d -> HsAccReal (t d) -> HsAccReal (t d) -> IO ()

  _multinomial            :: IndexTensor t d -> t d -> Int -> Int -> IO ()
  _multinomialAliasSetup  :: t d -> IndexTensor t d -> t d -> IO ()
  _multinomialAliasDraw   :: IndexTensor t d -> IndexTensor t d -> t d -> IO ()

  _rand                   :: t d -> TH.LongStorage -> IO ()
  _randn                  :: t d -> TH.LongStorage -> IO ()


random :: (IsTensor t, THCTensorRandom t, Dimensions d) => IO (t d)
random = new >>= \d -> _random d >> pure d

clampedRandom :: (IsTensor t, THCTensorRandom t, Dimensions d) => Integer -> Integer -> IO (t d)
clampedRandom a b = new >>= \d -> _clampedRandom d a b >> pure d

cappedRandom :: (IsTensor t, THCTensorRandom t, Dimensions d) => Integer -> IO (t d)
cappedRandom a = new >>= \d -> _cappedRandom d a >> pure d

bernoulli :: (IsTensor t, THCTensorRandom t, Dimensions d) => HsAccReal (t d) -> IO (t d)
bernoulli a = new >>= \d -> _bernoulli d a >> pure d

geometric :: (IsTensor t, THCTensorRandom t, Dimensions d) => HsAccReal (t d) -> IO (t d)
geometric a = new >>= \d -> _geometric d a >> pure d

bernoulli_DoubleTensor :: (IsTensor t, THCTensorRandom t, Dimensions d) => t d -> IO (t d)
bernoulli_DoubleTensor t = new >>= \r -> _bernoulli_DoubleTensor r t >> pure r

uniform :: (IsTensor t, THCTensorRandom t, Dimensions d) => HsAccReal (t d) -> HsAccReal (t d) -> IO (t d)
uniform a b = new >>= \d -> _uniform d a b >> pure d

normal :: (IsTensor t, THCTensorRandom t, Dimensions d) => HsAccReal (t d) -> HsAccReal (t d) -> IO (t d)
normal a b = new >>= \d -> _normal d a b >> pure d

normal_means :: (IsTensor t, THCTensorRandom t, Dimensions d) => t d -> HsAccReal (t d) -> IO (t d)
normal_means a b = new >>= \d -> _normal_means d a b >> pure d

normal_stddevs :: (IsTensor t, THCTensorRandom t, Dimensions d) => HsAccReal (t d) -> t d -> IO (t d)
normal_stddevs a b = new >>= \d -> _normal_stddevs d a b >> pure d

normal_means_stddevs :: (IsTensor t, THCTensorRandom t, Dimensions d) => t d -> t d -> IO (t d)
normal_means_stddevs a b = new >>= \d -> _normal_means_stddevs d a b >> pure d

logNormal :: (IsTensor t, THCTensorRandom t, Dimensions d) => HsAccReal (t d) -> HsAccReal (t d) -> IO (t d)
logNormal a b = new >>= \d -> _logNormal d a b >> pure d

exponential :: (IsTensor t, THCTensorRandom t, Dimensions d) => HsAccReal (t d) -> IO (t d)
exponential a = new >>= \d -> _exponential d a >> pure d

cauchy :: (IsTensor t, THCTensorRandom t, Dimensions d) => HsAccReal (t d) -> HsAccReal (t d) -> IO (t d)
cauchy a b = new >>= \d -> _cauchy d a b >> pure d

rand :: (IsTensor t, THCTensorRandom t, Dimensions d) => TH.LongStorage -> IO (t d)
rand a = new >>= \d -> _rand d a >> pure d

randn :: (IsTensor t, THCTensorRandom t, Dimensions d) => TH.LongStorage -> IO (t d)
randn a = new >>= \d -> _randn d a >> pure d


