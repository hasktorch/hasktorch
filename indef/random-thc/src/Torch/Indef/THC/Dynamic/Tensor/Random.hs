module Torch.Indef.THC.Dynamic.Tensor.Random where

import Torch.Indef.Types
import qualified Torch.Class.THC.Tensor.Random as Class
import qualified Torch.Sig.THC.Tensor.Random as Sig
import qualified Torch.Sig.Types as Sig
import qualified Torch.Sig.Types.Global as Sig
import qualified Torch.Types.TH as TH

instance Class.THCTensorRandom Dynamic where
  random :: Dynamic -> IO ()
  random t = withDynamicState t Sig.c_random

  clampedRandom :: Dynamic -> Integer -> Integer -> IO ()
  clampedRandom t mn mx = withDynamicState t (shuffle2'2 Sig.c_clampedRandom (fromIntegral mn) (fromIntegral mx))

  cappedRandom :: Dynamic -> Integer -> IO ()
  cappedRandom t c = withDynamicState t (shuffle2 Sig.c_cappedRandom (fromIntegral c))

  bernoulli :: Dynamic -> HsAccReal -> IO ()
  bernoulli t v = withDynamicState t (shuffle2 Sig.c_bernoulli (hs2cAccReal v))

  bernoulli_DoubleTensor :: Dynamic -> Dynamic -> IO ()
  bernoulli_DoubleTensor t d = with2DynamicState t d Sig.c_bernoulli_DoubleTensor

  geometric :: Dynamic -> HsAccReal -> IO ()
  geometric t v = withDynamicState t (shuffle2 Sig.c_geometric (hs2cAccReal v))

  uniform :: Dynamic -> HsAccReal -> HsAccReal -> IO ()
  uniform t mn mx = withDynamicState t (shuffle2'2 Sig.c_uniform (hs2cAccReal mn) (hs2cAccReal mx))

  normal :: Dynamic -> HsAccReal -> HsAccReal -> IO ()
  normal t a b = withDynamicState t (shuffle2'2 Sig.c_normal (hs2cAccReal a) (hs2cAccReal b))

  normal_means :: Dynamic -> Dynamic -> HsAccReal -> IO ()
  normal_means t t0 v = with2DynamicState t t0 (shuffle3 Sig.c_normal_means (hs2cAccReal v))

  normal_stddevs :: Dynamic -> HsAccReal -> Dynamic -> IO ()
  normal_stddevs t a b = with2DynamicState t b $ \s' t' b' -> Sig.c_normal_stddevs s' t' (hs2cAccReal a) b'

  normal_means_stddevs :: Dynamic -> Dynamic -> Dynamic -> IO ()
  normal_means_stddevs t t0 t1 = with3DynamicState t t0 t1 Sig.c_normal_means_stddevs

  logNormal :: Dynamic -> HsAccReal -> HsAccReal -> IO ()
  logNormal t a b = withDynamicState t (shuffle2'2 Sig.c_logNormal (hs2cAccReal a) (hs2cAccReal b))

  exponential :: Dynamic -> HsAccReal -> IO ()
  exponential t v = withDynamicState t (shuffle2 Sig.c_exponential (hs2cAccReal v))

  cauchy :: Dynamic -> HsAccReal -> HsAccReal -> IO ()
  cauchy t a b = withDynamicState t (shuffle2'2 Sig.c_cauchy (hs2cAccReal a) (hs2cAccReal b))

  multinomial :: IndexDynamic -> Dynamic -> Int -> Int -> IO ()
  multinomial r t a b = runManaged . joinIO $ Sig.c_multinomial
    <$> manage' (fst . Sig.longDynamicState) r
    <*> manage' (snd . Sig.longDynamicState) r
    <*> manage' Sig.ctensor t
    <*> pure (fromIntegral a)
    <*> pure (fromIntegral b)

  multinomialAliasSetup :: Dynamic -> LongDynamic -> Dynamic -> IO ()
  multinomialAliasSetup r l t = runManaged . joinIO $ Sig.c_multinomialAliasSetup
    <$> manage' (Sig.dynamicStateRef) r
    <*> manage' (Sig.ctensor) r
    <*> manage' (snd . Sig.longDynamicState) l
    <*> manage' (Sig.ctensor) t

  multinomialAliasDraw  :: LongDynamic -> LongDynamic -> Dynamic -> IO ()
  multinomialAliasDraw r l t = runManaged . joinIO $ Sig.c_multinomialAliasDraw
    <$> manage' (fst . Sig.longDynamicState) r
    <*> manage' (snd . Sig.longDynamicState) r
    <*> manage' (snd . Sig.longDynamicState) l
    <*> manage' (Sig.ctensor) t

  rand  :: Dynamic -> TH.LongStorage -> IO ()
  rand r l = runManaged . joinIO $ Sig.c_rand
    <$> manage' (Sig.dynamicStateRef) r
    <*> manage' (Sig.ctensor) r
    <*> manage' (snd . TH.longStorageState) l

  randn  :: Dynamic -> TH.LongStorage -> IO ()
  randn r l = runManaged . joinIO $ Sig.c_randn
    <$> manage' (Sig.dynamicStateRef) r
    <*> manage' (Sig.ctensor) r
    <*> manage' (snd . TH.longStorageState) l


