module Torch.Indef.THC.Dynamic.Tensor.Random where

import Torch.Indef.Types
import qualified Torch.Class.THC.Tensor.Random as Class
import qualified Torch.Sig.THC.Tensor.Random as Sig
import qualified Torch.Sig.Types as Sig
import qualified Torch.Sig.Types.Global as Sig
import qualified Torch.Types.TH as TH

instance Class.THCTensorRandom Dynamic where
  _random :: Dynamic -> IO ()
  _random t = withDynamicState t Sig.c_random

  _clampedRandom :: Dynamic -> Integer -> Integer -> IO ()
  _clampedRandom t mn mx = withDynamicState t (shuffle2'2 Sig.c_clampedRandom (fromIntegral mn) (fromIntegral mx))

  _cappedRandom :: Dynamic -> Integer -> IO ()
  _cappedRandom t c = withDynamicState t (shuffle2 Sig.c_cappedRandom (fromIntegral c))

  _bernoulli :: Dynamic -> HsAccReal -> IO ()
  _bernoulli t v = withDynamicState t (shuffle2 Sig.c_bernoulli (hs2cAccReal v))

  _bernoulli_DoubleTensor :: Dynamic -> Dynamic -> IO ()
  _bernoulli_DoubleTensor t d = with2DynamicState t d Sig.c_bernoulli_DoubleTensor

  _geometric :: Dynamic -> HsAccReal -> IO ()
  _geometric t v = withDynamicState t (shuffle2 Sig.c_geometric (hs2cAccReal v))

  _uniform :: Dynamic -> HsAccReal -> HsAccReal -> IO ()
  _uniform t mn mx = withDynamicState t (shuffle2'2 Sig.c_uniform (hs2cAccReal mn) (hs2cAccReal mx))

  _normal :: Dynamic -> HsAccReal -> HsAccReal -> IO ()
  _normal t a b = withDynamicState t (shuffle2'2 Sig.c_normal (hs2cAccReal a) (hs2cAccReal b))

  _normal_means :: Dynamic -> Dynamic -> HsAccReal -> IO ()
  _normal_means t t0 v = with2DynamicState t t0 (shuffle3 Sig.c_normal_means (hs2cAccReal v))

  _normal_stddevs :: Dynamic -> HsAccReal -> Dynamic -> IO ()
  _normal_stddevs t a b = with2DynamicState t b $ \s' t' b' -> Sig.c_normal_stddevs s' t' (hs2cAccReal a) b'

  _normal_means_stddevs :: Dynamic -> Dynamic -> Dynamic -> IO ()
  _normal_means_stddevs t t0 t1 = with3DynamicState t t0 t1 Sig.c_normal_means_stddevs

  _logNormal :: Dynamic -> HsAccReal -> HsAccReal -> IO ()
  _logNormal t a b = withDynamicState t (shuffle2'2 Sig.c_logNormal (hs2cAccReal a) (hs2cAccReal b))

  _exponential :: Dynamic -> HsAccReal -> IO ()
  _exponential t v = withDynamicState t (shuffle2 Sig.c_exponential (hs2cAccReal v))

  _cauchy :: Dynamic -> HsAccReal -> HsAccReal -> IO ()
  _cauchy t a b = withDynamicState t (shuffle2'2 Sig.c_cauchy (hs2cAccReal a) (hs2cAccReal b))

  _multinomial :: IndexDynamic -> Dynamic -> Int -> Int -> IO ()
  _multinomial r t a b = runManaged . joinIO $ Sig.c_multinomial
     <$> manage' (fst . Sig.longDynamicState) r
     <*> manage' (snd . Sig.longDynamicState) r
     <*> manage' Sig.ctensor t
     <*> pure (fromIntegral a)
     <*> pure (fromIntegral b)

  _multinomialAliasSetup :: Dynamic -> LongDynamic -> Dynamic -> IO ()
  _multinomialAliasSetup r l t = runManaged . joinIO $ Sig.c_multinomialAliasSetup
     <$> manage' (Sig.dynamicStateRef) r
     <*> manage' (Sig.ctensor) r
     <*> manage' (snd . Sig.longDynamicState) l
     <*> manage' (Sig.ctensor) t

  _multinomialAliasDraw  :: LongDynamic -> LongDynamic -> Dynamic -> IO ()
  _multinomialAliasDraw r l t = runManaged . joinIO $ Sig.c_multinomialAliasDraw
     <$> manage' (fst . Sig.longDynamicState) r
     <*> manage' (snd . Sig.longDynamicState) r
     <*> manage' (snd . Sig.longDynamicState) l
     <*> manage' (Sig.ctensor) t

  _rand  :: Dynamic -> TH.LongStorage -> IO ()
  _rand r l = runManaged . joinIO $ Sig.c_rand
     <$> manage' (Sig.dynamicStateRef) r
     <*> manage' (Sig.ctensor) r
     <*> manage' (snd . TH.longStorageState) l

  _randn  :: Dynamic -> TH.LongStorage -> IO ()
  _randn r l = runManaged . joinIO $ Sig.c_randn
    <$> manage' (Sig.dynamicStateRef) r
    <*> manage' (Sig.ctensor) r
    <*> manage' (snd . TH.longStorageState) l


