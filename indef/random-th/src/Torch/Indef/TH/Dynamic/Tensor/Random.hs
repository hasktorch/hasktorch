module Torch.Indef.TH.Dynamic.Tensor.Random () where

import Torch.Indef.Types
import qualified Torch.Class.TH.Tensor.Random as Class
import qualified Torch.Sig.TH.Tensor.Random as Sig
import qualified Torch.Sig.Types as Sig
import qualified Torch.Sig.Types.Global as Sig
import qualified Torch.Types.TH as TH

tenGen
  :: Dynamic
  -> Generator
  -> (Ptr CState -> Ptr CTensor -> Ptr CGenerator -> IO x)
  -> IO x
tenGen r g fn =
  withDynamicState r $ \s' r' ->
    withGen g (fn s' r')

tenGenTen
  :: Dynamic
  -> Generator
  -> ForeignPtr a
  -> (Ptr CState -> Ptr CTensor -> Ptr CGenerator -> Ptr a -> IO x)
  -> IO x
tenGenTen r g t fn = tenGen r g $ \s' r' g' ->
    withForeignPtr t (fn s' r' g')

instance Class.THTensorRandom Dynamic where
  _random t g = tenGen t g Sig.c_random
  _clampedRandom r g a b = tenGen r g $ shuffle3'2 Sig.c_clampedRandom (fromIntegral a) (fromIntegral b)
  _cappedRandom r g a = tenGen r g $ shuffle3 Sig.c_cappedRandom (fromIntegral a)
  _geometric r g a = tenGen r g $ shuffle3 Sig.c_geometric (hs2cAccReal a)
  _bernoulli r g a = tenGen r g $ shuffle3 Sig.c_bernoulli (hs2cAccReal a)
  _bernoulli_FloatTensor r g a = tenGenTen r g (snd $ TH.floatDynamicState a) Sig.c_bernoulli_FloatTensor
  _bernoulli_DoubleTensor r g a = tenGenTen r g (snd $ TH.doubleDynamicState a) Sig.c_bernoulli_DoubleTensor
  _uniform r g a b = tenGen r g $ shuffle3'2 Sig.c_uniform (hs2cAccReal a) (hs2cAccReal b)
  _normal r g a b = tenGen r g $ shuffle3'2 Sig.c_normal (hs2cAccReal a) (hs2cAccReal b)

  _normal_means :: Dynamic -> Generator -> Dynamic -> HsAccReal -> IO ()
  _normal_means r g m v = runManaged . joinIO $ Sig.c_normal_means
    <$> manage' Sig.dynamicStateRef r
    <*> manage' Sig.ctensor r
    <*> manage' Sig.rng g
    <*> manage' Sig.ctensor m
    <*> pure (Sig.hs2cAccReal v)

  _normal_stddevs :: Dynamic -> Generator -> HsAccReal -> Dynamic -> IO ()
  _normal_stddevs r g v m = runManaged . joinIO $ Sig.c_normal_stddevs
    <$> manage' Sig.dynamicStateRef r
    <*> manage' Sig.ctensor r
    <*> manage' Sig.rng g
    <*> pure (Sig.hs2cAccReal v)
    <*> manage' Sig.ctensor m

  _normal_means_stddevs :: Dynamic -> Generator -> Dynamic -> Dynamic -> IO ()
  _normal_means_stddevs r g a b = runManaged . joinIO $ Sig.c_normal_means_stddevs
    <$> manage' Sig.dynamicStateRef r
    <*> manage' Sig.ctensor r
    <*> manage' Sig.rng g
    <*> manage' Sig.ctensor a
    <*> manage' Sig.ctensor b

  _exponential :: Dynamic -> Generator -> HsAccReal -> IO ()
  _exponential r g v = runManaged . joinIO $ Sig.c_exponential
    <$> manage' Sig.dynamicStateRef r
    <*> manage' Sig.ctensor r
    <*> manage' Sig.rng g
    <*> pure (Sig.hs2cAccReal v)

  _standard_gamma :: Dynamic -> Generator -> Dynamic -> IO ()
  _standard_gamma r g m = runManaged . joinIO $ Sig.c_standard_gamma
    <$> manage' Sig.dynamicStateRef r
    <*> manage' Sig.ctensor r
    <*> manage' Sig.rng g
    <*> manage' Sig.ctensor m


  _cauchy :: Dynamic -> Generator -> HsAccReal -> HsAccReal -> IO ()
  _cauchy r g a b = runManaged . joinIO $ Sig.c_cauchy
    <$> manage' Sig.dynamicStateRef r
    <*> manage' Sig.ctensor r
    <*> manage' Sig.rng g
    <*> pure (Sig.hs2cAccReal a)
    <*> pure (Sig.hs2cAccReal b)

  _logNormal :: Dynamic -> Generator -> HsAccReal -> HsAccReal -> IO ()
  _logNormal r g a b = runManaged . joinIO $ Sig.c_logNormal
    <$> manage' Sig.dynamicStateRef r
    <*> manage' Sig.ctensor r
    <*> manage' Sig.rng g
    <*> pure (Sig.hs2cAccReal a)
    <*> pure (Sig.hs2cAccReal b)

  _multinomial :: LongDynamic -> Generator -> Dynamic -> Int -> Int -> IO ()
  _multinomial r g t a b = runManaged . joinIO $ Sig.c_multinomial
    <$> manage' (fst . Sig.longDynamicState) r
    <*> manage' (snd . Sig.longDynamicState) r
    <*> manage' Sig.rng g
    <*> manage' Sig.ctensor t
    <*> pure (fromIntegral a)
    <*> pure (fromIntegral b)

  _multinomialAliasSetup :: Dynamic -> LongDynamic -> Dynamic -> IO ()
  _multinomialAliasSetup r l t = runManaged . joinIO $ Sig.c_multinomialAliasSetup
    <$> manage' (Sig.dynamicStateRef) r
    <*> manage' (Sig.ctensor) r
    <*> manage' (snd . Sig.longDynamicState) l
    <*> manage' (Sig.ctensor) t

  _multinomialAliasDraw  :: LongDynamic -> Generator -> LongDynamic -> Dynamic -> IO ()
  _multinomialAliasDraw r g l t = runManaged . joinIO $ Sig.c_multinomialAliasDraw
    <$> manage' (fst . Sig.longDynamicState) r
    <*> manage' (snd . Sig.longDynamicState) r
    <*> manage' (Sig.rng) g
    <*> manage' (snd . Sig.longDynamicState) l
    <*> manage' (Sig.ctensor) t

