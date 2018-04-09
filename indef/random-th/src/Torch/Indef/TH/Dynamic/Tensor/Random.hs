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
  random_ t g = tenGen t g Sig.c_random
  clampedRandom_ r g a b = tenGen r g $ shuffle3'2 Sig.c_clampedRandom (fromIntegral a) (fromIntegral b)
  cappedRandom_ r g a = tenGen r g $ shuffle3 Sig.c_cappedRandom (fromIntegral a)
  geometric_ r g a = tenGen r g $ shuffle3 Sig.c_geometric (hs2cAccReal a)
  bernoulli_ r g a = tenGen r g $ shuffle3 Sig.c_bernoulli (hs2cAccReal a)
  bernoulli_FloatTensor_ r g a = tenGenTen r g (snd $ TH.floatDynamicState a) Sig.c_bernoulli_FloatTensor
  bernoulli_DoubleTensor_ r g a = tenGenTen r g (snd $ TH.doubleDynamicState a) Sig.c_bernoulli_DoubleTensor
  uniform_ r g a b = tenGen r g $ shuffle3'2 Sig.c_uniform (hs2cAccReal a) (hs2cAccReal b)
  normal_ r g a b = tenGen r g $ shuffle3'2 Sig.c_normal (hs2cAccReal a) (hs2cAccReal b)

  normal_means_ :: Dynamic -> Generator -> Dynamic -> HsAccReal -> IO ()
  normal_means_ r g m v = runManaged . joinIO $ Sig.c_normal_means
    <$> manage' Sig.dynamicStateRef r
    <*> manage' Sig.ctensor r
    <*> manage' Sig.rng g
    <*> manage' Sig.ctensor m
    <*> pure (Sig.hs2cAccReal v)

  normal_stddevs_ :: Dynamic -> Generator -> HsAccReal -> Dynamic -> IO ()
  normal_stddevs_ r g v m = runManaged . joinIO $ Sig.c_normal_stddevs
    <$> manage' Sig.dynamicStateRef r
    <*> manage' Sig.ctensor r
    <*> manage' Sig.rng g
    <*> pure (Sig.hs2cAccReal v)
    <*> manage' Sig.ctensor m

  normal_means_stddevs_ :: Dynamic -> Generator -> Dynamic -> Dynamic -> IO ()
  normal_means_stddevs_ r g a b = runManaged . joinIO $ Sig.c_normal_means_stddevs
    <$> manage' Sig.dynamicStateRef r
    <*> manage' Sig.ctensor r
    <*> manage' Sig.rng g
    <*> manage' Sig.ctensor a
    <*> manage' Sig.ctensor b

  exponential_ :: Dynamic -> Generator -> HsAccReal -> IO ()
  exponential_ r g v = runManaged . joinIO $ Sig.c_exponential
    <$> manage' Sig.dynamicStateRef r
    <*> manage' Sig.ctensor r
    <*> manage' Sig.rng g
    <*> pure (Sig.hs2cAccReal v)

  standard_gamma_ :: Dynamic -> Generator -> Dynamic -> IO ()
  standard_gamma_ r g m = runManaged . joinIO $ Sig.c_standard_gamma
    <$> manage' Sig.dynamicStateRef r
    <*> manage' Sig.ctensor r
    <*> manage' Sig.rng g
    <*> manage' Sig.ctensor m


  cauchy_ :: Dynamic -> Generator -> HsAccReal -> HsAccReal -> IO ()
  cauchy_ r g a b = runManaged . joinIO $ Sig.c_cauchy
    <$> manage' Sig.dynamicStateRef r
    <*> manage' Sig.ctensor r
    <*> manage' Sig.rng g
    <*> pure (Sig.hs2cAccReal a)
    <*> pure (Sig.hs2cAccReal b)

  logNormal_ :: Dynamic -> Generator -> HsAccReal -> HsAccReal -> IO ()
  logNormal_ r g a b = runManaged . joinIO $ Sig.c_logNormal
    <$> manage' Sig.dynamicStateRef r
    <*> manage' Sig.ctensor r
    <*> manage' Sig.rng g
    <*> pure (Sig.hs2cAccReal a)
    <*> pure (Sig.hs2cAccReal b)

  multinomial_ :: LongDynamic -> Generator -> Dynamic -> Int -> Int -> IO ()
  multinomial_ r g t a b = runManaged . joinIO $ Sig.c_multinomial
    <$> manage' (fst . Sig.longDynamicState) r
    <*> manage' (snd . Sig.longDynamicState) r
    <*> manage' Sig.rng g
    <*> manage' Sig.ctensor t
    <*> pure (fromIntegral a)
    <*> pure (fromIntegral b)

  multinomialAliasSetup_ :: Dynamic -> LongDynamic -> Dynamic -> IO ()
  multinomialAliasSetup_ r l t = runManaged . joinIO $ Sig.c_multinomialAliasSetup
    <$> manage' (Sig.dynamicStateRef) r
    <*> manage' (Sig.ctensor) r
    <*> manage' (snd . Sig.longDynamicState) l
    <*> manage' (Sig.ctensor) t

  multinomialAliasDraw_  :: LongDynamic -> Generator -> LongDynamic -> Dynamic -> IO ()
  multinomialAliasDraw_ r g l t = runManaged . joinIO $ Sig.c_multinomialAliasDraw
    <$> manage' (fst . Sig.longDynamicState) r
    <*> manage' (snd . Sig.longDynamicState) r
    <*> manage' (Sig.rng) g
    <*> manage' (snd . Sig.longDynamicState) l
    <*> manage' (Sig.ctensor) t

