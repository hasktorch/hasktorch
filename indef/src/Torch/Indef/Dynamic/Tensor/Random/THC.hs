-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Indef.Dynamic.Tensor.Random.THC
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
--
-- GPU-backed random functions. The difference between this package and
-- 'Torch.Indef.Dynamic.Tensor.Random.TH' is that these functions do not get
-- passed an explicit 'Generator' argument.
-------------------------------------------------------------------------------
module Torch.Indef.Dynamic.Tensor.Random.THC
  ( _random
  , _clampedRandom
  , _cappedRandom
  , _bernoulli
  , _bernoulli_DoubleTensor
  , _geometric
  , _uniform
  , _normal
  , _normal_means
  , _normal_stddevs
  , _normal_means_stddevs
  , _logNormal
  , _exponential
  , _cauchy
  , _multinomial
  , _multinomialAliasSetup
  , _multinomialAliasDraw
  , _rand
  , _randn


  , OpenUnit, openUnit, openUnitValue
  , ClosedUnit, closedUnit, closedUnitValue
  , Positive, positive, positiveValue
  , Ord2Tuple, ord2Tuple, ord2TupleValues
  ) where

import Control.Monad.Managed (runManaged)

import Torch.Indef.Types
import Torch.Indef.Dynamic.Tensor.Random.TH
  ( OpenUnit, openUnit, openUnitValue
  , ClosedUnit, closedUnit, closedUnitValue
  , Positive, positive, positiveValue
  , Ord2Tuple, ord2Tuple, ord2TupleValues
  )

import qualified Torch.Sig.Tensor.Random.THC as Sig
import qualified Torch.Sig.Types as Sig
import qualified Torch.Sig.Types.Global as Sig
import qualified Torch.Types.TH as TH

-- These import is for haddocks:
import qualified Torch.Indef.Dynamic.Tensor.Random.TH as THRandom
import qualified Torch.Indef.Dynamic.Tensor.Math.Random.TH as THRandomMath

-- | CUDA version of 'THRandom._random'
_random :: Dynamic -> IO ()
_random t = withDynamicState t Sig.c_random

-- | CUDA version of 'THRandom._clampedRandom'
_clampedRandom :: Dynamic -> Integer -> Integer -> IO ()
_clampedRandom t mn mx = withDynamicState t (shuffle2'2 Sig.c_clampedRandom (fromIntegral mn) (fromIntegral mx))

-- | CUDA version of 'THRandom._cappedRandom'
_cappedRandom :: Dynamic -> Integer -> IO ()
_cappedRandom t c = withDynamicState t (shuffle2 Sig.c_cappedRandom (fromIntegral c))

-- | CUDA version of 'THRandom._bernoulli'
_bernoulli :: Dynamic -> HsAccReal -> IO ()
_bernoulli t v = withDynamicState t (shuffle2 Sig.c_bernoulli (hs2cAccReal v))

-- | CUDA version of 'THRandom._bernoulli_DoubleTensor'
_bernoulli_DoubleTensor :: Dynamic -> Dynamic -> IO ()
_bernoulli_DoubleTensor t d = with2DynamicState t d Sig.c_bernoulli_DoubleTensor

-- | CUDA version of 'THRandom._geometric'
_geometric :: Dynamic -> HsAccReal -> IO ()
_geometric t v = withDynamicState t (shuffle2 Sig.c_geometric (hs2cAccReal v))

-- | CUDA version of 'THRandom._uniform'
_uniform :: Dynamic -> Ord2Tuple HsAccReal -> IO ()
_uniform t tup = withDynamicState t (shuffle2'2 Sig.c_uniform (hs2cAccReal a) (hs2cAccReal b))
  where
    (a, b) = ord2TupleValues tup

-- | CUDA version of 'THRandom._normal'
_normal :: Dynamic -> HsAccReal -> Positive HsAccReal -> IO ()
_normal r a b = runManaged . joinIO $ Sig.c_normal
   <$> manage' (fst . Sig.dynamicState) r
   <*> manage' (snd . Sig.dynamicState) r
   <*> pure (hs2cAccReal a)
   <*> pure (hs2cAccReal $ positiveValue b)

-- | CUDA version of 'THRandom._normal_means'
_normal_means :: Dynamic -> Dynamic -> Positive HsAccReal -> IO ()
_normal_means r m v = runManaged . joinIO $ Sig.c_normal_means
  <$> manage' Sig.dynamicStateRef r
  <*> manage' Sig.ctensor r
  <*> manage' Sig.ctensor m
  <*> pure (Sig.hs2cAccReal $ positiveValue v)

-- | CUDA version of 'THRandom._normal_stddevs'
_normal_stddevs :: Dynamic -> HsAccReal -> Dynamic -> IO ()
_normal_stddevs t a b = with2DynamicState t b $ \s' t' b' -> Sig.c_normal_stddevs s' t' (hs2cAccReal a) b'

-- | CUDA version of 'THRandom._normal_means_stddevs'
_normal_means_stddevs :: Dynamic -> Dynamic -> Dynamic -> IO ()
_normal_means_stddevs t t0 t1 = with3DynamicState t t0 t1 Sig.c_normal_means_stddevs

-- | call C-level @logNormal@
-- | CUDA version of 'THRandom._logNormal'
_logNormal :: Dynamic -> HsAccReal -> Positive HsAccReal -> IO ()
_logNormal r a b = runManaged . joinIO $ Sig.c_logNormal
  <$> manage' Sig.dynamicStateRef r
  <*> manage' Sig.ctensor r
  <*> pure (Sig.hs2cAccReal a)
  <*> pure (Sig.hs2cAccReal $ positiveValue b)


-- | CUDA version of 'THRandom._exponential'
_exponential :: Dynamic -> HsAccReal -> IO ()
_exponential t v = withDynamicState t (shuffle2 Sig.c_exponential (hs2cAccReal v))

-- | CUDA version of 'THRandom._cauchy'
_cauchy :: Dynamic -> HsAccReal -> HsAccReal -> IO ()
_cauchy t a b = withDynamicState t (shuffle2'2 Sig.c_cauchy (hs2cAccReal a) (hs2cAccReal b))

-- | CUDA version of 'THRandom._multinomial'
_multinomial :: IndexDynamic -> Dynamic -> Int -> Int -> IO ()
_multinomial r t a b = runManaged . joinIO $ Sig.c_multinomial
   <$> manage' (fst . Sig.longDynamicState) r
   <*> manage' (snd . Sig.longDynamicState) r
   <*> manage' Sig.ctensor t
   <*> pure (fromIntegral a)
   <*> pure (fromIntegral b)

-- | CUDA version of 'THRandom._multinomialAliasSetup'
_multinomialAliasSetup :: Dynamic -> LongDynamic -> Dynamic -> IO ()
_multinomialAliasSetup r l t = runManaged . joinIO $ Sig.c_multinomialAliasSetup
   <$> manage' (Sig.dynamicStateRef) r
   <*> manage' (Sig.ctensor) r
   <*> manage' (snd . Sig.longDynamicState) l
   <*> manage' (Sig.ctensor) t

-- | CUDA version of 'THRandom._multinomialAliasDraw'
_multinomialAliasDraw  :: LongDynamic -> LongDynamic -> Dynamic -> IO ()
_multinomialAliasDraw r l t = runManaged . joinIO $ Sig.c_multinomialAliasDraw
   <$> manage' (fst . Sig.longDynamicState) r
   <*> manage' (snd . Sig.longDynamicState) r
   <*> manage' (snd . Sig.longDynamicState) l
   <*> manage' (Sig.ctensor) t

-- | CUDA version of 'THRandomMath._rand'
_rand  :: Dynamic -> TH.LongStorage -> IO ()
_rand r l = runManaged . joinIO $ Sig.c_rand
   <$> manage' (Sig.dynamicStateRef) r
   <*> manage' (Sig.ctensor) r
   <*> manage' (snd . TH.longStorageState) l

-- | CUDA version of 'THRandomMath._randn'
_randn  :: Dynamic -> TH.LongStorage -> IO ()
_randn r l = runManaged . joinIO $ Sig.c_randn
  <$> manage' (Sig.dynamicStateRef) r
  <*> manage' (Sig.ctensor) r
  <*> manage' (snd . TH.longStorageState) l


