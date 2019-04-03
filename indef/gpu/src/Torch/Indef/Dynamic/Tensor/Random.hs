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
  , Ord2Tuple, ord2Tuple, ord2TupleValue
  ) where

import Control.Monad.Managed (runManaged, managed)
import Foreign (withForeignPtr)
import Control.Monad.IO.Class (liftIO)

import Torch.Indef.Types
import Torch.Indef.Dynamic.Tensor.Random.TH
  ( OpenUnit, openUnit, openUnitValue
  , ClosedUnit, closedUnit, closedUnitValue
  , Positive, positive, positiveValue
  , Ord2Tuple, ord2Tuple, ord2TupleValue
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
_random t = runManaged $ do
  s' <- managedState
  t' <- managedTensor t
  liftIO $ Sig.c_random s' t'

-- | CUDA version of 'THRandom._clampedRandom'
_clampedRandom :: Dynamic -> Integer -> Integer -> IO ()
_clampedRandom t mn mx = runManaged $ do
  s' <- managedState
  t' <- managedTensor t
  liftIO $ Sig.c_clampedRandom s' t' (fromIntegral mn) (fromIntegral mx)

-- | CUDA version of 'THRandom._cappedRandom'
_cappedRandom :: Dynamic -> Integer -> IO ()
_cappedRandom t c = runManaged $ do
  s' <- managedState
  t' <- managedTensor t
  liftIO $ Sig.c_cappedRandom s' t' (fromIntegral c)

-- | CUDA version of 'THRandom._bernoulli'
_bernoulli :: Dynamic -> HsAccReal -> IO ()
_bernoulli t v = runManaged $ do
  s' <- managedState
  t' <- managedTensor t
  liftIO $ Sig.c_bernoulli s' t' (hs2cAccReal v)

-- | CUDA version of 'THRandom._bernoulli_DoubleTensor'
_bernoulli_DoubleTensor :: Dynamic -> Dynamic -> IO ()
_bernoulli_DoubleTensor t d = with2DynamicState t d Sig.c_bernoulli_DoubleTensor

-- | CUDA version of 'THRandom._geometric'
_geometric :: Dynamic -> HsAccReal -> IO ()
_geometric t v = runManaged $ do
  s' <- managedState
  t' <- managedTensor t
  liftIO $ Sig.c_geometric s' t' (hs2cAccReal v)

-- | CUDA version of 'THRandom._uniform'
_uniform :: Dynamic -> Ord2Tuple HsAccReal -> IO ()
_uniform t tup = runManaged $ do
  s' <- managedState
  t' <- managedTensor t
  liftIO $ Sig.c_uniform s' t' (hs2cAccReal a) (hs2cAccReal b)
  where
    (a, b) = ord2TupleValue tup

-- | CUDA version of 'THRandom._normal'
_normal :: Dynamic -> HsAccReal -> Positive HsAccReal -> IO ()
_normal r a b = runManaged . (liftIO =<<) $ Sig.c_normal
   <$> managedState
   <*> managedTensor r
   <*> pure (hs2cAccReal a)
   <*> pure (hs2cAccReal $ positiveValue b)

-- | CUDA version of 'THRandom._normal_means'
_normal_means :: Dynamic -> Dynamic -> Positive HsAccReal -> IO ()
_normal_means r m v = runManaged . (liftIO =<<) $ Sig.c_normal_means
  <$> managedState
  <*> managedTensor r
  <*> managedTensor m
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
_logNormal r a b = runManaged . (liftIO =<<) $ Sig.c_logNormal
  <$> managedState
  <*> managedTensor r
  <*> pure (Sig.hs2cAccReal a)
  <*> pure (Sig.hs2cAccReal $ positiveValue b)


-- | CUDA version of 'THRandom._exponential'
_exponential :: Dynamic -> HsAccReal -> IO ()
_exponential t v = runManaged . (liftIO =<<) $ Sig.c_exponential
  <$> managedState
  <*> managedTensor t
  <*> pure (hs2cAccReal v)

-- | CUDA version of 'THRandom._cauchy'
_cauchy :: Dynamic -> HsAccReal -> HsAccReal -> IO ()
_cauchy t a b = runManaged . (liftIO =<<) $ Sig.c_cauchy
  <$> managedState
  <*> managedTensor t
  <*> pure (hs2cAccReal a)
  <*> pure (hs2cAccReal b)

-- | CUDA version of 'THRandom._multinomial'
_multinomial :: IndexDynamic -> Dynamic -> Int -> Int -> IO ()
_multinomial r t a b = runManaged . (liftIO =<<) $ Sig.c_multinomial
   <$> managedState
   <*> managed (withForeignPtr . snd . Sig.longDynamicState $ r)
   <*> managedTensor t
   <*> pure (fromIntegral a)
   <*> pure (fromIntegral b)

-- | CUDA version of 'THRandom._multinomialAliasSetup'
_multinomialAliasSetup :: Dynamic -> LongDynamic -> Dynamic -> IO ()
_multinomialAliasSetup r l t = runManaged . (liftIO =<<) $ Sig.c_multinomialAliasSetup
   <$> managedState
   <*> managedTensor r
   <*> managed (withForeignPtr . snd . Sig.longDynamicState $ l)
   <*> managedTensor t

-- | CUDA version of 'THRandom._multinomialAliasDraw'
_multinomialAliasDraw  :: LongDynamic -> LongDynamic -> Dynamic -> IO ()
_multinomialAliasDraw r l t = runManaged . (liftIO =<<) $ Sig.c_multinomialAliasDraw
   <$> managedState
   <*> managed (withForeignPtr . snd . Sig.longDynamicState $ r)
   <*> managed (withForeignPtr . snd . Sig.longDynamicState $ l)
   <*> managedTensor t

-- | CUDA version of 'THRandomMath._rand'
_rand  :: Dynamic -> TH.LongStorage -> IO ()
_rand r l = runManaged . (liftIO =<<) $ Sig.c_rand
   <$> managedState
   <*> managedTensor r
   <*> managed (withForeignPtr . snd . TH.longStorageState $ l)

-- | CUDA version of 'THRandomMath._randn'
_randn  :: Dynamic -> TH.LongStorage -> IO ()
_randn r l = runManaged . (liftIO =<<) $ Sig.c_randn
  <$> managedState
  <*> managedTensor r
  <*> managed (withForeignPtr . snd . TH.longStorageState $ l)


