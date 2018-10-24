{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE NoImplicitPrelude #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeSynonymInstances #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE CPP #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Normalise #-}
{-# OPTIONS_GHC -fno-cse #-}
module Dense3
  ( y2cat
  , crossEntropyIO

  -- * test
  , FC3Arch
  , mkFC3
  , dense3BatchIO

  , linearBatchIO
  , linearBatchWithIO
  , reluIO
  , thresholdIO
  , softMaxBatch
  , softMaxBatchIO
  ) where

import Criterion

import Prelude
import qualified Prelude as P
import Control.Exception.Safe (throwString)
import Control.Monad
import Data.IORef
import Data.Maybe
import Data.Function ((&))
import Data.Singletons.Prelude (Fst, Snd)
import Data.Singletons.Prelude.List (Product)
import Foreign hiding (new)
import Lens.Micro ((^.))
import System.IO (stdout, hFlush)
import System.IO.Unsafe
import GHC.TypeLits
import Control.Concurrent
import Data.Singletons
import Data.Singletons.Prelude.Bool
import Torch.Double hiding (logSoftMaxBatch, conv2dMMBatch)

import Torch.Double.NN.Linear (Linear(..))

import Torch.Data.Loaders.Cifar10

import qualified Torch.Double.NN           as NN
import qualified Torch.Double.NN.Linear    as Linear
import qualified Torch.Double.NN.Conv2d    as Conv2d
import qualified Torch.Long                as Long
import qualified Torch.Long.Dynamic        as LDynamic
import qualified Torch.Double              as Torch
import qualified Torch.Double.Storage      as Storage
import qualified Torch.Long.Storage        as LStorage
import qualified Torch.Models.Vision.LeNet as Vision

import qualified Torch.Double.Dynamic as Dynamic
import qualified Torch.Double.Dynamic.NN as Dynamic
import qualified Torch.Double.Dynamic.NN.Criterion as Dynamic
import qualified Torch.Double.Dynamic.NN.Activation as Dynamic

y2cat :: Tensor '[4, 10] -> IO [Category]
y2cat ys = pure $
  fmap ((\i -> toEnum . fromIntegral . fromJust $ Long.get2d rez i 0)) [0..3]
  where
    rez = fromJust . snd $ Torch.max2d1 ys keep

type FC3Arch = (Linear (32*32*3) (32*32), Linear (32*32) 32, Linear 32 10)

mkFC3 :: IO FC3Arch
mkFC3 = do
  -- xavier initialization
  (,,)
    <$> pure (Linear (constant (1/(32*32)), constant (1/(32*32))))
    <*> pure (Linear (constant (1/32),      constant (1/32)))
    <*> pure (Linear (constant (1/10),      constant (1/10)))
{-
  g <- newRNG
  manualSeed g 1
  (,,)
    <$> (Linear <$> ((,) <$> uniform g rg <*> uniform g rg))
    <*> (Linear <$> ((,) <$> uniform g rg <*> uniform g rg))
    <*> (Linear <$> ((,) <$> uniform g rg <*> uniform g rg))
 where
  Just rg = ord2Tuple (-1, 1)
-}

dense3BatchIO
  :: HsReal
  -> FC3Arch
  ->    (Tensor '[4, 32*32*3])  -- ^ input
  -> IO (Tensor '[4, 10], Tensor '[4, 10] -> IO (FC3Arch, Tensor '[4, 32*32*3]))
dense3BatchIO lr (l1, l2, l3) i = do
  (fc1out, fc1getgrad) <- linearBatchIO 1 l1 i
  ( r1out, unrelu1)    <- reluIO fc1out

  (fc2out, fc2getgrad) <- linearBatchIO 1 l2 r1out
  ( r2out, unrelu2)    <- reluIO fc2out

  (fc3out, fc3getgrad) <- linearBatchIO 1 l3 r2out
  (fin, smgrads)       <- softMaxBatch fc3out

  pure (fin, \gout -> do
    smg <- smgrads gout
    (fc3g, fc3gin) <- fc3getgrad smg
    (fc2g, fc2gin) <- fc2getgrad =<< unrelu2 fc3gin
    (fc1g, fc1gin) <- fc1getgrad =<< unrelu1 fc2gin

    pure ((fc1g, fc2g, fc3g), fc1gin))


-- ========================================================================= --

-- | ReLU activation function
reluIO :: Dimensions d => Tensor d -> IO (Tensor d, Tensor d -> IO (Tensor d))
reluIO = thresholdIO 0 0

linearBatchIO
  :: forall i o b
   . All KnownDim '[b,i,o]
  => HsReal
  -> (Linear i o)
  -> (Tensor '[b, i])
  -> IO (Tensor '[b, o], Tensor '[b, o] -> IO ((Linear i o),  (Tensor '[b, i])))     --- by "simple autodifferentiation", I am seeing that this is a fork
linearBatchIO = linearBatchWithIO Nothing Nothing Nothing

-- | 'linear' with a batch dimension
linearBatchWithIO
  :: forall i o b
   . All KnownDim '[b,i,o]
  => Maybe (Tensor '[b, o])       -- output buffer. currently mutable.
  -> Maybe (Tensor '[b, i])       -- gradin buffer. currently mutable.
  -> Maybe (Linear i o)           -- gradparam buffer. currently mutable.

  -> HsReal

  -> (Linear i o)
  -> (Tensor '[b, i])
  -> IO (Tensor '[b, o], Tensor '[b, o] -> IO ((Linear i o),  (Tensor '[b, i])))     --- by "simple autodifferentiation", I am seeing that this is a fork
linearBatchWithIO moutbuffer mgradinbuf mgradparambuf lr l i = do
  let o = updateOutput i l
  pure (o, \gout -> do
    let g@(Linear (gw, gb)) = accGradParameters i gout l
    let gin = updateGradInput i gout (Linear.weights l)
    pure (g, gin))
   where
    updateOutput :: Tensor '[b, i] -> Linear i o -> Tensor '[b, o]
    updateOutput i (Linear (w,b)) =
      let
        o = addmm 1 (constant 0) 1 i w
      in
        addr 1 o 1 (constant 1) b

    updateGradInput :: Tensor '[b, i] -> Tensor '[b, o] -> Tensor '[i,o] -> Tensor '[b, i]
    updateGradInput i gout w = addmm 0 (constant 0) 1 gout (transpose2d w)

    accGradParameters :: Tensor '[b,i] -> Tensor '[b,o] -> Linear i o -> Linear i o
    accGradParameters i gout (Linear (w, b)) = Linear (gw, gb)
      where
        gw :: Tensor '[i, o]
        gw = addmm 1 (constant 0) lr (transpose2d i) gout

        gb :: Tensor '[o]
        gb = addmv 1 (constant 0) lr tgout (constant 1)

        tgout :: Tensor '[o,b]
        tgout = transpose2d gout


-- ========================================================================= --

-- | run a threshold function againts two BVar variables
softMaxBatch
  :: Tensor '[4, 10]    -- ^ input
  -> IO (Tensor '[4, 10], Tensor '[4, 10] -> IO (Tensor '[4, 10]))   -- ^ output and gradient
softMaxBatch = softMaxBatchIO (Just new)

-- | run a threshold function againts two BVar variables
softMaxBatchIO
  -- cachable buffers
  :: Maybe (Tensor '[4, 10])

  -> Tensor '[4, 10]    -- ^ input
  -> IO (Tensor '[4, 10], Tensor '[4, 10] -> IO (Tensor '[4, 10]))   -- ^ output and gradient
softMaxBatchIO mgin inp = do
  let out = constant 0
  updateOutput_ inp i out
  pure (out, \gout -> do
    let gin = constant 0
    updateGradInput_ inp gout out i gin
    pure gin
    )

 where
  i = (dim :: Dim 1)

  updateOutput_ :: Tensor '[4, 10] -> Dim 1 -> Tensor '[4, 10] -> IO ()
  updateOutput_ inp i out =
    Dynamic._softMax_updateOutput (asDynamic inp) (asDynamic out) (fromIntegral $ dimVal i)

  updateGradInput_
    :: Tensor '[4, 10]  -- input
    -> Tensor '[4, 10]  -- gradOutput
    -> Tensor '[4, 10]  -- output
    -> Dim 1            -- dimension

    -> Tensor '[4, 10]  -- gradInput
    -> IO ()
  updateGradInput_ inp gout out i gin =
    Dynamic._softMax_updateGradInput
      (asDynamic inp)             -- input
      (asDynamic gout)            -- gradOutput
      (asDynamic gin)             -- gradInput
      (asDynamic out)             -- output
      (fromIntegral $ dimVal i)   -- dimension


