{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE CPP #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE Strict #-}
{- OPTIONS_GHC -fplugin GHC.TypeLits.Normalise #-}
module Main where

import Prelude
import Utils hiding (lr, bs, bsz)
import LeNet
-- import LeNet.Forward


import Data.Maybe
import Data.Time (getCurrentTime)
import Data.List.NonEmpty (NonEmpty)
import Debug.Trace
import Text.Printf
import GHC.TypeLits (KnownNat)
import Control.Exception.Safe (throwString)
import Control.Monad.Trans.Except
import System.IO (hFlush, stdout)
import System.IO.Unsafe
import qualified Data.Vector as V
import qualified Data.List.NonEmpty as NE

#ifdef CUDA
import Torch.Cuda.Double -- (Tensor, HsReal, (.:), mSECriterion)
import qualified Torch.Cuda.Double as D
import qualified Torch.Cuda.Long as Long
#else
import Torch.Double -- (Tensor, HsReal, (.:), mSECriterion)
import qualified Torch.Double as D
import qualified Torch.Long as Long
#endif

import Torch.Models.Vision.LeNet hiding (LeNet)
import Torch.Data.Loaders.Cifar10
import Torch.Data.OneHot (onehotf)
import qualified Torch.Double.NN.Linear    as Linear
import qualified System.Random.MWC as MWC


lr = 1
type BatchSize = 4
bsz = (dim :: Dim BatchSize)

bs :: Num n => n
bs = (fromIntegral $ dimVal bsz)

main :: IO ()
main = seedAll >>= \g -> do
-- main = MWC.withSystemRandom $ \g -> do
  ltrain <- prepdata . V.take bs <$> cifar10set g default_cifar_path Train
  tg <- newRNG
  net0 :: LeNet
#ifdef LENET_HEAD_ONLY
    <- mkFC3
#else
    <- newLeNet tg
#endif
  print net0

  t0 <- getCurrentTime

  let
    lbatch :: LDataSet
    [lbatch] = mkBatches (fromIntegral bs) ltrain

  putStrLn "looking at lazily loaded data:"
  print lbatch

  let
    btensor :: LDataSet -> IO (Maybe (Tensor '[BatchSize, 10], Tensor '[BatchSize, 3, 32, 32]))
    btensor lds
      | V.length lds /= (fromIntegral bs) = pure Nothing
      | otherwise = do
        foo <- NE.fromList . V.toList <$> V.mapM getdata lds
        ys <- toYs foo
        xs <- toXs foo
        pure $ Just (ys, xs)

  putStrLn "starting to load"
  fcd <- mapM forcetensor lbatch
  putStrLn "done loading!"
  btensor fcd >>= \case
    Nothing -> throwString "error loading data!"
    Just (ys, xs) -> do
      putStrLn "running first batch..."
      printf ("\ntruth      : %s") (show (t2cat ys))

      pred0 <- t2cat <$> forward net0 xs
      printf ("\nprediction0: %s") (show pred0)

      (net', !loss) <- trainStep True lr net0 xs ys
      let l = getloss loss

      pred1 <- t2cat <$> forward net' xs

      printf "\nprediction1: %s" (show pred1)

      (_, !loss') <- trainStep False lr net' xs ys
      let l' = getloss loss'
      pred' <- t2cat <$> forward net' xs

      printf ("\nimprovement? %.4f -> %.4f : %s\n") l l' (show (l > l'))


  printf "\nFinished single-batch training!\n"

toYs :: NonEmpty (Category, Tensor '[3, 32, 32]) -> IO (Tensor '[BatchSize, 10])
toYs = D.unsafeMatrix . fmap (onehotf . fst) . NE.toList

toXs :: NonEmpty (Category, Tensor '[3, 32, 32]) -> IO (Tensor '[BatchSize, 3, 32, 32])
toXs xs = case D.catArray0 (fmap (D.unsqueeze1d (dim :: Dim 0) . snd) xs) of
  Left  s -> throwString s
  Right t -> pure t

t2cat :: Tensor '[BatchSize, 10] -> [Category]
t2cat = (fmap (toEnum . fromIntegral) . Long.tensordata . fromJust . snd . flip max2d1 keep)

#ifdef LENET_HEAD_ONLY
forward arch xs = do
  fst <$> ff3Batch False 1 arch (resizeAs xs)

trainStep
  :: Bool
  -> HsReal
  -> FC3Arch
  -> Tensor '[BatchSize, 3, 32, 32]
  -> Tensor '[BatchSize, 10]
  -> IO (FC3Arch, Tensor '[1])
trainStep istraining lr net@(l1, l2, l3) xs ys = do
  (ff3out, getff3grads) <- ff3Batch istraining 1 net (resizeAs xs)
  (loss, getCEgrad) <- crossentropy (Long.resizeAs . fromJust . snd $ max2d1 ys keep) ff3out
  putStrLn "\nLOSS"
  print loss
  pure (if not istraining then undefined else unsafePerformIO $ do
    (g1, g2, g3) <- fmap fst . getff3grads =<< getCEgrad loss
    let l1' = (l1 + g1 ^* (-lr))
    let l2' = (l2 + g2 ^* (-lr))
    let l3' = (l3 + g3 ^* (-lr))
    pure (l1', l2', l3')
    , loss)

#else
forward arch xs = do
  undefined
  -- lenetBatchForward arch xs

trainStep
  :: Bool
  -> HsReal
  -> LeNet
  -> Tensor '[BatchSize, 3, 32, 32]
  -> Tensor '[BatchSize, 10]
  -> IO (LeNet, Tensor '[1])
trainStep istraining lr net xs ys = do
  (out, gnet) <- undefined -- lenetBatchBP net (Long.resizeAs . fromJust . snd $ max2d1 ys keep) xs
  let Just plr = positive 1.0
  putStrLn "\nLOSS"
  print out
  pure (if not istraining then undefined else unsafePerformIO (myupdate net (plr,gnet)), out)
#endif

