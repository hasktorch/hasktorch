{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE NoImplicitPrelude #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Normalise #-}
module Main where

import Prelude
import qualified Prelude as P
import LeNet (maxPooling2dBatchIO, conv2dBatchIO, LeNet, myupdate)
import Utils (getloss)
import DataLoading (mkVBatches, dataloader')
import Dense3 (softMaxBatch, linearBatchIO, reluIO, y2cat)
import Criterion (crossEntropyIO)
import Control.Exception.Safe

-------------------------------------------------------------------------------

import Data.Vector (Vector)
import Data.DList (DList)
import Data.Maybe (fromJust)
import Data.List (genericLength)
import Data.Singletons (sing)
import Data.Singletons.Prelude.Bool (SBool)
import Data.Singletons.Prelude.List (Product)
import Data.Time (getCurrentTime, diffUTCTime)
import Control.Arrow (second)
import Control.Monad (forM, forM_, when)
import Lens.Micro ((^.))
import System.IO (stdout, hFlush)
import System.IO.Unsafe (unsafePerformIO)
import System.Mem (performGC, performMajorGC)
import Text.Printf (printf)

import qualified Data.List.NonEmpty as NE
import qualified Data.HashMap.Strict as HM
import qualified Data.DList as DL
import qualified Data.Vector as V
import qualified System.Random.MWC as MWC

-------------------------------------------------------------------------------

import Torch.Data.Loaders.Cifar10 (Category(..), Mode(..), cifar10set, default_cifar_path)
import Torch.Data.Loaders.RGBVector (Normalize(..))
import Torch.Data.OneHot (onehotf)
import Torch.Double

import qualified Torch.Long as Long
import qualified Torch.Long.Dynamic as LDyn
import qualified Torch.Double.NN.Conv2d as Conv2d
import qualified Torch.Double as Torch
import qualified Torch.Models.Vision.LeNet as Vision

type Arch = LeNet
mkNet = Vision.newLeNet
-- netforward = LeNet.lenetBatch undefined undefined
netupdate = LeNet.myupdate

-- type Arch = NIN
-- mkNet = mkNIN
-- forward = ninForwardBatch
-- netupdate = ninUpdate
-- netBatchBP = ninBatchBP

Just plr = positive 0.3

datasize = 400
reportat' = 399
epos = 30
type BatchSize = 4
bsz :: Dim BatchSize
bsz = dim
bs :: Integral i => i
bs = fromIntegral (dimVal bsz)

main :: IO ()
main = do
  g <- MWC.initialize (V.singleton 44)
  tg <- newRNG

  lbatches :: Vector (Vector (Category, FilePath))
    <- mkVBatches 4 . V.take datasize <$> cifar10set g default_cifar_path Train
  printf "\nloaded %i filepaths" (length lbatches * 4)
  putStrLn "\ntransforming to tensors"
  batches <- dataloader' bsz lbatches

  net0 <- mkNet tg
  print net0

  net <- epochs plr epos batches net0

  putStrLn "reporting:"
  report net batches
  putStrLn ""

  where
    report :: Arch -> Vector (Tensor '[4, 10], Tensor '[4, 3, 32, 32]) -> IO ()
    report net ltest = do
      foo :: [DList (Category, Category)]
        <- forM (V.toList ltest) $ \(y, x) -> do
          bar <- y2cat y
          DL.fromList . zip bar . t2cat <$> forward net x

      let
        test :: [(Category, Category)]
        test = DL.toList (DL.concat foo)

      let
        cathm :: [(Category, [Category])]
        cathm = HM.toList . HM.fromListWith (++) $ (second (:[]) <$> test)

        totalacc :: Float
        totalacc = genericLength (filter (uncurry (==)) test) / (genericLength test)

      printf ("[test accuracy: %.2f%% / %d]") (100*totalacc) (length test)

      forM_ cathm $ \(y, preds) -> do
        let
          correct = length (filter (==y) preds)
          acc = fromIntegral correct / genericLength preds :: Float
        printf "\n[%s]:\t%.2f%% (%d / %d)" (show y) (acc*100) correct (length preds)
        hFlush stdout


-- ========================================================================= --
-- Training on a dataset
-- ========================================================================= --
epochs
  :: Positive HsReal     -- ^ learning rate
  -> Int                 -- number of epochs to train on
  -> Vector (Tensor '[BatchSize, 10], Tensor '[BatchSize, 3, 32, 32])
  -> Arch               -- ^ initial model
  -> IO Arch            -- ^ final model
epochs plr mx batches = runEpoch 1
  where
    runEpoch :: Int -> Arch -> IO Arch
    runEpoch e net
      | e > mx = P.putStrLn "\nfinished training loops" >> pure net
      | otherwise = do
      t0 <- getCurrentTime
      (net', losstot) <- V.ifoldM go (net, 0) batches
      t1 <- getCurrentTime
      let diff = realToFrac (t1 `diffUTCTime` t0) :: Float
      printf ("%s[avg_cross_entropy %.8f] %.0fs\n") estr (losstot / fromIntegral (V.length batches)) diff
      runEpoch (e+1) net'
      where
        estr :: String
        estr = "[Epoch "++ show e ++ "/"++ show mx ++ "]"

        go :: (Arch, HsReal) -> Int -> (Tensor '[4, 10], Tensor '[4, 3, 32, 32]) -> IO (Arch, HsReal)
        go (!net, !runningloss) !bid (ys, xs) = do
          (loss, net') <- backward plr net xs ys
          pure (net', runningloss + getloss loss)

          -- ********* This is for intra-epoch reporting *********
          -- let reportat = reportat' :: Int
          --     reporttime = (bid `mod` reportat == (reportat - 1))

          -- when reporttime $ do
          --   printf ("\n%s(%db#%03d)[ce %.4f]") estr
          --     (bs::Int) (bid+1)
          --     (runningloss / (if reportat == 1 then 1 else (fromIntegral reportat - 1)))
          --   hFlush stdout

          -- performGC

          -- let l = getloss loss
          -- -- P.print l
          -- pure (net', if reporttime then 0 else runningloss + l)
          -- *****************************************************

toYs :: [(Category, Tensor '[3, 32, 32])] -> IO (Tensor '[BatchSize, 10])
toYs = Torch.unsafeMatrix . fmap (onehotf . fst)

toXs :: [(Category, Tensor '[3, 32, 32])] -> IO (Tensor '[BatchSize, 3, 32, 32])
toXs xs =
  case Torch.catArray0 $ fmap (Torch.unsqueeze1d (dim :: Dim 0) . snd) (NE.fromList xs) of
    Left s -> throwString s
    Right t -> pure t

t2cat :: Tensor '[BatchSize, 10] -> [Category]
t2cat = (fmap (toEnum . fromIntegral) . Long.tensordata . fromJust . snd . flip max2d1 keep)

forward :: Arch -> Tensor '[BatchSize, 3, 32, 32] -> IO (Tensor '[BatchSize, 10])
forward arch xs = fst <$> netforward arch xs

backward
  :: Positive HsReal
  -> Arch
  -> Tensor '[BatchSize, 3, 32, 32]
  -> Tensor '[BatchSize, 10]
  -> IO (Tensor '[1], Arch)
backward plr net xs ys = do
  (out, getgrads) <- netforward net xs
  (loss, getCEgrad) <- crossEntropyIO (Long.resizeAs . fromJust . snd $ max2d1 ys keep) out
  printf "\rcur_loss: %.12f" (fromJust $ get1d loss 0)
  hFlush stdout
  pure (loss, unsafePerformIO $ do
    grad <- fmap fst . getgrads =<< getCEgrad loss
    net' <- netupdate net (plr, grad)
    pure net'
    )

-- ========================================================================= --
-- lenet from scratch
-- ========================================================================= --

netforward
  :: LeNet
  ->    (Tensor '[4, 3, 32, 32])  -- ^ input
  -> IO (Tensor '[4, 10], Tensor '[4, 10] -> IO (LeNet, Tensor '[4, 3, 32, 32]))
netforward arch i = do
  (convout1, unconvout1) <- conv2dBatchIO
      ((Step2d    :: Step2d '(1,1)))
      ((Padding2d :: Padding2d '(0,0)))
      1
      (arch ^. Vision.conv1)
      i
  (reluout1, unreluCONV1) <- reluIO convout1
  (mpout1 :: Tensor '[4, 6, 14, 14], getmpgrad1) <-
    maxPooling2dBatchIO
        ((Kernel2d  :: Kernel2d '(2,2)))
        ((Step2d    :: Step2d '(2,2)))
        ((Padding2d :: Padding2d '(0,0)))
        ((sing      :: SBool 'True))
        (reluout1 :: Tensor '[4, 6, 28, 28])

  (convout2, unconvout2) <- conv2dBatchIO
      ((Step2d    :: Step2d '(1,1)))
      ((Padding2d :: Padding2d '(0,0)))
      1
      (arch ^. Vision.conv2)
      mpout1
  (reluout2, unreluCONV2) <- reluIO convout2
  (mpout2, getmpgrad2) <-
    maxPooling2dBatchIO
        ((Kernel2d  :: Kernel2d '(2,2)))
        ((Step2d    :: Step2d '(2,2)))
        ((Padding2d :: Padding2d '(0,0)))
        ((sing      :: SBool 'True))
        reluout2


  (flatout, unflatten) <- flattenBatchIO (mpout2 :: Tensor '[4, 16, 5, 5])

  (fc1out, fc1getgrad) <- linearBatchIO (arch ^. Vision.fc1) flatout
  ( r1out, unreluFC1)    <- reluIO fc1out

  (fc2out, fc2getgrad) <- linearBatchIO (arch ^. Vision.fc2) r1out
  ( r2out, unreluFC2)    <- reluIO fc2out

  (fc3out, fc3getgrad) <- linearBatchIO (arch ^. Vision.fc3) r2out
  (fin, smgrads)       <- softMaxBatch fc3out

  pure (fin, \gout -> do
    smg <- smgrads gout
    (fc3g, fc3gin) <- fc3getgrad smg
    (fc2g, fc2gin) <- fc2getgrad =<< unreluFC2 fc3gin
    (fc1g, fc1gin) <- fc1getgrad =<< unreluFC1 fc2gin

    unflattenedGIN :: Tensor '[4, 16, 5, 5] <- unflatten fc1gin

    (conv2g, conv2gin) <- unconvout2 =<< unreluCONV2 =<< getmpgrad2 unflattenedGIN
    (conv1g, conv1gin) <- unconvout1 =<< unreluCONV1 =<< getmpgrad1 conv2gin
    pure (Vision.LeNet conv1g conv2g fc1g fc2g fc3g, conv1gin))


