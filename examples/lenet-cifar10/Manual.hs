{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE NoImplicitPrelude #-}
module Main where

import Prelude hiding (print, putStrLn)
import qualified Prelude as P
import LeNet
import Utils (bs, bsz, lr, getloss)
import DataLoading (mkVBatches, dataloader')
import Criterion ()

-------------------------------------------------------------------------------

import Data.Vector (Vector)
import Data.DList (DList)
import Data.List (genericLength)
import Control.Arrow (second)
import Control.Monad (forM, forM_, when)
import System.IO (stdout, hFlush)
import System.Mem (performGC, performMajorGC)
import Text.Printf (printf)

import qualified Data.HashMap.Strict as HM
import qualified Data.DList as DL
import qualified Data.Vector as V
import qualified System.Random.MWC as MWC

-------------------------------------------------------------------------------

import Torch.Data.Loaders.Cifar10 (Category(..), Mode(..), cifar10set, default_cifar_path)
import Torch.Data.Loaders.RGBVector (Normalize(..))
import Torch.Double

import qualified Torch.Long as Long
import qualified Torch.Long.Dynamic as LDyn
import qualified Torch.Double.NN.Conv2d as Conv2d
import qualified Torch.Double as Torch
import qualified Torch.Models.Vision.LeNet as Vision

print :: Show a => a -> IO ()
print =
  -- P.print
  const (pure ())

putStrLn :: String -> IO ()
putStrLn =
  -- P.putStrLn
  const (pure ())

-- ========================================================================= --
-- global variables

printL1Weights :: LeNet -> IO ()
printL1Weights x = (print . fst . Conv2d.getTensors . Vision._conv1) x

-- ========================================================================= --

-- type Arch = LeNet
-- mkNet = Vision.newLeNet
-- forward = _forward lenetBatchForward
-- netupdate = myupdate
-- netBatchBP = lenetBatchBP
type Arch = NIN
mkNet = mkNIN
forward = _forward ninForwardBatch
netupdate = ninUpdate
netBatchBP = ninBatchBP
trainingLR = 1.0
updateLR = 1.0

datasize = 400
reportat' = 399
epos = 20

main :: IO ()
main = do
  g <- MWC.initialize (V.singleton 44)

  lbatches :: Vector (Vector (Category, FilePath))
    <- mkVBatches 4 . V.take datasize <$> cifar10set g default_cifar_path Train
  printf "loaded %i filepaths\n\n" (length lbatches * 4)

  P.putStrLn "transforming to tensors"
  batches <- dataloader' bsz lbatches

  net0 <- mkNet
  P.print net0

  P.putStrLn "\nepochs start"
  net <- epochs lr epos batches net0
  P.putStrLn "\nepochs stop"

  P.putStrLn "reporting:"
  report net batches
  performMajorGC
  P.putStrLn ""

  where
    report :: Arch -> Vector (Tensor '[4, 10], Tensor '[4, 3, 32, 32]) -> IO ()
    report net ltest = do
      -- putStrLn "x"

      foo :: [DList (Category, Category)]
        <- forM (V.toList ltest) $ \(y, x) -> do
          bar <- y2cat y
          DL.fromList . zip bar <$> forward net x

      let
        test :: [(Category, Category)]
        test = DL.toList (DL.concat foo)

      print test

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
  :: HsReal              -- ^ learning rate
  -> Int                 -- number of epochs to train on
  -> Vector (Tensor '[4, 10], Tensor '[4, 3, 32, 32])
  -> Arch               -- ^ initial model
  -> IO Arch            -- ^ final model
epochs _ mx batches = runEpoch 1
  where
    runEpoch :: Int -> Arch -> IO Arch
    runEpoch e net
      | e > mx = P.putStrLn "\nfinished training loops" >> pure net
      | otherwise = do
      P.putStrLn $ "epoch " ++ show (e, mx)

      (net', _) <- V.ifoldM go (net, 0) batches

      performMajorGC
      runEpoch (e+1) net'

      where
        estr :: String
        estr = "[Epoch "++ show e ++ "/"++ show mx ++ "]"

        go :: (Arch, HsReal) -> Int -> (Tensor '[4, 10], Tensor '[4, 3, 32, 32]) -> IO (Arch, HsReal)
        go (!net, !runningloss) !bid (ys, xs) = do
          (net', loss) <- step undefined net xs ys
          -- P.print loss
          let reportat = reportat' :: Int
              reporttime = (bid `mod` reportat == (reportat - 1))

          when reporttime $ do
            printf ("\n%s(%db#%03d)[ce %.4f]") estr
              (bs::Int) (bid+1)
              (runningloss / (if reportat == 1 then 1 else (fromIntegral reportat - 1)))
            hFlush stdout

          performGC

          let l = getloss loss
          -- P.print l
          pure (net', if reporttime then 0 else runningloss + l)


_forward :: (Arch -> Tensor '[4, 3, 32, 32] -> IO (Tensor '[4, 10])) -> Arch -> Tensor '[4, 3, 32, 32] -> IO [Category]
_forward fwdfn net xs = y2cat =<< fwdfn net xs

step
  :: HsReal
  -> Arch
  -> Tensor '[4, 3, 32, 32]
  -> Tensor '[4, 10]
  -> IO (Arch, Tensor '[1])
step _ net xs ys = do
  let (_, Just ix) = Torch.max2d1 ys keep
  LDyn.resizeDim_ (Long.asDynamic ix) (dims :: Dims '[4])
  (out, gnet) <- netBatchBP trainingLR net (Long.asStatic (Long.asDynamic ix)) xs

  let Just plr = positive updateLR
  net' <- netupdate net (plr, gnet)

  when False $ do
    (out', _) <- netBatchBP trainingLR net' (Long.asStatic (Long.asDynamic ix)) xs
    let o = getloss out
    let o' = getloss out'
    printf ("\nimprovement? %.4f -> %.4f : %s\n") o o' (show (o < o'))

  pure (net', out)

