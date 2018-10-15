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
import LeNet (LeNet, lenetUpdate, lenetUpdate_, lenetBatchBP, lenetBatchForward, y2cat, myupdate)
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

datasize = 80
reportat' = 200
epos = 1

main :: IO ()
main = do
  g <- MWC.initialize (V.singleton 44)

  printf "loading %i images\n\n" datasize
  lbatches :: Vector (Vector (Category, FilePath))
    <- mkVBatches 4 . V.take datasize <$> cifar10set g default_cifar_path Train

  P.putStrLn "transforming to tensors"
  batches <- dataloader' bsz lbatches

  net0 <- Vision.newLeNet
  P.print net0

  P.putStrLn "\nepochs start"
  net <- epochs lr epos batches net0
  P.putStrLn "\nepochs stop"

  P.putStrLn "reporting:"
  report net batches
  performMajorGC
  P.putStrLn ""

  -- putStr "\nInitial Holdout:\n"
  -- net <- epochs 0.001 3 batches net0
  -- printf "\nFinished training in %s!\n"

  -- putStrLn "\nHoldout Results on final net:\n"
  -- report net batches
  -- putStrLn "\nDone!\n"

  where
    report :: LeNet -> Vector (Tensor '[4, 10], Tensor '[4, 3, 32, 32]) -> IO ()
    report net ltest = do
      -- putStrLn "x"

      foo :: [DList (Category, Category)]
        <- forM (V.toList ltest) $ \(y, x) -> do
          bar <- y2cat y
          -- print "bar"
          -- print bar
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
        printf "\n[%s]: %.2f%% (%d / %d)" (show y) (acc*100) correct (length preds)
        hFlush stdout

-- ========================================================================= --
-- Training on a dataset
-- ========================================================================= --
epochs
  :: HsReal              -- ^ learning rate
  -> Int                 -- number of epochs to train on
  -> Vector (Tensor '[4, 10], Tensor '[4, 3, 32, 32])
  -> LeNet               -- ^ initial model
  -> IO LeNet            -- ^ final model
epochs lr mx batches = runEpoch 1
  where
    runEpoch :: Int -> LeNet -> IO LeNet
    runEpoch e net
      | e > mx = P.putStrLn "\nfinished training loops" >> pure net
      | otherwise = do
      P.putStrLn $ "\nepoch " ++ show (e, mx)
      putStrLn "y"

      (net', _) <- V.ifoldM go (net, 0) batches

      putStrLn "xy"
      performMajorGC
      runEpoch (e+1) net'

      where
        estr :: String
        estr = "[Epoch "++ show e ++ "/"++ show mx ++ "]"

        go (!net, !runningloss) !bid (ys, xs) = do
          putStrLn "start step"
          (net', loss) <- step lr net xs ys
          putStrLn "stop step"
          let reportat = reportat' :: Int
              reporttime = (bid `mod` reportat == (reportat - 1))

          when reporttime $ do
            printf ("\n%s(%db#%03d)[ce %.4f]") estr
              (bs::Int) (bid+1)
              (runningloss / (if reportat == 1 then 1 else (fromIntegral reportat - 1)))
            hFlush stdout

          performGC

          let l = getloss loss
          pure (net', if reporttime then 0 else runningloss + l)


forward :: LeNet -> Tensor '[4, 3, 32, 32] -> IO [Category]
forward net xs = y2cat =<< lenetBatchForward net xs

step
  :: HsReal
  -> LeNet
  -> Tensor '[4, 3, 32, 32]
  -> Tensor '[4, 10]
  -> IO (LeNet, Tensor '[1])
step lr net xs ys = do
  let (!dyn, Just ix) = Torch.max2d1 ys keep
  LDyn.resizeDim_ (Long.asDynamic ix) (dims :: Dims '[4])
  putStrLn "start lenetBatchBP"

  (out, gnet) <- lenetBatchBP net (Long.asStatic (Long.asDynamic ix)) xs
  let o = getloss out

  putStrLn "stop lenetBatchBP"
  putStrLn "out"

  -- lenetUpdate_ net (0.1, gnet)
  net' <- myupdate net 0.1 gnet

  (out', _) <- lenetBatchBP net' (Long.asStatic (Long.asDynamic ix)) xs

  let o' = getloss out'
  printf ("\nimprovement? %.4f -> %.4f : %s\n") o o' (show (o < o'))

  pure (net', out)

