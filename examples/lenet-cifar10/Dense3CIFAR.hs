{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE CPP #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE Strict #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Normalise #-}
module Main where

import Prelude
import Utils hiding (lr, bs, bsz)
import DataLoading (mkVBatches, dataloader')
import Dense3

import Data.DList (DList)
import Data.Maybe (fromJust)
import Data.List (genericLength)
import Data.Vector (Vector)
import Control.Arrow (second)
import Control.Monad (forM, forM_, when)
import System.IO (stdout, hFlush)
import System.IO.Unsafe (unsafePerformIO)
import System.Mem (performGC, performMajorGC)
import Text.Printf (printf)

import qualified Data.HashMap.Strict as HM
import qualified Data.DList as DL
import qualified Data.Vector as V
import qualified System.Random.MWC as MWC

#ifdef CUDA
import Torch.Cuda.Double -- (Tensor, HsReal, (.:), mSECriterion)
import qualified Torch.Cuda.Double as D
import qualified Torch.Cuda.Long as Long
#else
import Torch.Double -- (Tensor, HsReal, (.:), mSECriterion)
import qualified Torch.Double as D
import qualified Torch.Long as Long
#endif

import Torch.Double.NN.Linear (Linear(..))
import Torch.Data.Loaders.Cifar10
import Torch.Data.OneHot (onehotf)

lr = 1.0
datasize = 40
reportat' = 399
epos = 1

type BatchSize = 4
bsz :: Dim BatchSize
bsz = dim
bs :: Integral i => i
bs = fromIntegral (dimVal bsz)

main :: IO ()
main = do
  g <- MWC.initialize (V.singleton 44)

  lbatches :: Vector (Vector (Category, FilePath))
    <- mkVBatches 4 . V.take datasize <$> cifar10set g default_cifar_path Train
  printf "loaded %i filepaths\n\n" (length lbatches * 4)

  putStrLn "transforming to tensors"
  batches <- dataloader' bsz lbatches

  net0 <- mkFC3
  print net0

  net <- epochs lr epos batches net0

  putStrLn "reporting:"
  report net batches
  performMajorGC
  putStrLn ""

  where
    report :: FC3Arch -> Vector (Tensor '[BatchSize, 10], Tensor '[BatchSize, 3, 32, 32]) -> IO ()
    report net ltest = do
      -- putStrLn "x"

      foo :: [DList (Category, Category)]
        <- forM (V.toList ltest) $ \(y, x) -> do
          bar <- y2cat y
          DL.fromList . zip bar . t2cat <$> forward net x

      let
        test :: [(Category, Category)]
        test = DL.toList (DL.concat foo)

      -- print test

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
  -> Vector (Tensor '[BatchSize, 10], Tensor '[BatchSize, 3, 32, 32])
  -> FC3Arch               -- ^ initial model
  -> IO FC3Arch            -- ^ final model
epochs lr mx batches = runEpoch 1
  where
    runEpoch :: Int -> FC3Arch -> IO FC3Arch
    runEpoch e net
      | e > mx = putStrLn "\nfinished training loops" >> pure net
      | otherwise = do
      (net', losstot) <- V.ifoldM go (net, 0) batches
      printf ("%s[ce %.4f]\n") estr (losstot / fromIntegral (V.length batches))

      performMajorGC
      runEpoch (e+1) net'

      where
        estr :: String
        estr = "[Epoch "++ show e ++ "/"++ show mx ++ "]"

        go (!net, !runningloss) !bid (ys, xs) = do
          (net', loss) <- step True lr net xs ys
          performGC
          pure (net', runningloss + getloss loss)

toYs :: [(Category, Tensor '[3, 32, 32])] -> IO (Tensor '[BatchSize, 10])
toYs = D.unsafeMatrix . fmap (onehotf . fst)

toXs :: [(Category, Tensor '[3, 32, 32])] -> IO (Tensor '[BatchSize, 3, 32, 32])
toXs xs = pure . D.catArray0 $ fmap (D.unsqueeze1d (dim :: Dim 0) . snd) xs

t2cat :: Tensor '[BatchSize, 10] -> [Category]
t2cat = (fmap (toEnum . fromIntegral) . Long.tensordata . fromJust . snd . flip max2d1 keep)

forward arch xs = do
  fst <$> dense3BatchIO lr arch (resizeAs xs)

-- step
--   :: Bool
--   -> HsReal
--   -> FC3Arch
--   -> Tensor '[BatchSize, 3, 32, 32]
--   -> Tensor '[BatchSize, 10]
--   -> IO (FC3Arch, Tensor '[1])
-- step istraining lr net@(l1, l2, l3) xs ys = do
--   (ff3out, getff3grads) <- dense3BatchIO lr net (resizeAs xs)
--   (loss, getCEgrad) <- crossEntropyIO (Long.resizeAs . fromJust . snd $ max2d1 ys keep) ff3out
--   pure (if not istraining then undefined else unsafePerformIO $ do
--     (g1, g2, g3) <- fmap fst . getff3grads =<< getCEgrad loss
--     let l1' = (l1 - g1)
--     let l2' = (l2 - g2)
--     let l3' = (l3 - g3)
--     pure (l1', l2', l3')
--     , loss)

step
  :: Bool
  -> HsReal
  -> FC3Arch
  -> Tensor '[BatchSize, 3, 32, 32]
  -> Tensor '[BatchSize, 10]
  -> IO (FC3Arch, Tensor '[1])
step istraining lr net@(Linear (l1, b1)) xs ys = do
  -- print xs
  (ff3out, getff3grads) <- dense3BatchIO lr net (resizeAs xs)
  (loss, getCEgrad) <- crossEntropyIO (Long.resizeAs . fromJust . snd $ max2d1 ys keep) ff3out
  pure (if not istraining then undefined else unsafePerformIO $ do
    (Linear (g1, _)) <- fmap fst . getff3grads =<< getCEgrad loss
    -- let l1' = (l1 + g1)
    pure (Linear (l1-g1, b1))
    , loss)

