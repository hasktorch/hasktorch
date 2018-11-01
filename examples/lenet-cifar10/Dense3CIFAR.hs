{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE CPP #-}
{-# LANGUAGE DataKinds #-}
module Main where

import Prelude
import Utils hiding (lr, bs, bsz)
import DataLoading (mkVBatches, dataloader')
import Dense3

import Data.DList (DList)
import Data.List (genericLength)
import Data.Maybe (fromJust)
import Data.Time (getCurrentTime, diffUTCTime)
import Data.Vector (Vector)
import Control.Arrow (second)
import Control.Monad (forM, forM_, when)
import System.IO (stdout, hFlush)
import System.IO.Unsafe (unsafePerformIO)
-- import System.Mem (performGC, performMajorGC)
import Text.Printf (printf)
import Data.List.NonEmpty (NonEmpty)
import Control.Exception.Safe

import qualified Data.List.NonEmpty as NE
import qualified Data.HashMap.Strict as HM
import qualified Data.DList as DL
import qualified Data.Vector as V
import qualified System.Random.MWC as MWC

#ifdef CUDA
import qualified Torch.Cuda.Double as D
import qualified Torch.Cuda.Long as Long
import Torch.Cuda.Double
#else
import qualified Torch.Double as D
import qualified Torch.Long as Long
import Torch.Double
#endif
  ( Tensor, HsReal
  , (.:), Dim, dim, dimVal
  , mSECriterionIO
  , Positive, positive, positiveValue
  , (^*), get1d, keep
  , max2d1
  , resizeAs
  )

import Torch.Double.NN.Linear (Linear(..))
import Torch.Data.Loaders.Cifar10
import Torch.Data.OneHot (onehotf)

datasize = 10000
reportat' = 1000
epos = 30
Just plr = positive 0.01

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
  printf "\nloaded %i filepaths" (length lbatches * 4)
  putStrLn "\ntransforming to tensors"
  batches <- dataloader' bsz lbatches

  net0 <- mkFC3
  print net0

  net <- epochs plr epos batches net0

  putStrLn "reporting:"
  report net batches
  putStrLn ""

  where
    report :: FC3Arch -> Vector (Tensor '[BatchSize, 10], Tensor '[BatchSize, 3, 32, 32]) -> IO ()
    report net ltest = do
      foo :: [DList (Category, Category)]
        <- forM (V.toList ltest) $ \(y, x) -> do
          bar <- y2cat y
          DL.fromList . zip bar . t2cat <$> forward3 net x

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
  -> FC3Arch               -- ^ initial model
  -> IO FC3Arch            -- ^ final model
epochs plr mx batches = runEpoch 1
  where
    runEpoch :: Int -> FC3Arch -> IO FC3Arch
    runEpoch e net
      | e > mx = putStrLn "\nfinished training loops" >> pure net
      | otherwise = do
      t0 <- getCurrentTime
      (net', losstot) <- V.ifoldM go (net, 0) batches
      t1 <- getCurrentTime
      let diff = realToFrac (t1 `diffUTCTime` t0) :: Float
      printf ("%s[ce %.8f] %.0fs\n") estr (losstot / fromIntegral (V.length batches)) diff
      runEpoch (e+1) net'

      where
        estr :: String
        estr = "[Epoch "++ show e ++ "/"++ show mx ++ "]"

        go (!net, !runningloss) !bid (ys, xs) = do
          (loss, net') <- backward plr net xs ys
          pure (net', runningloss + getloss loss)

toYs :: [(Category, Tensor '[3, 32, 32])] -> IO (Tensor '[BatchSize, 10])
toYs = D.unsafeMatrix . fmap (onehotf . fst)

toXs :: [(Category, Tensor '[3, 32, 32])] -> IO (Tensor '[BatchSize, 3, 32, 32])
toXs xs =
  case D.catArray0 $ fmap (D.unsqueeze1d (dim :: Dim 0) . snd) (NE.fromList xs) of
    Left s -> throwString s
    Right t -> pure t

t2cat :: Tensor '[BatchSize, 10] -> [Category]
t2cat = (fmap (toEnum . fromIntegral) . Long.tensordata . fromJust . snd . flip max2d1 keep)

forward3 :: FC3Arch -> Tensor '[BatchSize, 3, 32, 32] -> IO (Tensor '[BatchSize, 10])
forward3 arch xs = do
  fst <$> dense3BatchIO undefined arch (resizeAs xs)

backward
  :: Positive HsReal
  -> FC3Arch
  -> Tensor '[BatchSize, 3, 32, 32]
  -> Tensor '[BatchSize, 10]
  -> IO (Tensor '[1], FC3Arch)
backward plr net xs ys = do
  (ff3out, getff3grads) <- dense3BatchIO undefined net (resizeAs xs)
  (loss, getCEgrad) <- crossEntropyIO (Long.resizeAs . fromJust . snd $ max2d1 ys keep) ff3out
  printf "\rloss: %.12f" (fromJust $ get1d loss 0)
  hFlush stdout
  pure (loss, unsafePerformIO $ do
    grad <- fmap fst . getff3grads =<< getCEgrad loss
    pure $ net `update` (plr, grad)
    )

update :: FC3Arch -> (Positive HsReal, FC3Arch) -> FC3Arch
update (l1, l2, l3) (plr, (g1, g2, g3)) =
  ( l1 - (g1 ^* lr)
  , l2 - (g2 ^* lr)
  , l3 - (g3 ^* lr)
  )
 where
  lr :: HsReal
  lr = positiveValue plr


