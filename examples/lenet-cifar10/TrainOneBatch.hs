{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TupleSections #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE Strict #-}
{-# LANGUAGE CPP #-}
module Main where

import Prelude
import Utils
import LeNet
import LeNet.Forward

import Data.DList (DList)
import Data.Either -- (fromRight, is)
import Data.Maybe
import GHC.Exts
import Data.Typeable
import Data.List
import Control.Arrow
import Control.Monad
import Control.Monad.Loops
import Data.Monoid
import Data.Time
import Control.Monad.IO.Class
import Text.Printf
import ListT (ListT)
import qualified ListT
-- import Numeric.Backprop as Bp
import Numeric.Dimensions
import System.IO.Unsafe
import GHC.TypeLits (KnownNat)
import qualified Prelude as P
import qualified Data.List as P ((!!))
import Control.Exception.Safe
-- import qualified Foreign.CUDA as Cuda

#ifdef CUDA
import Torch.Cuda.Double -- (Tensor, HsReal, (.:), mSECriterion)
import qualified Torch.Cuda.Double as D
import qualified Torch.Cuda.Double.Storage as S
import qualified Torch.Cuda.Long as Long
import Torch.FFI.THC.TensorRandom
import Foreign
import Torch.FFI.THC.State
#else
import Torch.Double -- (Tensor, HsReal, (.:), mSECriterion)
import qualified Torch.Double as D
import qualified Torch.Long as Long
#endif

import Torch.Models.Vision.LeNet
import Torch.Data.Loaders.Cifar10
import Torch.Data.OneHot
import Torch.Data.Metrics

import Control.Monad.Trans.Except
import System.IO (hFlush, stdout)

import Data.Vector (Vector)
import qualified Data.DList as DL
import qualified Data.Vector as V
import qualified System.Random.MWC as MWC
import qualified Data.Singletons.Prelude.List as Sing (All)

import Debug.Trace
import qualified Torch.Double as CPU
import qualified Torch.Double.Storage as CPUS
import Control.Concurrent
import Data.IORef

import Torch.Data.Loaders.Internal
import Torch.Data.Loaders.RGBVector hiding (HsReal, assertList)
import Utils

main :: forall batch . batch ~ 4 => IO ()
main = seedAll >>= \g -> do
  ltrain <- prepdata . V.take bs <$> cifar10set g default_cifar_path Train
  net0 <- newLeNet @3 @5
  print net0

  t0 <- getCurrentTime

  let
    lbatch :: LDataSet
    [lbatch] = mkBatches (fromIntegral bs) ltrain


  putStrLn "looking at lazily loaded data:"
  print lbatch

  let
    btensor :: LDataSet -> IO (Maybe (Tensor '[batch, 10], Tensor '[batch, 3, 32, 32]))
    btensor lds
      | V.length lds /= (fromIntegral bs) = pure Nothing
      | otherwise = do
        foo <- V.toList <$> V.mapM getdata lds
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
      (net', loss) <- trainStep lr net0 xs ys
      putStrLn "..training succeeded!"
      let diff = 0.0 :: Float
          front = "\n"
      printf (front ++ "(%db#%03d)[mse %.4f]\n")
        (bs :: Integer) (1 :: Int)
        (getloss loss)

  printf "\nFinished single-batch training!\n"

toYs :: [(Category, Tensor '[3, 32, 32])] -> IO (Tensor '[4, 10])
toYs = D.unsafeMatrix . fmap (onehotf . fst)

toXs :: [(Category, Tensor '[3, 32, 32])] -> IO (Tensor '[4, 3, 32, 32])
toXs xs = pure . D.catArray0 $ fmap (D.unsqueeze1d (dim :: Dim 0) . snd) xs

trainStep
  :: KnownDim 4
  => KnownNat 4
  => KnownDim (4 * 10)
  => KnownNat (4 * 10)

  => HsReal
  -> LeNet 3 5
  -> Tensor '[4, 3, 32, 32]
  -> Tensor '[4, 10]
  -> IO (LeNet 3 5, Tensor '[1])
trainStep lr net xs ys = do
  (out, gnet) <- lenetBatchBP net (Long.resizeAs . fromJust . snd $ max2d1 ys keep) xs
  lenetUpdate net (lr, gnet)
  pure (net, out)


