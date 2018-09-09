{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TupleSections #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE Strict #-}
{-# LANGUAGE CPP #-}
module Main where

import Prelude

import Data.DList (DList)
import Data.Either -- (fromRight, is)
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
import Numeric.Backprop as Bp
import Numeric.Dimensions
import System.IO.Unsafe
import GHC.TypeLits (KnownNat)
import Control.Concurrent
import qualified Prelude as P
import qualified Data.List as P ((!!))
import Control.Exception.Safe
-- import qualified Foreign.CUDA as Cuda

#ifdef CUDA
import Torch.Cuda.Double (Tensor, HsReal, (.:))
import qualified Torch.Cuda.Double as D
import qualified Torch.Cuda.Double.Storage as S
import qualified Torch.Cuda.Long as Long
import Torch.FFI.THC.TensorRandom
import Foreign
import Torch.FFI.THC.State
#else
import Torch.Double (Tensor, HsReal, (.:))
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
  ltrain <- prepdata . V.take (bs) <$> cifar10set g default_cifar_path Train
  net0 <- newLeNet @3 @5
  print net0

  t0 <- getCurrentTime
  let
    lbatches :: Vector LDataSet
    lbatches = V.fromList $ mkBatches (fromIntegral bs) ltrain

  print lbatches

  let
    btensor :: LDataSet -> IO (Maybe (Tensor '[batch, 10], Tensor '[batch, 3, 32, 32]))
    btensor lds
      | V.length lds /= (fromIntegral bs) = pure Nothing
      | otherwise = do
        foo <- V.toList <$> V.mapM getdata lds
        ys <- toYs foo
        xs <- toXs foo
        pure $ Just (ys, xs)

    go
      :: (LeNet 3 5, DList LDataSet)
      -> Int
      -> LDataSet
      -> IO (LeNet 3 5, DList LDataSet)
    go (!net, !seen) !bid !lzy = do
      print "starting to load"
      fcd <- V.mapM forcetensor lzy
      print "done loading!"
      mten <- btensor fcd
      case mten of
        Nothing -> pure (net, seen)
        Just (ys, xs) -> do
          print "starting to train"
          (net', loss) <- trainStep lr net xs ys
          t1 <- getCurrentTime
          let diff = 0.0 :: Float
              front = "\n"
          printf (front ++ "(%db#%03d)[mse %.4f](elapsed: %.2fs)")
            (bs::Integer) (bid+1)
            (loss `D.get1d` 0)
            diff -- ((t1 `diffUTCTime` t00))
          hFlush stdout
          pure (net', seen `DL.snoc` fcd)


  res <- V.ifoldM go (net0, DL.empty) lbatches
  let (net', train) = second (V.concat . DL.toList) res

  printf "\nFinished trainFourTest\n"

toYs :: [(Category, Tensor '[3, 32, 32])] -> IO (Tensor '[4, 10])
toYs ys =
  pure . D.unsafeMatrix . fmap (onehotf . fst) $ ys

toXs :: [(Category, Tensor '[3, 32, 32])] -> IO (Tensor '[4, 3, 32, 32])
toXs xs = do
  print "XS PREPO"
  -- mapM_ assertTen (snd <$> xs)
  print xs
  print "XS CATTED"
  -- FIXME
  let xs' = D.catArray0 $ fmap (D.unsqueeze1d (dim :: Dim 0) . snd) xs
  -- assertTen xs'
  print xs'
  throwString "bleeergh"
  pure xs'




-- assertTen :: Tensor d -> IO ()
-- assertTen t =
-- #ifdef CUDA
--     CPU.tensordata (copyDouble t) >>= \rs -> do
-- #else
--     tensordata t >>= \rs -> do
-- #endif
--       assertList rs
--
-- assertList :: [HsReal] -> IO ()
-- assertList rs = do
--     let
--       oob = filter (\x -> x < -0.1 || x > 1.1) rs
--       oox = filter (<= 2) oob
--     if not (null oob)
--     then throwString $ show (oob, length oob, length rs, "OOB found!")
--     else
--       if all (== 0) rs
--       then throwString ("all-zeros found!")
--       else pure ()


-- ========================================================================= --
-- Data processing + bells and whistles for a slow loader
-- ========================================================================= --

preprocess :: FilePath -> IO (Tensor '[3, 32, 32])
preprocess f =
  runExceptT (rgb2torch (Normalize False) f) >>= \case
    Left s -> throwString s
    Right t -> do
      print "preprocessed, assert data was allocated correctly"
      -- assertTen t
      pure t


-- | get something usable from a lazy datapoint
getdata :: LDatum -> IO (Category, Tensor '[3, 32, 32])
getdata (c, Right t) = pure (c, t)
getdata (c, Left fp) = undefined -- (c,) <$> preprocess fp

-- | force a file into a tensor
forcetensor :: LDatum -> IO LDatum
forcetensor = \case
  (c, Left fp) -> (c,) . Right <$> preprocess fp
  tens -> pure tens

infer
  :: (ch ~ 3, step ~ 5)
  => LeNet ch step
  -> Tensor '[ch, 32, 32]
  -> Category
infer net

  -- cast from Integer to 'Torch.Data.Loaders.Cifar10.Category'
  = toEnum . fromIntegral

  -- Unbox the LongTensor '[1] to get 'Integer'
  . (`Long.get1d` 0)

  -- argmax the output Tensor '[10] distriubtion. Returns LongTensor '[1]
  -- . maxIndex1d
  . undefined

  . foo

 where
  foo x
    -- take an input tensor and run 'lenet' with the model (undefined is the
    -- learning rate, which we can ignore)
    = unsafePerformIO $ do
        -- print $ x Math.!! (dim :: Dim 0) Math.!! (dim :: Dim 0)
        let x' = evalBP2 (lenet undefined) net x
        -- print x'
        pure x'

trainStep
  :: forall ch step batch
  .  (ch ~ 3, step ~ 5)
  => KnownDim batch
  => KnownNat batch
  => KnownDim (batch * 10)
  => KnownNat (batch * 10)

  => HsReal
  -> LeNet ch step
  -> Tensor '[batch, ch, 32, 32]
  -> Tensor '[batch, 10]
  -> IO (LeNet ch step, Tensor '[1])
trainStep lr net xs ys = do
  -- print xs
  pure (Bp.add net gnet, out)
  where
    (out, (gnet, _)) = undefined -- backprop2 ( mSECriterion ys .: lenetBatch lr) net xs


