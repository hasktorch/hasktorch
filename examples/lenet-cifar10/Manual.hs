{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TupleSections #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE CPP #-}
module Main where

import Prelude
import Data.Vector (Vector)
import GHC.TypeLits (KnownNat)
import Data.HashMap.Strict (HashMap)
import Data.DList (DList)

import Data.Function
import Data.Maybe
import Data.Either
import Data.Typeable
import Data.List
import Control.Arrow
import Control.Concurrent
import Control.Exception.Safe
import Control.Monad
import Control.Monad.Loops
import Control.Monad.Trans.Except
import Control.Monad.IO.Class
import Data.Monoid
import Data.Time
import Debug.Trace
import Data.IORef

import GHC.Exts
import GHC.Int
import Numeric.Dimensions
import System.IO
import System.IO.Unsafe
import System.Mem
import Text.Printf

import qualified Prelude as P
import qualified Data.List as P ((!!))
import qualified Data.HashMap.Strict as HM
import qualified Data.DList as DL
import qualified Data.Vector as V
import qualified System.Random.MWC as MWC
import qualified Data.Singletons.Prelude.List as Sing (All)
import qualified Numeric.Backprop as Bp

-------------------------------------------------------------------------------

import Torch.Double
import qualified Torch.Long as Long
import qualified Torch.Long.Dynamic as LDyn
import qualified Torch.Double.NN.Conv2d as Conv2d
import qualified Torch.Double as Torch
import qualified Torch.Double.Dynamic as Dyn
import qualified Torch.Double.Storage as CPUS

import Torch.Data.Loaders.Cifar10
import Torch.Data.Loaders.Internal
import Torch.Data.Loaders.RGBVector (Normalize(..))
import Torch.Data.OneHot
import Torch.Data.Metrics

-------------------------------------------------------------------------------

import DataLoading
import Criterion
import LeNet as MyLeNet

-- ========================================================================= --
-- global variables

printL1Weights :: LeNet -> IO ()
printL1Weights x = (print . fst . Conv2d.getTensors . MyLeNet._conv1) x

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
      | e > mx = pure net
      | otherwise = do
      putStr "\n"

      print $ "epoch " ++ show (e, mx)
      (net', _) <- V.ifoldM go (net, 0) batches

      performMajorGC
      runEpoch (e+1) net'

      where
        estr :: String
        estr = "[Epoch "++ show e ++ "/"++ show mx ++ "]"

        go (!net, !runningloss) !bid (ys, xs) = do
          (net', loss) <- step lr net xs ys
          let reportat = reportat' :: Int
              reporttime = (bid `mod` reportat == (reportat - 1))

          when reporttime $ do
            printf ("\n%s(%db#%03d)[ce %.4f]") estr
              bs (bid+1)
              (runningloss / (if reportat == 1 then 1 else (fromIntegral reportat - 1)))
            hFlush stdout
            -- performGC
          l <- loss `get1d` 0
          pure (net', if reporttime then 0 else runningloss + l)


forward :: LeNet -> Tensor '[4, 3, 32, 32] -> IO [Category]
forward net xs = y2cat =<< MyLeNet.lenetBatchForward net xs


step
  :: HsReal
  -> LeNet
  -> Tensor '[4, 3, 32, 32]
  -> Tensor '[4, 10]
  -> IO (LeNet, Tensor '[1])
step lr net xs ys = do
  let (dyn, Just ix) = Torch.max ys (dim :: Dim 1) keep
  LDyn._resizeDim (Long.asDynamic ix) (dims :: Dims '[4])
  let rix = Long.asStatic (Long.asDynamic ix)
  (out, gnet) <- lenetBatchBP net rix xs
  lenetUpdate net (-lr, gnet)
  pure (net, out)


-- ========================================================================= --

-- datasize = 10000
-- reportat' = 200
-- epos = 3
-- lr = 0.01

datasize = 20000
reportat' = 500
epos = 3
lr = 0.001

main :: IO ()
main = do
  g <- MWC.initialize (V.singleton 44)

  -- let xs = (fmap (\x -> fmap ((11*x)+) [0..24]) [0..14])
  -- print xs
  -- runExceptT (Dyn.matrix
  --   xs
  -- -- runExceptT (Dyn.vector
  -- --   ([1..107])
  --   :: ExceptT String IO (Dynamic))
  --   >>= \case
  --     Left s -> print s
  --     Right t -> do
  --       print "x"
  --       (Dyn.tensordata t) >>= print
  --       print t

  -- throwString ""





  lbatches :: Vector (Vector (Category, FilePath))
    <- mkVBatches 4 . V.take datasize <$> cifar10set g default_cifar_path Train
  -- print (length lbatches)
  -- print (fmap length lbatches)
  batches <- dataloader' bsz lbatches

  net0 <- newLeNet
  print net0

  -- report net0 batches
  -- performMajorGC
  putStrLn "\nepochs start"
  net <- epochs lr epos batches net0
  putStrLn "\nepochs stop"

  report net batches
  performMajorGC
  -- putStr "\nInitial Holdout:\n"
  -- net <- epochs 0.001 3 batches net0
  -- printf "\nFinished training in %s!\n"

  -- putStrLn "\nHoldout Results on final net:\n"
  -- report net batches
  -- putStrLn "\nDone!\n"

  where
    report :: LeNet -> Vector (Tensor '[4, 10], Tensor '[4, 3, 32, 32]) -> IO ()
    report net ltest = do

      foo :: [DList (Category, Category)]
        <- forM (V.toList ltest) $ \(y, x) -> do
          bar <- y2cat y
          DL.fromList . zip bar <$> forward net x

      let
        test :: [(Category, Category)]
        test = DL.toList (DL.concat foo)

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

