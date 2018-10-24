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
import Data.Function
import Data.Maybe
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

import Data.HashMap.Strict (HashMap)
import qualified Data.HashMap.Strict as HM

#ifdef DEBUG
import Debug.Trace
import Data.IORef
#endif
-- import qualified Foreign.CUDA as Cuda

#ifdef CUDA
import Torch.Cuda.Double as Math hiding (Sum)
import qualified Torch.Cuda.Double.Storage as S
import qualified Torch.Cuda.Long as Long
import Torch.FFI.THC.TensorRandom
import Foreign (withForeignPtr)
import Torch.FFI.THC.State
import qualified Torch.Cuda.Double.NN.Conv2d as Conv2d
#else
import Torch.Double as Math hiding (Sum)
import qualified Torch.Long as Long
import qualified Torch.Long.Dynamic as LDyn
import qualified Torch.Double.NN.Conv2d as Conv2d
#endif

import Torch.Models.Vision.LeNet as LeNet
import Torch.Data.Loaders.Cifar10
import Torch.Data.Loaders.Internal
import Torch.Data.Loaders.RGBVector (Normalize(..))
import Torch.Data.OneHot
import Torch.Data.Metrics

import Control.Monad.Trans.Except
import System.IO (hFlush, stdout)

import Data.Vector (Vector)
import qualified Data.DList as DL
import qualified Data.Vector as V
import qualified System.Random.MWC as MWC
import qualified Data.Singletons.Prelude.List as Sing (All)

#ifdef DEBUG
import Debug.Trace
import qualified Torch.Double as CPU
import qualified Torch.Double.Dynamic as Dyn
import qualified Torch.Double.Storage as CPUS
import Control.Concurrent
#endif

lr = 0.001
bsz = (dim :: Dim 4)
bs = (fromIntegral $ dimVal bsz)

main :: IO ()
main = do
#ifndef DEBUG
  clearScreen
#endif
  g <- seedAll
  ltrain <- prepdata . V.take 5000 <$> cifar10set g default_cifar_path Train
  ltest  <- prepdata . V.take 2000 <$> cifar10set g default_cifar_path Test

  let (lval, lhold) = V.splitAt (V.length ltest `P.div` 2) ltest
  net0 <- newLeNet @3 @5
  print net0

  putStr "\nInitial Holdout:\t"
  hold <- testNet net0 lhold
  report net0 hold

  t0 <- getCurrentTime
  net <- epochs lval 0.001 t0 5 ltrain net0
  t1 <- getCurrentTime
  printf "\nFinished training!\n"

  -- putStrLn "\nHoldout Results on final net: "
  -- _ <- testNet net hold
  -- putStrLn "\nDone!"

-- ========================================================================= --
-- Data processing + bells and whistles for a slow loader
-- ========================================================================= --
counter :: IORef Integer
counter = unsafePerformIO $ newIORef 0
{-# NOINLINE counter #-}

preprocess :: FilePath -> IO (Tensor '[3, 32, 32])
preprocess f =
  runExceptT (rgb2torch NegOneToOne f) >>= \case
    Left s -> throwString s
    Right t -> do
      pure t

seedAll :: IO MWC.GenIO
seedAll =
  MWC.initialize (V.singleton 42) >>= \g ->
#ifdef CUDA
  withForeignPtr torchstate (\s -> c_THCRandom_manualSeed s 42) >>
#endif
  pure g


-- | potentially lazily loaded data point
type LDatum = (Category, Either FilePath (Tensor '[3, 32, 32]))
-- | potentially lazily loaded dataset
type LDataSet = Vector (Category, Either FilePath (Tensor '[3, 32, 32]))

-- | prep cifar10set
prepdata :: Vector (Category, FilePath) -> LDataSet
prepdata = fmap (second Left)

-- | get something usable from a lazy datapoint
getdata :: LDatum -> IO (Category, Tensor '[3, 32, 32])
getdata (c, Right t) = pure (c, t)
getdata (c, Left fp) = (c,) <$> preprocess fp

-- | force a file into a tensor
forcetensor :: LDatum -> IO LDatum
forcetensor = \case
  (c, Left fp) -> (c,) . Right <$> preprocess fp
  tens -> pure tens

-- ========================================================================= --
-- Training on a dataset
-- ========================================================================= --
epochs
  :: forall ch step . (ch ~ 3, step ~ 5)
  => LDataSet            -- ^ validation set
  -> HsReal              -- ^ learning rate
  -> UTCTime             -- ^ start time (for logging)
  -> Int                 -- ^ number of epochs to run
  -> LDataSet            -- ^ training set
  -> LeNet ch step       -- ^ initial architecture
  -> IO (LeNet ch step)
epochs lval lr t0 mx ltrain net0 = do
  putStr "\nInitial Validation:\t"
  val <- testNet net0 lval
  runEpoch val 1 net0
  where
    runEpoch :: LDataSet -> Int -> LeNet ch step -> IO (LeNet ch step)
    runEpoch val e net
      | e > mx    = pure net
      | otherwise = do
        putStr "\n"
        let estr = "[Epoch "++ show e ++ "/"++ show mx ++ "]"
        (net', train) <- runBatches estr (dim :: Dim 4) lr t0 e ltrain net
        testNet net' val
        report net0 val
        runEpoch val (e + 1) net'

mkBatches :: Int -> LDataSet -> [LDataSet]
mkBatches sz ds = DL.toList $ go mempty ds
 where
  go :: DList LDataSet -> LDataSet -> DList LDataSet
  go bs src =
    if V.null src
    then bs
    else
      let (b, nxt) = V.splitAt sz src
      in go (bs `DL.snoc` b) nxt


runBatches
  :: forall ch step (batch::Nat) . (ch ~ 3, step ~ 5)
  => KnownDim batch
  => KnownNat batch
  => KnownDim (batch * 10)
  => KnownNat (batch * 10)

  => String
  -> Dim batch
  -> HsReal
  -> UTCTime
  -> Int
  -> LDataSet
  -> LeNet ch step

  -> IO (LeNet ch step, LDataSet)
runBatches estr d lr t00 e lds net = do
  res <- V.ifoldM go (net, DL.empty) lbatches
  pure $ second (V.concat . DL.toList) res
 where
  bs :: Word
  bs = dimVal d

  lbatches :: Vector LDataSet
  lbatches = V.fromList $ mkBatches (fromIntegral bs) lds

  go
    :: (LeNet ch step, DList LDataSet)
    -> Int
    -> LDataSet
    -> IO (LeNet ch step, DList LDataSet)
  go (!net, !seen) !bid !lzy = do
    fcd <- V.mapM forcetensor lzy
    btensor fcd >>= \case
        Nothing -> pure (net, seen)
        Just (ys, xs) -> do
          (net', loss) <- trainStep lr net xs ys
          t1 <- getCurrentTime
#ifdef DEBUG
          let diff = 0.0 :: Float
              front = "\n"
#else
          let diff = realToFrac (t1 `diffUTCTime` t00) :: Float
              front = setRewind
#endif
          printf (front ++ estr ++ "(%db#%03d)[ce %.4f](elapsed: %.2fs)")
            bs (bid+1)
            (loss `get1d` 0)
            diff -- ((t1 `diffUTCTime` t00))
          hFlush stdout
          pure (net', seen `DL.snoc` fcd)

  btensor :: LDataSet -> IO (Maybe (Tensor '[batch, 10], Tensor '[batch, 3, 32, 32]))
  btensor lds
    | V.length lds /= (fromIntegral bs) = pure Nothing
    | otherwise = do
      -- ds <- V.mapM getdata lds
      -- pure $ (Just . (toYs &&& toXs) . V.toList) ds
      foo <- V.toList <$> V.mapM getdata lds
      ys <- toYs foo
      xs <- toXs foo
      pure $ Just (ys, xs)
    where
      toYs :: [(Category, Tensor '[3, 32, 32])] -> IO (Tensor '[batch, 10])
      toYs ys =
        pure . unsafeMatrix . fmap (onehotf . fst) $ ys

      toXs :: [(Category, Tensor '[3, 32, 32])] -> IO (Tensor '[batch, 3, 32, 32])
      toXs xs =
        pure . catArray0 $ fmap (unsqueeze1d (dim :: Dim 0) . snd) xs


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
  . maxIndex1d

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
  let (dyn, Just ix) = CPU.max ys (dim :: Dim 1) keep
  rix :: Long.Tensor '[batch] <- Long._resizeDim ix
  let (out, (gnet, _)) = backprop2 (crossEntropy rix .: lenetBatch lr) net xs
  pure (LeNet.update net lr gnet, out)


crossEntropy
  :: (Reifies s W, CPU.All KnownDim '[b, p])
  => IndexTensor '[b]            -- THIndexTensor *target,
  -> BVar s (Tensor '[b, p])     -- THTensor *input,
  -> BVar s (Tensor '[1])        -- THTensor *output,
crossEntropy ys inp
  = logSoftMaxN (dim :: Dim 0) inp
  & classNLLCriterion ys


-- ========================================================================= --
-- Testing a dataset
-- ========================================================================= --

testNet :: (ch ~ 3, step ~ 5) => LeNet ch step -> LDataSet -> IO LDataSet
testNet net ltest = do
  test <-
    if all isRight (V.map snd ltest)
    then pure $ V.map (second (fromRight undefined)) ltest
    else do
      -- print "getting test data"
      let l = fromIntegral (length ltest) :: Float
      V.mapM getdata ltest

  let
    testX = V.toList $ fmap snd test
    testY = V.toList $ fmap fst test
    preds = map (infer net) testX
    acc = genericLength (filter id $ zipWith (==) preds testY) / genericLength testY

#ifdef DEBUG
  printf "\n"
#endif
  printf ("[test accuracy: %.1f%% / %d]\tAll same? %s")
    (acc * 100 :: Float)
    (length testY)
    (if all (== head preds) preds then show (head preds) else "No.")

  hFlush stdout

  pure $ fmap (second Right) test


report :: (ch ~ 3, step ~ 5) => LeNet ch step -> LDataSet -> IO ()
report net ltest = do
  -- assert $ all isRight (V.map snd ltest)
  let
    test = V.map (second (fromRight undefined)) ltest
    cathm :: [(Category, [Tensor '[3, 32, 32]])]
    cathm = HM.toList $ HM.fromListWith (++) $ V.toList (second (:[]) <$> test)

  forM_ cathm $ \(y, xs) -> do
    let
      preds = map (infer net) xs
      correct = length (filter (==y) preds)
      acc = fromIntegral correct / genericLength xs :: Float

    printf "\n[%s]: %.2f%% (%d / %d)" (show y) (acc*100) correct (length xs)
    hFlush stdout




-------------------------------------------------------------------------------
-- Sanity check tests

-- There is a bug here having to do with CUDA.
loadtest :: IO ()
loadtest = do
  g <- MWC.initialize (V.singleton 42)
  ltrain <- prepdata . V.take 5000 <$> cifar10set g default_cifar_path Train
  forM_ ltrain $ \(_, y) -> case y of
    Left x -> insanity x
    Right x -> pure x


insanity :: FilePath -> IO (Tensor '[3, 32, 32])
insanity f = go 0
 where
  go x =
    runExceptT (rgb2torch ZeroToOne f) >>= \case
      Left s -> throwString s
      Right t -> do
#ifdef CUDA
        CPU.tensordata (copyDouble t) >>= \rs -> do
#else
        tensordata t >>= \rs -> do
#endif
          let
              oob = filter (\x -> x < 0 || x > 1) rs
              oox = filter (<= 2) oob
          if not (null oob)
          then throwString (show (oob, oox))
          else
            if not (all (== 0) rs)
            then pure t
            else if x == 10
              then throwString $ f ++ ": 10 retries -- failing on all-zero tensor"
              else do
                print $ f ++ ": retrying from " ++ show x
                threadDelay 1000000
                go (x+1)

-- ========================================================================= --
-- printing helpers

-- | Erase the last line in an ANSI terminal
clearLn :: IO ()
clearLn = printf "\ESC[2K"

-- | set rewind marker for 'clearLn'
setRewind :: String
setRewind = "\r"

-- | clear the screen in an ANSI terminal
clearScreen :: IO ()
clearScreen = putStr "\ESC[2J"

printL1Weights :: KnownDim ch => KnownDim step => LeNet ch step -> IO ()
printL1Weights = print . fst . Conv2d.getTensors . _conv1

