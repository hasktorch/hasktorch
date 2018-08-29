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

import Data.DList (DList)
import Data.Either (fromRight)
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
import Torch.Cuda.Double as Math hiding (Sum)
import qualified Torch.Cuda.Long as Long
#else
import Torch.Double as Math hiding (Sum)
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

-- batch dimension
-- shuffle data
-- normalize inputs

main :: IO ()
main = do
  clearScreen
  g <- MWC.initialize (V.singleton 42)
  ltrain <- prepdata . V.take 250 <$> cifar10set g default_cifar_path Train
  ltest  <- prepdata . V.take 100 <$> cifar10set g default_cifar_path Test
  let (lval, lhold) = V.splitAt (V.length ltest `P.div` 2) ltest
  net0 <- newLeNet @3 @5
  print net0

  putStrLn "\nHoldout Results on initial net: "
  hold <- testNet net0 lhold

  putStrLn "Start training:"
  t0 <- getCurrentTime
  net <- epochs lval 0.01 t0 2 ltrain net0
  t1 <- getCurrentTime
  printf "\nFinished training!\n"

  putStrLn "\nHoldout Results on final net: "
  _ <- testNet net hold
  putStrLn "\nDone!"

-- ========================================================================= --
-- Data processing + bells and whistles for a slow loader
-- ========================================================================= --
preprocess :: FilePath -> IO (Tensor '[3, 32, 32])
preprocess f =
  runExceptT ((^/ 255) <$> rgb2torch f) >>= \case
    Left s -> throwString s
    Right t -> do
#ifdef DEBUG
      assert t
#endif
      pure t
 where
  assert :: Tensor '[3, 32, 32] -> IO ()
  assert t = tensordata t >>= \rs ->
    if getAll (mconcat (fmap (\x -> All $ x >= 0 && x <= 1) rs))
    then pure ()
    else throwString (show t)

-- | potentially lazily loaded data point
type LDatum = (Category, Either FilePath (Tensor '[3, 32, 32]))
-- | potentially lazily loaded dataset
type LDataSet = Vector (Category, Either FilePath (Tensor '[3, 32, 32]))

-- | prep cifar10set
prepdata :: Vector (Category, FilePath) -> LDataSet
prepdata = fmap (second Left)

-- | get something usable from a lazy datapoint
getdata :: LDatum -> IO (Category, Tensor '[3, 32, 32])
getdata = fmap (second (fromRight impossible)) . forcetensor
  where
    impossible = error "impossible: left after forcetensor"

-- | force a file into a tensor
forcetensor :: LDatum -> IO LDatum
forcetensor = \case
  (c, Left fp) -> (c,) . Right <$> preprocess fp
  tens -> pure tens

-- ========================================================================= --
-- Testing a dataset
-- ========================================================================= --

testNet :: (ch ~ 3, step ~ 5) => LeNet ch step -> LDataSet -> IO LDataSet
testNet net ltest = do
  test <- V.mapM getdata ltest
  let
    testX = V.toList $ fmap snd test
    testY = V.toList $ fmap fst test
    preds = map (infer net) testX
    acc = genericLength (filter id $ zipWith (==) preds testY) / genericLength testY

  printf ("[test accuracy: %.1f%% / %d] All same? %s") (acc * 100 :: Float) (length testY)
    (if all (== head preds) preds then show (head preds) else "No.")
  hFlush stdout

  pure (fmap (id *** Right) test)


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
  printf "initial "
  val <- testNet net0 lval
  runEpoch val 1 net0
  where
    runEpoch :: LDataSet -> Int -> LeNet ch step -> IO (LeNet ch step)
    runEpoch val e net
      | e > mx    = pure net
      | otherwise = do
        printf "\n[Epoch %d/%d]\n" e mx
        (net', train) <- runBatches (dim :: Dim 4) lr t0 e ltrain net
        testNet net' val
        runEpoch val (e + 1) net'

mkBatches :: Int -> LDataSet -> [LDataSet]
mkBatches sz ds = DL.toList $ go mempty ds
 where
  go :: DList LDataSet -> LDataSet -> DList LDataSet
  go bs src =
    if V.null src
    then bs
    else
      let (b, nxt) = V.splitAt sz ds
      in go (bs `DL.snoc` b) nxt


runBatches
  :: forall ch step (batch::Nat) . (ch ~ 3, step ~ 5)
  => KnownDim batch
  => KnownNat batch
  => KnownDim (batch * 10)
  => KnownNat (batch * 10)

  => Dim batch
  -> HsReal
  -> UTCTime
  -> Int
  -> LDataSet
  -> LeNet ch step

  -> IO (LeNet ch step, LDataSet)
runBatches d lr t00 e lds net = do
  res <- V.ifoldM' go (net, DL.empty) lbatches
  pure $ second (V.concat . DL.toList) res
 where
  lbatches :: Vector LDataSet
  lbatches = V.fromList $ mkBatches (fromIntegral bs) lds

  btensor :: LDataSet -> IO (Maybe (Tensor '[batch, 10], Tensor '[batch, 3, 32, 32]))
  btensor lds
    | V.length lds /= (fromIntegral bs) = pure Nothing
    | otherwise = (Just . (toYs &&& toXs) . V.toList) <$> V.mapM getdata lds
    where
      toYs :: [(Category, Tensor '[3, 32, 32])] -> Tensor '[batch, 10]
      toYs = unsafeMatrix . fmap (onehotf . fst)

      toXs :: [(Category, Tensor '[3, 32, 32])] ->  Tensor '[batch, 3, 32, 32]
      toXs = catArray0 . fmap (unsqueeze1d (dim :: Dim 0) . snd)

  bs :: Word
  bs = dimVal d

  -- ifoldM' :: Monad m => (a -> Int -> b -> m a) -> a -> Vector b -> m a
  go
    :: (LeNet ch step, DList LDataSet)
    -> Int
    -> LDataSet
    -> IO (LeNet ch step, DList LDataSet)
  go (net, seen) bid lzy = do
    fcd <- V.mapM forcetensor lzy
    btensor fcd >>= \case
        Nothing -> pure (net, seen)
        Just (ys, xs) -> do
          let (net', loss) = trainStep lr net xs ys
          t1 <- getCurrentTime
          printf (setRewind ++ "(%d-batch #%d)[mse %.4f] (elapsed: %s)")
            bs (bid+1)
            (loss `get1d` 0)
            (show (t1 `diffUTCTime` t00))
          hFlush stdout
          pure (net', seen `DL.snoc` fcd)

 -- | Erase the last line in an ANSI terminal
clearLn :: IO ()
clearLn = printf "\ESC[2K"

-- | set rewind marker for 'clearLn'
setRewind :: String
setRewind = "\r"

-- | clear the screen in an ANSI terminal
clearScreen :: IO ()
clearScreen = putStr "\ESC[2J"

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
  -> (LeNet ch step, Tensor '[1])
trainStep lr net xs ys = (Bp.add net gnet, out)
  where
    out :: Tensor '[1]
    (out, (gnet, _)) = backprop2 ( mSECriterion ys .: lenetBatch lr) net xs


