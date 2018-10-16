{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TupleSections #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE CPP #-}
module Utils where

import Prelude

import Control.Arrow (second)
import Control.Monad.Trans.Except (runExceptT)
import Control.Exception.Safe (throwString)
import Data.DList (DList)
import Data.Either (fromRight)
import Data.IORef
import Data.Maybe (fromJust)
import Data.Vector (Vector)
import System.IO.Unsafe (unsafePerformIO)
import Text.Printf (printf)
import qualified Data.DList as DL
import qualified Data.Vector as V
import qualified System.Random.MWC as MWC

import Torch.Models.Vision.LeNet (_conv1, LeNet)
import qualified Torch.Double.NN.Conv2d as Conv2d (getTensors)
import Torch.Data.Loaders.Cifar10 (Category)
import Torch.Data.Loaders.Internal (rgb2torch)
import Torch.Data.Loaders.RGBVector (Normalize(NegOneToOne))
import Torch.Core.Random (newRNG)

#ifdef CUDA
import Torch.Cuda.Double hiding (Sum)
import qualified Torch.Cuda.Long as Long
import Torch.FFI.THC.TensorRandom
#else
import Torch.Double hiding (Sum)
import qualified Torch.Long as Long
#endif

lr = 0.001
bsz = (dim :: Dim 4)

bs :: Num n => n
bs = (fromIntegral $ dimVal bsz)

counter :: IORef Integer
counter = unsafePerformIO $ newIORef 0
{-# NOINLINE counter #-}

seedAll :: IO MWC.GenIO
seedAll = do
  g <- MWC.initialize (V.singleton 48)
  -- g <- MWC.initialize (V.singleton 48)
#ifdef CUDA
  withForeignPtr torchstate $ \s -> c_THCRandom_manualSeed s 42
#endif
  pure g


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

getloss :: Tensor '[1] -> HsReal
getloss = fromJust . (`get1d` 0)

getindex :: Long.Tensor '[1] -> Long.HsReal
getindex = fromJust . (`Long.get1d` 0)


-- ========================================================================= --
-- Data processing + bells and whistles for a slow loader
-- ========================================================================= --
preprocess :: FilePath -> IO (Tensor '[3, 32, 32])
preprocess f =
  runExceptT (rgb2torch NegOneToOne f) >>= \case
    Left s -> throwString s
    Right t -> do
      pure t

-- | potentially lazily loaded data point
type LDatum = (Category, Either FilePath (Tensor '[3, 32, 32]))
-- | potentially lazily loaded dataset
type LDataSet = Vector (Category, Either FilePath (Tensor '[3, 32, 32]))

-- | get something usable from a lazy datapoint
getdata :: LDatum -> IO (Category, Tensor '[3, 32, 32])
getdata (c, Right t) = pure (c, t)
getdata (c, Left fp) = (c,) <$> preprocess fp

-- | force a file into a tensor
forcetensor :: LDatum -> IO LDatum
forcetensor = \case
  (c, Left fp) -> (c,) . Right <$> preprocess fp
  tens -> pure tens

prepdata :: Functor f => f (a, b) -> f (a, Either b c)
prepdata = fmap (second Left)

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

