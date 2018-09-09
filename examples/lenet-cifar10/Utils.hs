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
module Utils where

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
import Torch.Cuda.Double as Math hiding (Sum)
import qualified Torch.Cuda.Double.Storage as S
import qualified Torch.Cuda.Long as Long
import Torch.FFI.THC.TensorRandom
import Foreign
import Torch.FFI.THC.State
#else
import Torch.Double as Math hiding (Sum)
import qualified Torch.Long as Long
#endif

import Torch.Models.Vision.LeNet
import Torch.Data.Loaders.Cifar10
import Torch.Data.Loaders.Internal
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

lr = 0.001
bsz = (dim :: Dim 4)

bs :: Num n => n
bs = (fromIntegral $ dimVal bsz)

counter :: IORef Integer
counter = unsafePerformIO $ newIORef 0
{-# NOINLINE counter #-}

seedAll :: IO MWC.GenIO
seedAll = do
  g <- MWC.initialize (V.singleton 42)
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

-- | potentially lazily loaded data point
type LDatum = (Category, Either FilePath (Tensor '[3, 32, 32]))
-- | potentially lazily loaded dataset
type LDataSet = Vector (Category, Either FilePath (Tensor '[3, 32, 32]))

-- | prep cifar10set
prepdata :: Vector (Category, FilePath) -> LDataSet
prepdata = fmap (second Left)

