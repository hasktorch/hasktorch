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
import Foreign
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


