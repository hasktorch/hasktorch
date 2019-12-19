{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}

module Torch.Random
  ( mkGenerator
  , Generator
  , randn
  , randn'
  , rand
  , rand'
  , randint
  , randint'
  , normal
  , normal'
  ) where

import Torch.Internal.Cast
import Torch.Internal.Class (Castable(..))
import qualified Torch.Internal.Const as ATen
import qualified Torch.Internal.Type as ATen
import Data.Word
import qualified Torch.Internal.Managed.Type.Generator as ATen
import qualified Torch.Internal.Managed.TensorFactories as LibTorch
import Torch.Device
import Torch.TensorOptions
import Torch.Tensor
import Foreign.ForeignPtr
import System.IO.Unsafe
import           Control.Concurrent
import           Control.Concurrent.STM
import           Control.Monad.IO.Class
import           Control.Monad.STM

instance Show (TVar (Maybe (ForeignPtr ATen.Generator))) where
  show _ = "_"

data Generator
  = Generator
  { device :: Device
  , seed :: Word64
  , generator :: TVar (Maybe (ForeignPtr ATen.Generator))
  }
  deriving (Eq,Show)

mkGenerator :: Device -> Word64 -> IO Generator
mkGenerator device seed =
  case device of
    Device CPU _ -> do
      genPtr <- ATen.newCPUGenerator seed
      genenerator <- newTVarIO (Just genPtr)
      return $ Generator device seed genenerator
    Device CUDA idx -> do
      genPtr <- ATen.newCUDAGenerator (fromIntegral idx)
      ATen.generator_set_current_seed genPtr seed
      genenerator <- newTVarIO (Just genPtr)
      return $ Generator device seed genenerator

type RandomGenFunc = ForeignPtr ATen.IntArray -> ForeignPtr ATen.Generator -> ForeignPtr ATen.TensorOptions -> IO (ForeignPtr ATen.Tensor)

generatorFactory :: RandomGenFunc -> [Int] -> TensorOptions -> Generator -> (Tensor,Generator)
generatorFactory func size options Generator{..} =
  unsafePerformIO $ do
    mGenerator <- atomically $ do
      v <- readTVar generator
      case v of
        Just v' -> do
          writeTVar generator Nothing
          return $ Just v'
        Nothing -> return Nothing
    genPtr <- case mGenerator of
                Just gen -> return gen
                Nothing -> case device of
                  Device CPU _ -> ATen.newCPUGenerator seed
                  Device CUDA idx -> do
                    gen <- ATen.newCUDAGenerator (fromIntegral idx)
                    ATen.generator_set_current_seed gen seed
                    return gen
    tensor <- (cast3 func) size genPtr options
    nextSeed <- ATen.generator_current_seed genPtr
    nextGenenerator <- newTVarIO (Just genPtr)
    return (tensor,Generator device nextSeed nextGenenerator)


randn :: [Int] -> TensorOptions -> Generator -> (Tensor,Generator)
randn size opts gen = generatorFactory LibTorch.randn_lpo size opts gen

randn' :: [Int] -> Generator -> (Tensor,Generator)
randn' size gen = randn size defaultOpts gen

rand :: [Int] -> TensorOptions -> Generator -> (Tensor,Generator)
rand size opts gen = generatorFactory LibTorch.rand_lpo size opts gen

rand' :: [Int] -> Generator -> (Tensor,Generator)
rand' size gen = rand size defaultOpts gen

randint :: Int -> Int -> [Int] -> TensorOptions -> Generator -> (Tensor,Generator)
randint low high size opts gen = generatorFactory (LibTorch.randint_lllpo (fromIntegral low) (fromIntegral high)) size opts gen

randint' :: Int -> Int -> [Int] -> Generator -> (Tensor,Generator)
randint' low high size gen = randint low high size defaultOpts gen

normal :: Double -> Double -> [Int] -> TensorOptions -> Generator -> (Tensor,Generator)
normal mean std size opts gen = generatorFactory (LibTorch.normal_ddlpo (realToFrac mean) (realToFrac std)) size opts gen

normal' :: Double -> Double -> [Int] -> Generator -> (Tensor,Generator)
normal' mean std size gen = normal mean std size defaultOpts gen
