{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}

module Torch.Random
  ( mkGenerator,
    Generator,
    randn,
    randn',
    rand,
    rand',
    randint,
    randint',
    normal,
    normal',
  )
where

import Control.Concurrent
import Control.Concurrent.STM
import Control.Monad.IO.Class
import Control.Monad.STM
import Data.Word
import Foreign.ForeignPtr
import System.IO.Unsafe
import Torch.Device
import Torch.Internal.Cast
import Torch.Internal.Class (Castable (..))
import qualified Torch.Internal.Const as ATen
import qualified Torch.Internal.Managed.TensorFactories as LibTorch
import qualified Torch.Internal.Managed.Type.Generator as ATen
import qualified Torch.Internal.Type as ATen
import Torch.Tensor
import Torch.TensorOptions

instance Show (TVar (Maybe (ForeignPtr ATen.Generator))) where
  show _ = "_"

data Generator
  = Generator
      { device :: Device,
        seed :: Word64,
        generator :: TVar (Maybe (ForeignPtr ATen.Generator))
      }
  deriving (Eq, Show)

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

generatorFactory :: RandomGenFunc -> [Int] -> TensorOptions -> Generator -> (Tensor, Generator)
generatorFactory func size options Generator {..} =
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
    tensor <- cast3 func size genPtr options
    nextSeed <- ATen.generator_current_seed genPtr
    nextGenenerator <- newTVarIO (Just genPtr)
    return (tensor, Generator device nextSeed nextGenenerator)

randn :: [Int] -> TensorOptions -> Generator -> (Tensor, Generator)
randn = generatorFactory LibTorch.randn_lGo

randn' :: [Int] -> Generator -> (Tensor, Generator)
randn' size = randn size defaultOpts

rand :: [Int] -> TensorOptions -> Generator -> (Tensor, Generator)
rand = generatorFactory LibTorch.rand_lGo

rand' :: [Int] -> Generator -> (Tensor, Generator)
rand' size = rand size defaultOpts

randint :: Int -> Int -> [Int] -> TensorOptions -> Generator -> (Tensor, Generator)
randint low high = generatorFactory (LibTorch.randint_lllGo (fromIntegral low) (fromIntegral high))

randint' :: Int -> Int -> [Int] -> Generator -> (Tensor, Generator)
randint' low high size = randint low high size defaultOpts

normal :: Double -> Double -> [Int] -> TensorOptions -> Generator -> (Tensor, Generator)
normal mean std = generatorFactory (LibTorch.normal_ddlGo (realToFrac mean) (realToFrac std))

normal' :: Double -> Double -> [Int] -> Generator -> (Tensor, Generator)
normal' mean std size = normal mean std size defaultOpts
