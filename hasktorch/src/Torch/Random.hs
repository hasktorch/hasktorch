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
import Data.Int
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

instance Show (TVar (Either (Word64, Device) (ForeignPtr ATen.Generator))) where
  show _ = "_"

newtype Generator = UnsafeGenerator
  { unGenerator :: TVar (Either (Word64, Device) (ForeignPtr ATen.Generator))
  }
  deriving (Eq, Show)

mkGenerator :: Device -> Word64 -> IO Generator
mkGenerator device seed =
  case device of
    Device CPU _ -> do
      genPtr <- ATen.newCPUGenerator seed
      genenerator <- newTVarIO (Right genPtr)
      return $ UnsafeGenerator genenerator
    Device CUDA idx -> do
      genPtr <- ATen.newCUDAGenerator (fromIntegral idx)
      ATen.generator_set_current_seed genPtr seed
      genenerator <- newTVarIO (Right genPtr)
      return $ UnsafeGenerator genenerator

type RandomGenFunc = ForeignPtr ATen.IntArray -> ForeignPtr ATen.Generator -> ForeignPtr ATen.TensorOptions -> IO (ForeignPtr ATen.Tensor)

generatorFactory :: RandomGenFunc -> [Int] -> TensorOptions -> Generator -> (Tensor, Generator)
generatorFactory func size options (UnsafeGenerator generator) =
  unsafePerformIO $ do
    mGenerator <- atomically $ do
      v <- readTVar generator
      case v of
        Right v' -> do
          let device =
                if generatorIsCuda v'
                  then Device {deviceType = CUDA, deviceIndex = fromIntegral $ generatorDevice v'}
                  else Device {deviceType = CPU, deviceIndex = 0}
              seed = generatorSeed v'
          writeTVar generator $ seed `seq` deviceType device `seq` deviceIndex device `seq` Left (seed, device)
          return $ Right v'
        Left v -> return (Left v)
    genPtr <- case mGenerator of
      Right gen -> return gen
      Left (seed, device) -> case device of
        Device CPU _ -> ATen.newCPUGenerator seed
        Device CUDA idx -> do
          gen <- ATen.newCUDAGenerator (fromIntegral idx)
          ATen.generator_set_current_seed gen seed
          return gen
    tensor <- cast3 func size genPtr options
    nextGenenerator <- newTVarIO (Right genPtr)
    return (tensor, UnsafeGenerator nextGenenerator)
  where
    generatorIsCpu :: ForeignPtr ATen.Generator -> Bool
    generatorIsCpu gen = unsafePerformIO $ cast1 ATen.generator_is_cpu gen

    generatorIsCuda :: ForeignPtr ATen.Generator -> Bool
    generatorIsCuda gen = unsafePerformIO $ cast1 ATen.generator_is_cuda gen

    generatorDevice :: ForeignPtr ATen.Generator -> Int
    generatorDevice gen = unsafePerformIO $ cast1 ATen.generator_get_device gen

    generatorSeed :: ForeignPtr ATen.Generator -> Word64
    generatorSeed gen = unsafePerformIO $ cast1 ATen.generator_current_seed gen

randn ::
  -- | size
  [Int] ->
  -- | options
  TensorOptions ->
  -- | generator
  Generator ->
  -- | output
  (Tensor, Generator)
randn = generatorFactory LibTorch.randn_lGo

randn' ::
  -- | size
  [Int] ->
  -- | generator
  Generator ->
  -- | output
  (Tensor, Generator)
randn' size = randn size defaultOpts

rand ::
  -- | size
  [Int] ->
  -- | options
  TensorOptions ->
  -- | generator
  Generator ->
  -- | output
  (Tensor, Generator)
rand = generatorFactory LibTorch.rand_lGo

rand' ::
  -- | size
  [Int] ->
  -- | generator
  Generator ->
  -- | output
  (Tensor, Generator)
rand' size = rand size defaultOpts

randint ::
  -- | low
  Int ->
  -- | high
  Int ->
  -- | size
  [Int] ->
  -- | options
  TensorOptions ->
  -- | generator
  Generator ->
  -- | output
  (Tensor, Generator)
randint low high = generatorFactory (LibTorch.randint_lllGo (fromIntegral low) (fromIntegral high))

randint' ::
  -- | low
  Int ->
  -- | high
  Int ->
  -- | size
  [Int] ->
  -- | generator
  Generator ->
  -- | output
  (Tensor, Generator)
randint' low high size = randint low high size defaultOpts

normal ::
  -- | mean
  Double ->
  -- | std
  Double ->
  -- | size
  [Int] ->
  -- | options
  TensorOptions ->
  -- | generator
  Generator ->
  -- | output
  (Tensor, Generator)
normal mean std = generatorFactory (LibTorch.normal_ddlGo (realToFrac mean) (realToFrac std))

normal' ::
  -- | mean
  Double ->
  -- | std
  Double ->
  -- | size
  [Int] ->
  -- | generator
  Generator ->
  -- | output
  (Tensor, Generator)
normal' mean std size = normal mean std size defaultOpts
