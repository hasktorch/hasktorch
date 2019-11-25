{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}

module Torch.Random where

import ATen.Cast
import ATen.Class (Castable(..))
import qualified ATen.Const as ATen
import qualified ATen.Type as ATen
import Data.Word
import qualified ATen.Managed.Type.Generator as ATen
import qualified Torch.Managed.Native as LibTorch
import Torch.Device
import Torch.TensorOptions
import Torch.Tensor
import Foreign.ForeignPtr
import System.IO.Unsafe

data Generator
  = Generator
  { device :: Device
  , seed :: Word64
  }
  deriving (Eq, Show)

cpuGenerator :: Word64 -> Generator
cpuGenerator seed = Generator (Device CPU 0) seed

type RandomGenFunc = ForeignPtr ATen.IntArray -> ForeignPtr ATen.Generator -> ForeignPtr ATen.TensorOptions -> IO (ForeignPtr ATen.Tensor)

generatorFactory :: RandomGenFunc -> [Int] -> TensorOptions -> Generator -> (Tensor,Generator)
generatorFactory func size options Generator{..} =
  unsafePerformIO $ do
    gen <- case device of
             Device CPU _ -> ATen.newCPUGenerator seed
             Device CUDA idx -> do
               gen <- ATen.newCUDAGenerator (fromIntegral idx)
               ATen.generator_set_current_seed gen seed
               return gen
    tensor <- (cast3 func) size gen options
    nextSeed <- ATen.generator_current_seed gen
    return (tensor,Generator device nextSeed)


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
