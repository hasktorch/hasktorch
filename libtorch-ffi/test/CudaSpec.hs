{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE DataKinds #-}
module CudaSpec (main, spec) where

import Test.Hspec
import Control.Exception.Safe (bracket,catch,throwIO)
import Control.Monad (forM_,forM)
import Data.Int
import Foreign
import Torch.Internal.Const
import Torch.Internal.Type
import Torch.Internal.Managed.Type.TensorOptions
import Torch.Internal.Managed.Type.Tensor
import Torch.Internal.Managed.Type.IntArray
import Torch.Internal.Managed.Type.Context
import Torch.Internal.Managed.Native
import Torch.Internal.GC

main :: IO ()
main = hspec spec

spec :: Spec
spec = do
  describe "CudaSpec" $ do
    it "When CUDA is out of memory, do GC and retry" $ do
      flag <- hasCUDA
      monitorMemory $ do
        forM_ [0..1000] $ \i -> do -- 80MByte x 1000 = 80GByte
          dims <- fromList [1000,1000,10] -- 8 byte x 10M = 80MByte
          to <- device_D $ if flag == 0 then kCPU else kCUDA
          tod <- tensorOptions_dtype_s to kDouble
          zeros_lo dims tod
          return ()

fromList :: [Int64] -> IO (ForeignPtr IntArray)
fromList dims = do
  ary <- newIntArray
  forM_ dims $ intArray_push_back_l ary
  return ary
