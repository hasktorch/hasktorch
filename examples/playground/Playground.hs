{-# LANGUAGE LambdaCase #-}

module Main where

-- Minimal implementation

import Foreign.ForeignPtr(withForeignPtr, newForeignPtr)
import System.IO.Unsafe (unsafePerformIO)

import Torch.FFI.TH.Double.Tensor as T

import PlaygroundRawUtils

td_newWithTensor :: TensorDouble -> TensorDouble
td_newWithTensor t = unsafePerformIO $ do
  newPtr <- withForeignPtr (tdTensor t) (\tPtr -> T.c_newWithTensor tPtr)
  newFPtr <- newForeignPtr T.p_free newPtr
  pure $ TensorDouble newFPtr (dimFromRaw newPtr)
{-# NOINLINE td_newWithTensor #-}

-- |Create a new (double) tensor of specified dimensions and fill it with 0
td_new :: TensorDim Word -> TensorDouble
td_new dims = unsafePerformIO $ do
  newPtr <- tensorRaw dims 0.0
  fPtr <- newForeignPtr T.p_free newPtr
  withForeignPtr fPtr fillRaw0
  pure $ TensorDouble fPtr dims
{-# NOINLINE td_new #-}


main :: IO ()
main = do
  putStrLn "Done"

twoTensor :: IO ()
twoTensor = do
  let dim = D4 (200, 200, 200, 200)
  let t1 = td_new dim
  let t2 = td_new dim
  putStrLn "Done"

-- -- |Iteration - allocate a tensor, print a value, allocate another tensor... etc.
-- memoryTest :: TensorDim Word -> Int -> IO ()
-- memoryTest dim niter = do
--   putStrLn $ show (memSizeGB dim) ++ " GB per allocation x " ++ show niter
--   forM_ [1..niter] $ \iter -> do
--     putStr ("Iteration : " ++ show iter ++ " / ")
--     let t = td_new dim
--     x <- td_get (D4 (0, 0, 0, 0)) t
--     putStrLn $ "Printing dummy value: " ++ show x
--   putStrLn "Done"
