{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}

module LibTorch.ATen.Managed.Cast where

import Foreign.ForeignPtr
import Control.Monad

import LibTorch.ATen.Class
import LibTorch.ATen.Cast
import LibTorch.ATen.Type
import LibTorch.ATen.Managed.Type.IntArray
import LibTorch.ATen.Managed.Type.TensorList

instance Castable Int (ForeignPtr IntArray) where
  cast xs f = do
    arr <- newIntArray
    intArray_push_back_l arr $ fromIntegral xs
    f arr
  uncast xs f = do
    v <- intArray_at_s xs 0
    f (fromIntegral v)

instance Castable [Int] (ForeignPtr IntArray) where
  cast xs f = do
    arr <- newIntArray
    forM_ xs $ (intArray_push_back_l arr) . fromIntegral
    f arr
  uncast xs f = do
    len <- intArray_size xs
    -- NB: This check is necessary, because len is unsigned and it will wrap around if
    --     we subtract 1 when it's 0.
    if len == 0
      then f []
      else f =<< mapM (\i -> intArray_at_s xs i >>= return . fromIntegral) [0..(len - 1)]

instance Castable [ForeignPtr Tensor] (ForeignPtr TensorList) where
  cast xs f = do
    l <- newTensorList
    forM_ xs $ (tensorList_push_back_t l)
    f l
  uncast xs f = do
    len <- tensorList_size xs
    f =<< mapM (tensorList_at_s xs) [0..(len - 1)]
