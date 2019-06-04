{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}

module ATen.Managed.Cast where

import Foreign.ForeignPtr
import Control.Monad

import ATen.Class
import ATen.Cast
import ATen.Type
import ATen.Managed.Type.IntArray
import ATen.Managed.Type.TensorList

instance Castable [Int] (ForeignPtr IntArray) where
  cast xs f = do
    arr <- newIntArray
    forM_ xs $ (intArray_push_back_l arr) . fromIntegral
    f arr
  uncast xs f = do
    len <- intArray_size xs
    f =<< mapM (\i -> intArray_at_s xs i >>= return . fromIntegral) [0..(len - 1)]

instance Castable [ForeignPtr Tensor] (ForeignPtr TensorList) where
  cast xs f = do
    l <- newTensorList
    forM_ xs $ (tensorList_push_back_t l)
    f l
  uncast xs f = do
    len <- tensorList_size xs
    f =<< mapM (tensorList_at_s xs) [0..(len - 1)]


instance Castable (ForeignPtr Scalar) (ForeignPtr Scalar) where
  cast x f = f x
  uncast x f = f x
