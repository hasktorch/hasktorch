{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}

module Torch.Internal.Managed.Cast where

import Control.Exception.Safe (throwIO)
import Foreign.ForeignPtr
import Foreign.C.Types
import Data.Int
import Control.Monad

import Torch.Internal.Class
import Torch.Internal.Cast
import Torch.Internal.Type
import Torch.Internal.Managed.Type.IntArray
import Torch.Internal.Managed.Type.TensorList
import Torch.Internal.Managed.Type.C10List
import Torch.Internal.Managed.Type.IValueList
import Torch.Internal.Managed.Type.C10Tuple
import Torch.Internal.Managed.Type.C10Dict
import Torch.Internal.Managed.Type.StdVector

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
    intArray_fromList arr (map fromIntegral xs)
    f arr
  uncast xs f = do
    xs <- intArray_toList xs
    f (map fromIntegral xs)

instance Castable [Double] (ForeignPtr (StdVector CDouble)) where
  cast xs f = do
    arr <- newStdVectorDouble
    forM_ xs $ (stdVectorDouble_push_back arr) . realToFrac
    f arr
  uncast xs f = do
    len <- stdVectorDouble_size xs
    -- NB: This check is necessary, because len is unsigned and it will wrap around if
    --     we subtract 1 when it's 0.
    if len == 0
      then f []
      else f =<< mapM (\i -> stdVectorDouble_at xs i >>= return . realToFrac) [0..(len - 1)]

instance Castable [ForeignPtr Tensor] (ForeignPtr TensorList) where
  cast xs f = do
    l <- newTensorList
    forM_ xs $ (tensorList_push_back_t l)
    f l
  uncast xs f = do
    len <- tensorList_size xs
    f =<< mapM (tensorList_at_s xs) [0..(len - 1)]

instance Castable [ForeignPtr Tensor] (ForeignPtr (C10List Tensor)) where
  cast xs f = do
    l <- newC10ListTensor
    forM_ xs $ (c10ListTensor_push_back l)
    f l
  uncast xs f = do
    len <- c10ListTensor_size xs
    f =<< mapM (c10ListTensor_at xs) [0..(len - 1)]

instance Castable [ForeignPtr Tensor] (ForeignPtr (C10List (C10Optional Tensor))) where
  cast xs f = do
    l <- newC10ListOptionalTensor
    forM_ xs $ (c10ListOptionalTensor_push_back l)
    f l
  uncast xs f = do
    len <- c10ListOptionalTensor_size xs
    f =<< mapM (c10ListOptionalTensor_at xs) [0..(len - 1)]

instance Castable [CDouble] (ForeignPtr (C10List CDouble)) where
  cast xs f = do
    l <- newC10ListDouble
    forM_ xs $ (c10ListDouble_push_back l)
    f l
  uncast xs f = do
    len <- c10ListDouble_size xs
    f =<< mapM (c10ListDouble_at xs) [0..(len - 1)]

instance Castable [Int64] (ForeignPtr (C10List Int64)) where
  cast xs f = do
    l <- newC10ListInt
    forM_ xs $ (c10ListInt_push_back l)
    f l
  uncast xs f = do
    len <- c10ListInt_size xs
    f =<< mapM (c10ListInt_at xs) [0..(len - 1)]

instance Castable [CBool] (ForeignPtr (C10List CBool)) where
  cast xs f = do
    l <- newC10ListBool
    forM_ xs $ (c10ListBool_push_back l)
    f l
  uncast xs f = do
    len <- c10ListBool_size xs
    f =<< mapM (c10ListBool_at xs) [0..(len - 1)]

instance Castable [ForeignPtr IValue] (ForeignPtr IValueList) where
  cast xs f = do
    l <- newIValueList
    forM_ xs $ (ivalueList_push_back l)
    f l
  uncast xs f = do
    len <- ivalueList_size xs
    f =<< mapM (ivalueList_at xs) [0..(len - 1)]

instance Castable [ForeignPtr IValue] (ForeignPtr (C10Ptr IVTuple)) where
  cast xs f = do
    cast xs $ \ivalueList -> do
      f =<< newC10Tuple_tuple ivalueList
  uncast xs f = do
    len <- c10Tuple_size xs
    f =<< mapM (c10Tuple_at xs) [0..(len - 1)]

instance Castable [ForeignPtr IValue] (ForeignPtr (C10List IValue)) where
  cast [] _ = throwIO $ userError "[ForeignPtr IValue]'s length must be one or more."
  cast xs f = do
    l <- newC10ListIValue (head xs)
    forM_ xs $ (c10ListIValue_push_back l)
    f l
  uncast xs f = do
    len <- c10ListIValue_size xs
    f =<< mapM (c10ListIValue_at xs) [0..(len - 1)]

instance Castable [(ForeignPtr IValue,ForeignPtr IValue)] (ForeignPtr (C10Dict '(IValue,IValue))) where
  cast [] _ = throwIO $ userError "[(ForeignPtr IValue,ForeignPtr IValue)]'s length must be one or more."
  cast xs f = do
    let (k,v) = (head xs)
    l <- newC10Dict k v
    forM_ xs $ \(k,v) -> (c10Dict_insert l k v)
    f l
  uncast xs f = f =<< c10Dict_toList xs
