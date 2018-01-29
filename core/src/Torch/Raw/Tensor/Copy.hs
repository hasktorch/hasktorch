{-# LANGUAGE TypeSynonymInstances #-}

module Torch.Raw.Tensor.Copy
  ( THTensorCopy(..)
  , module X
  ) where

import Torch.Raw.Internal as X

import qualified THFloatTensorCopy as T
import qualified THDoubleTensorCopy as T
import qualified THIntTensorCopy as T
import qualified THShortTensorCopy as T
import qualified THLongTensorCopy as T
import qualified THByteTensorCopy as T

-- CTHDoubleTensor
class THTensorCopy t where
  c_copy       :: Ptr t -> Ptr t -> IO ()
  c_copyByte   :: Ptr t -> Ptr CTHByteTensor -> IO ()
  c_copyChar   :: Ptr t -> Ptr CTHCharTensor -> IO ()
  c_copyShort  :: Ptr t -> Ptr CTHShortTensor -> IO ()
  c_copyInt    :: Ptr t -> Ptr CTHIntTensor -> IO ()
  c_copyLong   :: Ptr t -> Ptr CTHLongTensor -> IO ()
  c_copyFloat  :: Ptr t -> Ptr CTHFloatTensor -> IO ()
  c_copyDouble :: Ptr t -> Ptr CTHDoubleTensor -> IO ()
  c_copyHalf   :: Ptr t -> Ptr CTHHalfTensor -> IO ()
  p_copy       :: FunPtr (Ptr t -> Ptr t -> IO ())
  p_copyByte   :: FunPtr (Ptr t -> Ptr CTHByteTensor -> IO ())
  p_copyChar   :: FunPtr (Ptr t -> Ptr CTHCharTensor -> IO ())
  p_copyShort  :: FunPtr (Ptr t -> Ptr CTHShortTensor -> IO ())
  p_copyInt    :: FunPtr (Ptr t -> Ptr CTHIntTensor -> IO ())
  p_copyLong   :: FunPtr (Ptr t -> Ptr CTHLongTensor -> IO ())
  p_copyFloat  :: FunPtr (Ptr t -> Ptr CTHFloatTensor -> IO ())
  p_copyDouble :: FunPtr (Ptr t -> Ptr CTHDoubleTensor -> IO ())
  p_copyHalf   :: FunPtr (Ptr t -> Ptr CTHHalfTensor -> IO ())

instance THTensorCopy CTHFloatTensor where
  c_copy       = T.c_THFloatTensor_copy
  c_copyByte   = T.c_THFloatTensor_copyByte
  c_copyChar   = T.c_THFloatTensor_copyChar
  c_copyShort  = T.c_THFloatTensor_copyShort
  c_copyInt    = T.c_THFloatTensor_copyInt
  c_copyLong   = T.c_THFloatTensor_copyLong
  c_copyFloat  = T.c_THFloatTensor_copyFloat
  c_copyDouble = T.c_THFloatTensor_copyDouble
  c_copyHalf   = T.c_THFloatTensor_copyHalf
  p_copy       = T.p_THFloatTensor_copy
  p_copyByte   = T.p_THFloatTensor_copyByte
  p_copyChar   = T.p_THFloatTensor_copyChar
  p_copyShort  = T.p_THFloatTensor_copyShort
  p_copyInt    = T.p_THFloatTensor_copyInt
  p_copyLong   = T.p_THFloatTensor_copyLong
  p_copyFloat  = T.p_THFloatTensor_copyFloat
  p_copyDouble = T.p_THFloatTensor_copyDouble
  p_copyHalf   = T.p_THFloatTensor_copyHalf

instance THTensorCopy CTHDoubleTensor where
  c_copy       = T.c_THDoubleTensor_copy
  c_copyByte   = T.c_THDoubleTensor_copyByte
  c_copyChar   = T.c_THDoubleTensor_copyChar
  c_copyShort  = T.c_THDoubleTensor_copyShort
  c_copyInt    = T.c_THDoubleTensor_copyInt
  c_copyLong   = T.c_THDoubleTensor_copyLong
  c_copyFloat  = T.c_THDoubleTensor_copyFloat
  c_copyDouble = T.c_THDoubleTensor_copyDouble
  c_copyHalf   = T.c_THDoubleTensor_copyHalf
  p_copy       = T.p_THDoubleTensor_copy
  p_copyByte   = T.p_THDoubleTensor_copyByte
  p_copyChar   = T.p_THDoubleTensor_copyChar
  p_copyShort  = T.p_THDoubleTensor_copyShort
  p_copyInt    = T.p_THDoubleTensor_copyInt
  p_copyLong   = T.p_THDoubleTensor_copyLong
  p_copyFloat  = T.p_THDoubleTensor_copyFloat
  p_copyDouble = T.p_THDoubleTensor_copyDouble
  p_copyHalf   = T.p_THDoubleTensor_copyHalf

instance THTensorCopy CTHIntTensor where
  c_copy       = T.c_THIntTensor_copy
  c_copyByte   = T.c_THIntTensor_copyByte
  c_copyChar   = T.c_THIntTensor_copyChar
  c_copyShort  = T.c_THIntTensor_copyShort
  c_copyInt    = T.c_THIntTensor_copyInt
  c_copyLong   = T.c_THIntTensor_copyLong
  c_copyFloat  = T.c_THIntTensor_copyFloat
  c_copyDouble = T.c_THIntTensor_copyDouble
  c_copyHalf   = T.c_THIntTensor_copyHalf
  p_copy       = T.p_THIntTensor_copy
  p_copyByte   = T.p_THIntTensor_copyByte
  p_copyChar   = T.p_THIntTensor_copyChar
  p_copyShort  = T.p_THIntTensor_copyShort
  p_copyInt    = T.p_THIntTensor_copyInt
  p_copyLong   = T.p_THIntTensor_copyLong
  p_copyFloat  = T.p_THIntTensor_copyFloat
  p_copyDouble = T.p_THIntTensor_copyDouble
  p_copyHalf   = T.p_THIntTensor_copyHalf

instance THTensorCopy CTHShortTensor where
  c_copy       = T.c_THShortTensor_copy
  c_copyByte   = T.c_THShortTensor_copyByte
  c_copyChar   = T.c_THShortTensor_copyChar
  c_copyShort  = T.c_THShortTensor_copyShort
  c_copyInt    = T.c_THShortTensor_copyInt
  c_copyLong   = T.c_THShortTensor_copyLong
  c_copyFloat  = T.c_THShortTensor_copyFloat
  c_copyDouble = T.c_THShortTensor_copyDouble
  c_copyHalf   = T.c_THShortTensor_copyHalf
  p_copy       = T.p_THShortTensor_copy
  p_copyByte   = T.p_THShortTensor_copyByte
  p_copyChar   = T.p_THShortTensor_copyChar
  p_copyShort  = T.p_THShortTensor_copyShort
  p_copyInt    = T.p_THShortTensor_copyInt
  p_copyLong   = T.p_THShortTensor_copyLong
  p_copyFloat  = T.p_THShortTensor_copyFloat
  p_copyDouble = T.p_THShortTensor_copyDouble
  p_copyHalf   = T.p_THShortTensor_copyHalf

instance THTensorCopy CTHLongTensor where
  c_copy       = T.c_THLongTensor_copy
  c_copyByte   = T.c_THLongTensor_copyByte
  c_copyChar   = T.c_THLongTensor_copyChar
  c_copyShort  = T.c_THLongTensor_copyShort
  c_copyInt    = T.c_THLongTensor_copyInt
  c_copyLong   = T.c_THLongTensor_copyLong
  c_copyFloat  = T.c_THLongTensor_copyFloat
  c_copyDouble = T.c_THLongTensor_copyDouble
  c_copyHalf   = T.c_THLongTensor_copyHalf
  p_copy       = T.p_THLongTensor_copy
  p_copyByte   = T.p_THLongTensor_copyByte
  p_copyChar   = T.p_THLongTensor_copyChar
  p_copyShort  = T.p_THLongTensor_copyShort
  p_copyInt    = T.p_THLongTensor_copyInt
  p_copyLong   = T.p_THLongTensor_copyLong
  p_copyFloat  = T.p_THLongTensor_copyFloat
  p_copyDouble = T.p_THLongTensor_copyDouble
  p_copyHalf   = T.p_THLongTensor_copyHalf

instance THTensorCopy CTHByteTensor where
  c_copy       = T.c_THByteTensor_copy
  c_copyByte   = T.c_THByteTensor_copyByte
  c_copyChar   = T.c_THByteTensor_copyChar
  c_copyShort  = T.c_THByteTensor_copyShort
  c_copyInt    = T.c_THByteTensor_copyInt
  c_copyLong   = T.c_THByteTensor_copyLong
  c_copyFloat  = T.c_THByteTensor_copyFloat
  c_copyDouble = T.c_THByteTensor_copyDouble
  c_copyHalf   = T.c_THByteTensor_copyHalf
  p_copy       = T.p_THByteTensor_copy
  p_copyByte   = T.p_THByteTensor_copyByte
  p_copyChar   = T.p_THByteTensor_copyChar
  p_copyShort  = T.p_THByteTensor_copyShort
  p_copyInt    = T.p_THByteTensor_copyInt
  p_copyLong   = T.p_THByteTensor_copyLong
  p_copyFloat  = T.p_THByteTensor_copyFloat
  p_copyDouble = T.p_THByteTensor_copyDouble
  p_copyHalf   = T.p_THByteTensor_copyHalf
