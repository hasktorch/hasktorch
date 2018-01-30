{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE TypeSynonymInstances #-}

module Torch.Raw.Storage.Copy
  ( THStorageCopy(..)
  , module X
  ) where

import Torch.Raw.Internal as X

import qualified THByteStorageCopy as S
import qualified THDoubleStorageCopy as S
import qualified THFloatStorageCopy as S
import qualified THIntStorageCopy as S
import qualified THLongStorageCopy as S
import qualified THShortStorageCopy as S
import qualified THHalfStorageCopy as S


-- CTHDoubleStorage -> CDouble
class THStorageCopy t where
  c_rawCopy    :: Ptr t -> Ptr (HaskReal t) -> IO ()
  c_copy       :: Ptr t -> Ptr t -> IO ()
  c_copyByte   :: Ptr t -> Ptr CTHByteStorage -> IO ()
  c_copyChar   :: Ptr t -> Ptr CTHCharStorage -> IO ()
  c_copyShort  :: Ptr t -> Ptr CTHShortStorage -> IO ()
  c_copyInt    :: Ptr t -> Ptr CTHIntStorage -> IO ()
  c_copyLong   :: Ptr t -> Ptr CTHLongStorage -> IO ()
  c_copyFloat  :: Ptr t -> Ptr CTHFloatStorage -> IO ()
  c_copyDouble :: Ptr t -> Ptr CTHDoubleStorage -> IO ()
  c_copyHalf   :: Ptr t -> Ptr CTHHalfStorage -> IO ()
  p_rawCopy    :: FunPtr (Ptr t -> Ptr (HaskReal t) -> IO ())
  p_copy       :: FunPtr (Ptr t -> Ptr t -> IO ())
  p_copyByte   :: FunPtr (Ptr t -> Ptr CTHByteStorage -> IO ())
  p_copyChar   :: FunPtr (Ptr t -> Ptr CTHCharStorage -> IO ())
  p_copyShort  :: FunPtr (Ptr t -> Ptr CTHShortStorage -> IO ())
  p_copyInt    :: FunPtr (Ptr t -> Ptr CTHIntStorage -> IO ())
  p_copyLong   :: FunPtr (Ptr t -> Ptr CTHLongStorage -> IO ())
  p_copyFloat  :: FunPtr (Ptr t -> Ptr CTHFloatStorage -> IO ())
  p_copyDouble :: FunPtr (Ptr t -> Ptr CTHDoubleStorage -> IO ())
  p_copyHalf   :: FunPtr (Ptr t -> Ptr CTHHalfStorage -> IO ())

instance THStorageCopy CTHByteStorageCopy where
  c_rawCopy    = S.c_THByteStorage_rawCopy
  c_copy       = S.c_THByteStorage_copy
  c_copyByte   = S.c_THByteStorage_copyByte
  c_copyChar   = S.c_THByteStorage_copyChar
  c_copyShort  = S.c_THByteStorage_copyShort
  c_copyInt    = S.c_THByteStorage_copyInt
  c_copyLong   = S.c_THByteStorage_copyLong
  c_copyFloat  = S.c_THByteStorage_copyFloat
  c_copyDouble = S.c_THByteStorage_copyDouble
  c_copyHalf   = S.c_THByteStorage_copyHalf
  p_rawCopy    = S.p_THByteStorage_rawCopy
  p_copy       = S.p_THByteStorage_copy
  p_copyByte   = S.p_THByteStorage_copyByte
  p_copyChar   = S.p_THByteStorage_copyChar
  p_copyShort  = S.p_THByteStorage_copyShort
  p_copyInt    = S.p_THByteStorage_copyInt
  p_copyLong   = S.p_THByteStorage_copyLong
  p_copyFloat  = S.p_THByteStorage_copyFloat
  p_copyDouble = S.p_THByteStorage_copyDouble
  p_copyHalf   = S.p_THByteStorage_copyHalf

instance THStorageCopy CTHShortStorageCopy where
  c_rawCopy    = S.c_THShortStorage_rawCopy
  c_copy       = S.c_THShortStorage_copy
  c_copyByte   = S.c_THShortStorage_copyByte
  c_copyChar   = S.c_THShortStorage_copyChar
  c_copyShort  = S.c_THShortStorage_copyShort
  c_copyInt    = S.c_THShortStorage_copyInt
  c_copyLong   = S.c_THShortStorage_copyLong
  c_copyFloat  = S.c_THShortStorage_copyFloat
  c_copyDouble = S.c_THShortStorage_copyDouble
  c_copyHalf   = S.c_THShortStorage_copyHalf
  p_rawCopy    = S.p_THShortStorage_rawCopy
  p_copy       = S.p_THShortStorage_copy
  p_copyByte   = S.p_THShortStorage_copyByte
  p_copyChar   = S.p_THShortStorage_copyChar
  p_copyShort  = S.p_THShortStorage_copyShort
  p_copyInt    = S.p_THShortStorage_copyInt
  p_copyLong   = S.p_THShortStorage_copyLong
  p_copyFloat  = S.p_THShortStorage_copyFloat
  p_copyDouble = S.p_THShortStorage_copyDouble
  p_copyHalf   = S.p_THShortStorage_copyHalf

instance THStorageCopy CTHIntStorageCopy where
  c_rawCopy    = S.c_THIntStorage_rawCopy
  c_copy       = S.c_THIntStorage_copy
  c_copyByte   = S.c_THIntStorage_copyByte
  c_copyChar   = S.c_THIntStorage_copyChar
  c_copyShort  = S.c_THIntStorage_copyShort
  c_copyInt    = S.c_THIntStorage_copyInt
  c_copyLong   = S.c_THIntStorage_copyLong
  c_copyFloat  = S.c_THIntStorage_copyFloat
  c_copyDouble = S.c_THIntStorage_copyDouble
  c_copyHalf   = S.c_THIntStorage_copyHalf
  p_rawCopy    = S.p_THIntStorage_rawCopy
  p_copy       = S.p_THIntStorage_copy
  p_copyByte   = S.p_THIntStorage_copyByte
  p_copyChar   = S.p_THIntStorage_copyChar
  p_copyShort  = S.p_THIntStorage_copyShort
  p_copyInt    = S.p_THIntStorage_copyInt
  p_copyLong   = S.p_THIntStorage_copyLong
  p_copyFloat  = S.p_THIntStorage_copyFloat
  p_copyDouble = S.p_THIntStorage_copyDouble
  p_copyHalf   = S.p_THIntStorage_copyHalf

instance THStorageCopy CTHLongStorageCopy where
  c_rawCopy    = S.c_THLongStorage_rawCopy
  c_copy       = S.c_THLongStorage_copy
  c_copyByte   = S.c_THLongStorage_copyByte
  c_copyChar   = S.c_THLongStorage_copyChar
  c_copyShort  = S.c_THLongStorage_copyShort
  c_copyInt    = S.c_THLongStorage_copyInt
  c_copyLong   = S.c_THLongStorage_copyLong
  c_copyFloat  = S.c_THLongStorage_copyFloat
  c_copyDouble = S.c_THLongStorage_copyDouble
  c_copyHalf   = S.c_THLongStorage_copyHalf
  p_rawCopy    = S.p_THLongStorage_rawCopy
  p_copy       = S.p_THLongStorage_copy
  p_copyByte   = S.p_THLongStorage_copyByte
  p_copyChar   = S.p_THLongStorage_copyChar
  p_copyShort  = S.p_THLongStorage_copyShort
  p_copyInt    = S.p_THLongStorage_copyInt
  p_copyLong   = S.p_THLongStorage_copyLong
  p_copyFloat  = S.p_THLongStorage_copyFloat
  p_copyDouble = S.p_THLongStorage_copyDouble
  p_copyHalf   = S.p_THLongStorage_copyHalf

instance THStorageCopy CTHFloatStorageCopy where
  c_rawCopy    = S.c_THFloatStorage_rawCopy
  c_copy       = S.c_THFloatStorage_copy
  c_copyByte   = S.c_THFloatStorage_copyByte
  c_copyChar   = S.c_THFloatStorage_copyChar
  c_copyShort  = S.c_THFloatStorage_copyShort
  c_copyInt    = S.c_THFloatStorage_copyInt
  c_copyLong   = S.c_THFloatStorage_copyLong
  c_copyFloat  = S.c_THFloatStorage_copyFloat
  c_copyDouble = S.c_THFloatStorage_copyDouble
  c_copyHalf   = S.c_THFloatStorage_copyHalf
  p_rawCopy    = S.p_THFloatStorage_rawCopy
  p_copy       = S.p_THFloatStorage_copy
  p_copyByte   = S.p_THFloatStorage_copyByte
  p_copyChar   = S.p_THFloatStorage_copyChar
  p_copyShort  = S.p_THFloatStorage_copyShort
  p_copyInt    = S.p_THFloatStorage_copyInt
  p_copyLong   = S.p_THFloatStorage_copyLong
  p_copyFloat  = S.p_THFloatStorage_copyFloat
  p_copyDouble = S.p_THFloatStorage_copyDouble
  p_copyHalf   = S.p_THFloatStorage_copyHalf

instance THStorageCopy CTHDoubleStorageCopy where
  c_rawCopy    = S.c_THDoubleStorage_rawCopy
  c_copy       = S.c_THDoubleStorage_copy
  c_copyByte   = S.c_THDoubleStorage_copyByte
  c_copyChar   = S.c_THDoubleStorage_copyChar
  c_copyShort  = S.c_THDoubleStorage_copyShort
  c_copyInt    = S.c_THDoubleStorage_copyInt
  c_copyLong   = S.c_THDoubleStorage_copyLong
  c_copyFloat  = S.c_THDoubleStorage_copyFloat
  c_copyDouble = S.c_THDoubleStorage_copyDouble
  c_copyHalf   = S.c_THDoubleStorage_copyHalf
  p_rawCopy    = S.p_THDoubleStorage_rawCopy
  p_copy       = S.p_THDoubleStorage_copy
  p_copyByte   = S.p_THDoubleStorage_copyByte
  p_copyChar   = S.p_THDoubleStorage_copyChar
  p_copyShort  = S.p_THDoubleStorage_copyShort
  p_copyInt    = S.p_THDoubleStorage_copyInt
  p_copyLong   = S.p_THDoubleStorage_copyLong
  p_copyFloat  = S.p_THDoubleStorage_copyFloat
  p_copyDouble = S.p_THDoubleStorage_copyDouble
  p_copyHalf   = S.p_THDoubleStorage_copyHalf

instance THStorageCopy CTHHalfStorageCopy where
  c_rawCopy    = S.c_THHalfStorage_rawCopy
  c_copy       = S.c_THHalfStorage_copy
  c_copyByte   = S.c_THHalfStorage_copyByte
  c_copyChar   = S.c_THHalfStorage_copyChar
  c_copyShort  = S.c_THHalfStorage_copyShort
  c_copyInt    = S.c_THHalfStorage_copyInt
  c_copyLong   = S.c_THHalfStorage_copyLong
  c_copyFloat  = S.c_THHalfStorage_copyFloat
  c_copyDouble = S.c_THHalfStorage_copyDouble
  c_copyHalf   = S.c_THHalfStorage_copyHalf
  p_rawCopy    = S.p_THHalfStorage_rawCopy
  p_copy       = S.p_THHalfStorage_copy
  p_copyByte   = S.p_THHalfStorage_copyByte
  p_copyChar   = S.p_THHalfStorage_copyChar
  p_copyShort  = S.p_THHalfStorage_copyShort
  p_copyInt    = S.p_THHalfStorage_copyInt
  p_copyLong   = S.p_THHalfStorage_copyLong
  p_copyFloat  = S.p_THHalfStorage_copyFloat
  p_copyDouble = S.p_THHalfStorage_copyDouble
  p_copyHalf   = S.p_THHalfStorage_copyHalf
