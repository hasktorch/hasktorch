module Torch.Raw.Storage.Copy
  ( THStorageCopy(..)
  , module X
  ) where

import Torch.Raw.Internal as X

-- CTHDoubleStorage -> CDouble
class THStorageCopy t where
  c_rawCopy :: Ptr t -> Ptr CDouble -> IO ()
  c_copy :: Ptr t -> Ptr t -> IO ()
  c_copyByte :: Ptr t -> Ptr CTHByteStorage -> IO ()
  c_copyChar :: Ptr t -> Ptr CTHCharStorage -> IO ()
  c_copyShort :: Ptr t -> Ptr CTHShortStorage -> IO ()
  c_copyInt :: Ptr t -> Ptr CTHIntStorage -> IO ()
  c_copyLong :: Ptr t -> Ptr CTHLongStorage -> IO ()
  c_copyFloat :: Ptr t -> Ptr CTHFloatStorage -> IO ()
  c_copyDouble :: Ptr t -> Ptr t -> IO ()
  c_copyHalf :: Ptr t -> Ptr CTHHalfStorage -> IO ()
  p_rawCopy :: FunPtr (Ptr t -> Ptr CDouble -> IO ())
  p_copy :: FunPtr (Ptr t -> Ptr t -> IO ())
  p_copyByte :: FunPtr (Ptr t -> Ptr CTHByteStorage -> IO ())
  p_copyChar :: FunPtr (Ptr t -> Ptr CTHCharStorage -> IO ())
  p_copyShort :: FunPtr (Ptr t -> Ptr CTHShortStorage -> IO ())
  p_copyInt :: FunPtr (Ptr t -> Ptr CTHIntStorage -> IO ())
  p_copyLong :: FunPtr (Ptr t -> Ptr CTHLongStorage -> IO ())
  p_copyFloat :: FunPtr (Ptr t -> Ptr CTHFloatStorage -> IO ())
  p_copyDouble :: FunPtr (Ptr t -> Ptr t -> IO ())
  p_copyHalf :: FunPtr (Ptr t -> Ptr CTHHalfStorage -> IO ())
