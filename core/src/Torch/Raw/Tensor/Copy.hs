module Torch.Raw.Tensor.Copy
  ( THTensorCopy(..)
  , module X
  ) where

import Torch.Raw.Internal as X

-- CTHDoubleTensor
class THTensorCopy t where
  c_copy :: Ptr t -> Ptr t -> IO ()
  c_copyByte :: Ptr t -> Ptr CTHByteTensor -> IO ()
  c_copyChar :: Ptr t -> Ptr CTHCharTensor -> IO ()
  c_copyShort :: Ptr t -> Ptr CTHShortTensor -> IO ()
  c_copyInt :: Ptr t -> Ptr CTHIntTensor -> IO ()
  c_copyLong :: Ptr t -> Ptr CTHLongTensor -> IO ()
  c_copyFloat :: Ptr t -> Ptr CTHFloatTensor -> IO ()
  c_copyDouble :: Ptr t -> Ptr t -> IO ()
  c_copyHalf :: Ptr t -> Ptr CTHHalfTensor -> IO ()
  p_copy :: FunPtr (Ptr t -> Ptr t -> IO ())
  p_copyByte :: FunPtr (Ptr t -> Ptr CTHByteTensor -> IO ())
  p_copyChar :: FunPtr (Ptr t -> Ptr CTHCharTensor -> IO ())
  p_copyShort :: FunPtr (Ptr t -> Ptr CTHShortTensor -> IO ())
  p_copyInt :: FunPtr (Ptr t -> Ptr CTHIntTensor -> IO ())
  p_copyLong :: FunPtr (Ptr t -> Ptr CTHLongTensor -> IO ())
  p_copyFloat :: FunPtr (Ptr t -> Ptr CTHFloatTensor -> IO ())
  p_copyDouble :: FunPtr (Ptr t -> Ptr t -> IO ())
  p_copyHalf :: FunPtr (Ptr t -> Ptr CTHHalfTensor -> IO ())
