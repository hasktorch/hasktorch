{-# LANGUAGE TypeSynonymInstances #-}
{-# LANGUAGE AllowAmbiguousTypes #-}

module Torch.Raw.Tensor.Conv
  ( THTensorConv(..)
  , module X
  ) where

import Torch.Raw.Internal as X

import qualified THFloatTensorConv as T
import qualified THDoubleTensorConv as T
import qualified THIntTensorConv as T
import qualified THShortTensorConv as T
import qualified THByteTensorConv as T
import qualified THLongTensorConv as T
-- import qualified THHalfTensorConv

-- CTHDoubleTensor CDouble
class THTensorConv t where
  c_validXCorr2Dptr    :: Ptr (HaskReal t) -> (HaskReal t) -> Ptr (HaskReal t) -> CLLong -> CLLong -> Ptr (HaskReal t) -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()
  c_validConv2Dptr     :: Ptr (HaskReal t) -> (HaskReal t) -> Ptr (HaskReal t) -> CLLong -> CLLong -> Ptr (HaskReal t) -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()
  c_fullXCorr2Dptr     :: Ptr (HaskReal t) -> (HaskReal t) -> Ptr (HaskReal t) -> CLLong -> CLLong -> Ptr (HaskReal t) -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()
  c_fullConv2Dptr      :: Ptr (HaskReal t) -> (HaskReal t) -> Ptr (HaskReal t) -> CLLong -> CLLong -> Ptr (HaskReal t) -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()
  c_validXCorr2DRevptr :: Ptr (HaskReal t) -> (HaskReal t) -> Ptr (HaskReal t) -> CLLong -> CLLong -> Ptr (HaskReal t) -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()
  c_conv2DRevger       :: Ptr t -> (HaskReal t) -> (HaskReal t) -> Ptr t -> Ptr t -> CLLong -> CLLong -> IO ()
  c_conv2DRevgerm      :: Ptr t -> (HaskReal t) -> (HaskReal t) -> Ptr t -> Ptr t -> CLLong -> CLLong -> IO ()
  c_conv2Dger          :: Ptr t -> (HaskReal t) -> (HaskReal t) -> Ptr t -> Ptr t -> CLLong -> CLLong -> Ptr CChar -> Ptr CChar -> IO ()
  c_conv2Dmv           :: Ptr t -> (HaskReal t) -> (HaskReal t) -> Ptr t -> Ptr t -> CLLong -> CLLong -> Ptr CChar -> Ptr CChar -> IO ()
  c_conv2Dmm           :: Ptr t -> (HaskReal t) -> (HaskReal t) -> Ptr t -> Ptr t -> CLLong -> CLLong -> Ptr CChar -> Ptr CChar -> IO ()
  c_conv2Dmul          :: Ptr t -> (HaskReal t) -> (HaskReal t) -> Ptr t -> Ptr t -> CLLong -> CLLong -> Ptr CChar -> Ptr CChar -> IO ()
  c_conv2Dcmul         :: Ptr t -> (HaskReal t) -> (HaskReal t) -> Ptr t -> Ptr t -> CLLong -> CLLong -> Ptr CChar -> Ptr CChar -> IO ()
  c_validXCorr3Dptr    :: Ptr (HaskReal t) -> (HaskReal t) -> Ptr (HaskReal t) -> CLLong -> CLLong -> CLLong -> Ptr (HaskReal t) -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()
  c_validConv3Dptr     :: Ptr (HaskReal t) -> (HaskReal t) -> Ptr (HaskReal t) -> CLLong -> CLLong -> CLLong -> Ptr (HaskReal t) -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()
  c_fullXCorr3Dptr     :: Ptr (HaskReal t) -> (HaskReal t) -> Ptr (HaskReal t) -> CLLong -> CLLong -> CLLong -> Ptr (HaskReal t) -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()
  c_fullConv3Dptr      :: Ptr (HaskReal t) -> (HaskReal t) -> Ptr (HaskReal t) -> CLLong -> CLLong -> CLLong -> Ptr (HaskReal t) -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()
  c_validXCorr3DRevptr :: Ptr (HaskReal t) -> (HaskReal t) -> Ptr (HaskReal t) -> CLLong -> CLLong -> CLLong -> Ptr (HaskReal t) -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()
  c_conv3DRevger       :: Ptr t -> (HaskReal t) -> (HaskReal t) -> Ptr t -> Ptr t -> CLLong -> CLLong -> CLLong -> IO ()
  c_conv3Dger          :: Ptr t -> (HaskReal t) -> (HaskReal t) -> Ptr t -> Ptr t -> CLLong -> CLLong -> CLLong -> Ptr CChar -> Ptr CChar -> IO ()
  c_conv3Dmv           :: Ptr t -> (HaskReal t) -> (HaskReal t) -> Ptr t -> Ptr t -> CLLong -> CLLong -> CLLong -> Ptr CChar -> Ptr CChar -> IO ()
  c_conv3Dmul          :: Ptr t -> (HaskReal t) -> (HaskReal t) -> Ptr t -> Ptr t -> CLLong -> CLLong -> CLLong -> Ptr CChar -> Ptr CChar -> IO ()
  c_conv3Dcmul         :: Ptr t -> (HaskReal t) -> (HaskReal t) -> Ptr t -> Ptr t -> CLLong -> CLLong -> CLLong -> Ptr CChar -> Ptr CChar -> IO ()
  p_validXCorr2Dptr    :: FunPtr (Ptr (HaskReal t) -> (HaskReal t) -> Ptr (HaskReal t) -> CLLong -> CLLong -> Ptr (HaskReal t) -> CLLong -> CLLong -> CLLong -> CLLong -> IO ())
  p_validConv2Dptr     :: FunPtr (Ptr (HaskReal t) -> (HaskReal t) -> Ptr (HaskReal t) -> CLLong -> CLLong -> Ptr (HaskReal t) -> CLLong -> CLLong -> CLLong -> CLLong -> IO ())
  p_fullXCorr2Dptr     :: FunPtr (Ptr (HaskReal t) -> (HaskReal t) -> Ptr (HaskReal t) -> CLLong -> CLLong -> Ptr (HaskReal t) -> CLLong -> CLLong -> CLLong -> CLLong -> IO ())
  p_fullConv2Dptr      :: FunPtr (Ptr (HaskReal t) -> (HaskReal t) -> Ptr (HaskReal t) -> CLLong -> CLLong -> Ptr (HaskReal t) -> CLLong -> CLLong -> CLLong -> CLLong -> IO ())
  p_validXCorr2DRevptr :: FunPtr (Ptr (HaskReal t) -> (HaskReal t) -> Ptr (HaskReal t) -> CLLong -> CLLong -> Ptr (HaskReal t) -> CLLong -> CLLong -> CLLong -> CLLong -> IO ())
  p_conv2DRevger       :: FunPtr (Ptr t -> (HaskReal t) -> (HaskReal t) -> Ptr t -> Ptr t -> CLLong -> CLLong -> IO ())
  p_conv2DRevgerm      :: FunPtr (Ptr t -> (HaskReal t) -> (HaskReal t) -> Ptr t -> Ptr t -> CLLong -> CLLong -> IO ())
  p_conv2Dger          :: FunPtr (Ptr t -> (HaskReal t) -> (HaskReal t) -> Ptr t -> Ptr t -> CLLong -> CLLong -> Ptr CChar -> Ptr CChar -> IO ())
  p_conv2Dmv           :: FunPtr (Ptr t -> (HaskReal t) -> (HaskReal t) -> Ptr t -> Ptr t -> CLLong -> CLLong -> Ptr CChar -> Ptr CChar -> IO ())
  p_conv2Dmm           :: FunPtr (Ptr t -> (HaskReal t) -> (HaskReal t) -> Ptr t -> Ptr t -> CLLong -> CLLong -> Ptr CChar -> Ptr CChar -> IO ())
  p_conv2Dmul          :: FunPtr (Ptr t -> (HaskReal t) -> (HaskReal t) -> Ptr t -> Ptr t -> CLLong -> CLLong -> Ptr CChar -> Ptr CChar -> IO ())
  p_conv2Dcmul         :: FunPtr (Ptr t -> (HaskReal t) -> (HaskReal t) -> Ptr t -> Ptr t -> CLLong -> CLLong -> Ptr CChar -> Ptr CChar -> IO ())
  p_validXCorr3Dptr    :: FunPtr (Ptr (HaskReal t) -> (HaskReal t) -> Ptr (HaskReal t) -> CLLong -> CLLong -> CLLong -> Ptr (HaskReal t) -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO ())
  p_validConv3Dptr     :: FunPtr (Ptr (HaskReal t) -> (HaskReal t) -> Ptr (HaskReal t) -> CLLong -> CLLong -> CLLong -> Ptr (HaskReal t) -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO ())
  p_fullXCorr3Dptr     :: FunPtr (Ptr (HaskReal t) -> (HaskReal t) -> Ptr (HaskReal t) -> CLLong -> CLLong -> CLLong -> Ptr (HaskReal t) -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO ())
  p_fullConv3Dptr      :: FunPtr (Ptr (HaskReal t) -> (HaskReal t) -> Ptr (HaskReal t) -> CLLong -> CLLong -> CLLong -> Ptr (HaskReal t) -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO ())
  p_validXCorr3DRevptr :: FunPtr (Ptr (HaskReal t) -> (HaskReal t) -> Ptr (HaskReal t) -> CLLong -> CLLong -> CLLong -> Ptr (HaskReal t) -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO ())
  p_conv3DRevger       :: FunPtr (Ptr t -> (HaskReal t) -> (HaskReal t) -> Ptr t -> Ptr t -> CLLong -> CLLong -> CLLong -> IO ())
  p_conv3Dger          :: FunPtr (Ptr t -> (HaskReal t) -> (HaskReal t) -> Ptr t -> Ptr t -> CLLong -> CLLong -> CLLong -> Ptr CChar -> Ptr CChar -> IO ())
  p_conv3Dmv           :: FunPtr (Ptr t -> (HaskReal t) -> (HaskReal t) -> Ptr t -> Ptr t -> CLLong -> CLLong -> CLLong -> Ptr CChar -> Ptr CChar -> IO ())
  p_conv3Dmul          :: FunPtr (Ptr t -> (HaskReal t) -> (HaskReal t) -> Ptr t -> Ptr t -> CLLong -> CLLong -> CLLong -> Ptr CChar -> Ptr CChar -> IO ())
  p_conv3Dcmul         :: FunPtr (Ptr t -> (HaskReal t) -> (HaskReal t) -> Ptr t -> Ptr t -> CLLong -> CLLong -> CLLong -> Ptr CChar -> Ptr CChar -> IO ())

instance THTensorConv CTHDoubleTensor where
  c_validXCorr2Dptr    = T.c_THDoubleTensor_validXCorr2Dptr
  c_validConv2Dptr     = T.c_THDoubleTensor_validConv2Dptr
  c_fullXCorr2Dptr     = T.c_THDoubleTensor_fullXCorr2Dptr
  c_fullConv2Dptr      = T.c_THDoubleTensor_fullConv2Dptr
  c_validXCorr2DRevptr = T.c_THDoubleTensor_validXCorr2DRevptr
  c_conv2DRevger       = T.c_THDoubleTensor_conv2DRevger
  c_conv2DRevgerm      = T.c_THDoubleTensor_conv2DRevgerm
  c_conv2Dger          = T.c_THDoubleTensor_conv2Dger
  c_conv2Dmv           = T.c_THDoubleTensor_conv2Dmv
  c_conv2Dmm           = T.c_THDoubleTensor_conv2Dmm
  c_conv2Dmul          = T.c_THDoubleTensor_conv2Dmul
  c_conv2Dcmul         = T.c_THDoubleTensor_conv2Dcmul
  c_validXCorr3Dptr    = T.c_THDoubleTensor_validXCorr3Dptr
  c_validConv3Dptr     = T.c_THDoubleTensor_validConv3Dptr
  c_fullXCorr3Dptr     = T.c_THDoubleTensor_fullXCorr3Dptr
  c_fullConv3Dptr      = T.c_THDoubleTensor_fullConv3Dptr
  c_validXCorr3DRevptr = T.c_THDoubleTensor_validXCorr3DRevptr
  c_conv3DRevger       = T.c_THDoubleTensor_conv3DRevger
  c_conv3Dger          = T.c_THDoubleTensor_conv3Dger
  c_conv3Dmv           = T.c_THDoubleTensor_conv3Dmv
  c_conv3Dmul          = T.c_THDoubleTensor_conv3Dmul
  c_conv3Dcmul         = T.c_THDoubleTensor_conv3Dcmul
  p_validXCorr2Dptr    = T.p_THDoubleTensor_validXCorr2Dptr
  p_validConv2Dptr     = T.p_THDoubleTensor_validConv2Dptr
  p_fullXCorr2Dptr     = T.p_THDoubleTensor_fullXCorr2Dptr
  p_fullConv2Dptr      = T.p_THDoubleTensor_fullConv2Dptr
  p_validXCorr2DRevptr = T.p_THDoubleTensor_validXCorr2DRevptr
  p_conv2DRevger       = T.p_THDoubleTensor_conv2DRevger
  p_conv2DRevgerm      = T.p_THDoubleTensor_conv2DRevgerm
  p_conv2Dger          = T.p_THDoubleTensor_conv2Dger
  p_conv2Dmv           = T.p_THDoubleTensor_conv2Dmv
  p_conv2Dmm           = T.p_THDoubleTensor_conv2Dmm
  p_conv2Dmul          = T.p_THDoubleTensor_conv2Dmul
  p_conv2Dcmul         = T.p_THDoubleTensor_conv2Dcmul
  p_validXCorr3Dptr    = T.p_THDoubleTensor_validXCorr3Dptr
  p_validConv3Dptr     = T.p_THDoubleTensor_validConv3Dptr
  p_fullXCorr3Dptr     = T.p_THDoubleTensor_fullXCorr3Dptr
  p_fullConv3Dptr      = T.p_THDoubleTensor_fullConv3Dptr
  p_validXCorr3DRevptr = T.p_THDoubleTensor_validXCorr3DRevptr
  p_conv3DRevger       = T.p_THDoubleTensor_conv3DRevger
  p_conv3Dger          = T.p_THDoubleTensor_conv3Dger
  p_conv3Dmv           = T.p_THDoubleTensor_conv3Dmv
  p_conv3Dmul          = T.p_THDoubleTensor_conv3Dmul
  p_conv3Dcmul         = T.p_THDoubleTensor_conv3Dcmul

instance THTensorConv CTHFloatTensor where
  c_validXCorr2Dptr = T.c_THFloatTensor_validXCorr2Dptr
  c_validConv2Dptr     = T.c_THFloatTensor_validConv2Dptr
  c_fullXCorr2Dptr     = T.c_THFloatTensor_fullXCorr2Dptr
  c_fullConv2Dptr      = T.c_THFloatTensor_fullConv2Dptr
  c_validXCorr2DRevptr = T.c_THFloatTensor_validXCorr2DRevptr
  c_conv2DRevger       = T.c_THFloatTensor_conv2DRevger
  c_conv2DRevgerm      = T.c_THFloatTensor_conv2DRevgerm
  c_conv2Dger          = T.c_THFloatTensor_conv2Dger
  c_conv2Dmv           = T.c_THFloatTensor_conv2Dmv
  c_conv2Dmm           = T.c_THFloatTensor_conv2Dmm
  c_conv2Dmul          = T.c_THFloatTensor_conv2Dmul
  c_conv2Dcmul         = T.c_THFloatTensor_conv2Dcmul
  c_validXCorr3Dptr    = T.c_THFloatTensor_validXCorr3Dptr
  c_validConv3Dptr     = T.c_THFloatTensor_validConv3Dptr
  c_fullXCorr3Dptr     = T.c_THFloatTensor_fullXCorr3Dptr
  c_fullConv3Dptr      = T.c_THFloatTensor_fullConv3Dptr
  c_validXCorr3DRevptr = T.c_THFloatTensor_validXCorr3DRevptr
  c_conv3DRevger       = T.c_THFloatTensor_conv3DRevger
  c_conv3Dger          = T.c_THFloatTensor_conv3Dger
  c_conv3Dmv           = T.c_THFloatTensor_conv3Dmv
  c_conv3Dmul          = T.c_THFloatTensor_conv3Dmul
  c_conv3Dcmul         = T.c_THFloatTensor_conv3Dcmul
  p_validXCorr2Dptr    = T.p_THFloatTensor_validXCorr2Dptr
  p_validConv2Dptr     = T.p_THFloatTensor_validConv2Dptr
  p_fullXCorr2Dptr     = T.p_THFloatTensor_fullXCorr2Dptr
  p_fullConv2Dptr      = T.p_THFloatTensor_fullConv2Dptr
  p_validXCorr2DRevptr = T.p_THFloatTensor_validXCorr2DRevptr
  p_conv2DRevger       = T.p_THFloatTensor_conv2DRevger
  p_conv2DRevgerm      = T.p_THFloatTensor_conv2DRevgerm
  p_conv2Dger          = T.p_THFloatTensor_conv2Dger
  p_conv2Dmv           = T.p_THFloatTensor_conv2Dmv
  p_conv2Dmm           = T.p_THFloatTensor_conv2Dmm
  p_conv2Dmul          = T.p_THFloatTensor_conv2Dmul
  p_conv2Dcmul         = T.p_THFloatTensor_conv2Dcmul
  p_validXCorr3Dptr    = T.p_THFloatTensor_validXCorr3Dptr
  p_validConv3Dptr     = T.p_THFloatTensor_validConv3Dptr
  p_fullXCorr3Dptr     = T.p_THFloatTensor_fullXCorr3Dptr
  p_fullConv3Dptr      = T.p_THFloatTensor_fullConv3Dptr
  p_validXCorr3DRevptr = T.p_THFloatTensor_validXCorr3DRevptr
  p_conv3DRevger       = T.p_THFloatTensor_conv3DRevger
  p_conv3Dger          = T.p_THFloatTensor_conv3Dger
  p_conv3Dmv           = T.p_THFloatTensor_conv3Dmv
  p_conv3Dmul          = T.p_THFloatTensor_conv3Dmul
  p_conv3Dcmul         = T.p_THFloatTensor_conv3Dcmul

instance THTensorConv CTHIntTensor where
  c_validXCorr2Dptr = T.c_THIntTensor_validXCorr2Dptr
  c_validConv2Dptr     = T.c_THIntTensor_validConv2Dptr
  c_fullXCorr2Dptr     = T.c_THIntTensor_fullXCorr2Dptr
  c_fullConv2Dptr      = T.c_THIntTensor_fullConv2Dptr
  c_validXCorr2DRevptr = T.c_THIntTensor_validXCorr2DRevptr
  c_conv2DRevger       = T.c_THIntTensor_conv2DRevger
  c_conv2DRevgerm      = T.c_THIntTensor_conv2DRevgerm
  c_conv2Dger          = T.c_THIntTensor_conv2Dger
  c_conv2Dmv           = T.c_THIntTensor_conv2Dmv
  c_conv2Dmm           = T.c_THIntTensor_conv2Dmm
  c_conv2Dmul          = T.c_THIntTensor_conv2Dmul
  c_conv2Dcmul         = T.c_THIntTensor_conv2Dcmul
  c_validXCorr3Dptr    = T.c_THIntTensor_validXCorr3Dptr
  c_validConv3Dptr     = T.c_THIntTensor_validConv3Dptr
  c_fullXCorr3Dptr     = T.c_THIntTensor_fullXCorr3Dptr
  c_fullConv3Dptr      = T.c_THIntTensor_fullConv3Dptr
  c_validXCorr3DRevptr = T.c_THIntTensor_validXCorr3DRevptr
  c_conv3DRevger       = T.c_THIntTensor_conv3DRevger
  c_conv3Dger          = T.c_THIntTensor_conv3Dger
  c_conv3Dmv           = T.c_THIntTensor_conv3Dmv
  c_conv3Dmul          = T.c_THIntTensor_conv3Dmul
  c_conv3Dcmul         = T.c_THIntTensor_conv3Dcmul
  p_validXCorr2Dptr    = T.p_THIntTensor_validXCorr2Dptr
  p_validConv2Dptr     = T.p_THIntTensor_validConv2Dptr
  p_fullXCorr2Dptr     = T.p_THIntTensor_fullXCorr2Dptr
  p_fullConv2Dptr      = T.p_THIntTensor_fullConv2Dptr
  p_validXCorr2DRevptr = T.p_THIntTensor_validXCorr2DRevptr
  p_conv2DRevger       = T.p_THIntTensor_conv2DRevger
  p_conv2DRevgerm      = T.p_THIntTensor_conv2DRevgerm
  p_conv2Dger          = T.p_THIntTensor_conv2Dger
  p_conv2Dmv           = T.p_THIntTensor_conv2Dmv
  p_conv2Dmm           = T.p_THIntTensor_conv2Dmm
  p_conv2Dmul          = T.p_THIntTensor_conv2Dmul
  p_conv2Dcmul         = T.p_THIntTensor_conv2Dcmul
  p_validXCorr3Dptr    = T.p_THIntTensor_validXCorr3Dptr
  p_validConv3Dptr     = T.p_THIntTensor_validConv3Dptr
  p_fullXCorr3Dptr     = T.p_THIntTensor_fullXCorr3Dptr
  p_fullConv3Dptr      = T.p_THIntTensor_fullConv3Dptr
  p_validXCorr3DRevptr = T.p_THIntTensor_validXCorr3DRevptr
  p_conv3DRevger       = T.p_THIntTensor_conv3DRevger
  p_conv3Dger          = T.p_THIntTensor_conv3Dger
  p_conv3Dmv           = T.p_THIntTensor_conv3Dmv
  p_conv3Dmul          = T.p_THIntTensor_conv3Dmul
  p_conv3Dcmul         = T.p_THIntTensor_conv3Dcmul

instance THTensorConv CTHShortTensor where
  c_validXCorr2Dptr = T.c_THShortTensor_validXCorr2Dptr
  c_validConv2Dptr     = T.c_THShortTensor_validConv2Dptr
  c_fullXCorr2Dptr     = T.c_THShortTensor_fullXCorr2Dptr
  c_fullConv2Dptr      = T.c_THShortTensor_fullConv2Dptr
  c_validXCorr2DRevptr = T.c_THShortTensor_validXCorr2DRevptr
  c_conv2DRevger       = T.c_THShortTensor_conv2DRevger
  c_conv2DRevgerm      = T.c_THShortTensor_conv2DRevgerm
  c_conv2Dger          = T.c_THShortTensor_conv2Dger
  c_conv2Dmv           = T.c_THShortTensor_conv2Dmv
  c_conv2Dmm           = T.c_THShortTensor_conv2Dmm
  c_conv2Dmul          = T.c_THShortTensor_conv2Dmul
  c_conv2Dcmul         = T.c_THShortTensor_conv2Dcmul
  c_validXCorr3Dptr    = T.c_THShortTensor_validXCorr3Dptr
  c_validConv3Dptr     = T.c_THShortTensor_validConv3Dptr
  c_fullXCorr3Dptr     = T.c_THShortTensor_fullXCorr3Dptr
  c_fullConv3Dptr      = T.c_THShortTensor_fullConv3Dptr
  c_validXCorr3DRevptr = T.c_THShortTensor_validXCorr3DRevptr
  c_conv3DRevger       = T.c_THShortTensor_conv3DRevger
  c_conv3Dger          = T.c_THShortTensor_conv3Dger
  c_conv3Dmv           = T.c_THShortTensor_conv3Dmv
  c_conv3Dmul          = T.c_THShortTensor_conv3Dmul
  c_conv3Dcmul         = T.c_THShortTensor_conv3Dcmul
  p_validXCorr2Dptr    = T.p_THShortTensor_validXCorr2Dptr
  p_validConv2Dptr     = T.p_THShortTensor_validConv2Dptr
  p_fullXCorr2Dptr     = T.p_THShortTensor_fullXCorr2Dptr
  p_fullConv2Dptr      = T.p_THShortTensor_fullConv2Dptr
  p_validXCorr2DRevptr = T.p_THShortTensor_validXCorr2DRevptr
  p_conv2DRevger       = T.p_THShortTensor_conv2DRevger
  p_conv2DRevgerm      = T.p_THShortTensor_conv2DRevgerm
  p_conv2Dger          = T.p_THShortTensor_conv2Dger
  p_conv2Dmv           = T.p_THShortTensor_conv2Dmv
  p_conv2Dmm           = T.p_THShortTensor_conv2Dmm
  p_conv2Dmul          = T.p_THShortTensor_conv2Dmul
  p_conv2Dcmul         = T.p_THShortTensor_conv2Dcmul
  p_validXCorr3Dptr    = T.p_THShortTensor_validXCorr3Dptr
  p_validConv3Dptr     = T.p_THShortTensor_validConv3Dptr
  p_fullXCorr3Dptr     = T.p_THShortTensor_fullXCorr3Dptr
  p_fullConv3Dptr      = T.p_THShortTensor_fullConv3Dptr
  p_validXCorr3DRevptr = T.p_THShortTensor_validXCorr3DRevptr
  p_conv3DRevger       = T.p_THShortTensor_conv3DRevger
  p_conv3Dger          = T.p_THShortTensor_conv3Dger
  p_conv3Dmv           = T.p_THShortTensor_conv3Dmv
  p_conv3Dmul          = T.p_THShortTensor_conv3Dmul
  p_conv3Dcmul         = T.p_THShortTensor_conv3Dcmul

instance THTensorConv CTHLongTensor where
  c_validXCorr2Dptr = T.c_THLongTensor_validXCorr2Dptr
  c_validConv2Dptr     = T.c_THLongTensor_validConv2Dptr
  c_fullXCorr2Dptr     = T.c_THLongTensor_fullXCorr2Dptr
  c_fullConv2Dptr      = T.c_THLongTensor_fullConv2Dptr
  c_validXCorr2DRevptr = T.c_THLongTensor_validXCorr2DRevptr
  c_conv2DRevger       = T.c_THLongTensor_conv2DRevger
  c_conv2DRevgerm      = T.c_THLongTensor_conv2DRevgerm
  c_conv2Dger          = T.c_THLongTensor_conv2Dger
  c_conv2Dmv           = T.c_THLongTensor_conv2Dmv
  c_conv2Dmm           = T.c_THLongTensor_conv2Dmm
  c_conv2Dmul          = T.c_THLongTensor_conv2Dmul
  c_conv2Dcmul         = T.c_THLongTensor_conv2Dcmul
  c_validXCorr3Dptr    = T.c_THLongTensor_validXCorr3Dptr
  c_validConv3Dptr     = T.c_THLongTensor_validConv3Dptr
  c_fullXCorr3Dptr     = T.c_THLongTensor_fullXCorr3Dptr
  c_fullConv3Dptr      = T.c_THLongTensor_fullConv3Dptr
  c_validXCorr3DRevptr = T.c_THLongTensor_validXCorr3DRevptr
  c_conv3DRevger       = T.c_THLongTensor_conv3DRevger
  c_conv3Dger          = T.c_THLongTensor_conv3Dger
  c_conv3Dmv           = T.c_THLongTensor_conv3Dmv
  c_conv3Dmul          = T.c_THLongTensor_conv3Dmul
  c_conv3Dcmul         = T.c_THLongTensor_conv3Dcmul
  p_validXCorr2Dptr    = T.p_THLongTensor_validXCorr2Dptr
  p_validConv2Dptr     = T.p_THLongTensor_validConv2Dptr
  p_fullXCorr2Dptr     = T.p_THLongTensor_fullXCorr2Dptr
  p_fullConv2Dptr      = T.p_THLongTensor_fullConv2Dptr
  p_validXCorr2DRevptr = T.p_THLongTensor_validXCorr2DRevptr
  p_conv2DRevger       = T.p_THLongTensor_conv2DRevger
  p_conv2DRevgerm      = T.p_THLongTensor_conv2DRevgerm
  p_conv2Dger          = T.p_THLongTensor_conv2Dger
  p_conv2Dmv           = T.p_THLongTensor_conv2Dmv
  p_conv2Dmm           = T.p_THLongTensor_conv2Dmm
  p_conv2Dmul          = T.p_THLongTensor_conv2Dmul
  p_conv2Dcmul         = T.p_THLongTensor_conv2Dcmul
  p_validXCorr3Dptr    = T.p_THLongTensor_validXCorr3Dptr
  p_validConv3Dptr     = T.p_THLongTensor_validConv3Dptr
  p_fullXCorr3Dptr     = T.p_THLongTensor_fullXCorr3Dptr
  p_fullConv3Dptr      = T.p_THLongTensor_fullConv3Dptr
  p_validXCorr3DRevptr = T.p_THLongTensor_validXCorr3DRevptr
  p_conv3DRevger       = T.p_THLongTensor_conv3DRevger
  p_conv3Dger          = T.p_THLongTensor_conv3Dger
  p_conv3Dmv           = T.p_THLongTensor_conv3Dmv
  p_conv3Dmul          = T.p_THLongTensor_conv3Dmul
  p_conv3Dcmul         = T.p_THLongTensor_conv3Dcmul

instance THTensorConv CTHByteTensor where
  c_validXCorr2Dptr = T.c_THByteTensor_validXCorr2Dptr
  c_validConv2Dptr     = T.c_THByteTensor_validConv2Dptr
  c_fullXCorr2Dptr     = T.c_THByteTensor_fullXCorr2Dptr
  c_fullConv2Dptr      = T.c_THByteTensor_fullConv2Dptr
  c_validXCorr2DRevptr = T.c_THByteTensor_validXCorr2DRevptr
  c_conv2DRevger       = T.c_THByteTensor_conv2DRevger
  c_conv2DRevgerm      = T.c_THByteTensor_conv2DRevgerm
  c_conv2Dger          = T.c_THByteTensor_conv2Dger
  c_conv2Dmv           = T.c_THByteTensor_conv2Dmv
  c_conv2Dmm           = T.c_THByteTensor_conv2Dmm
  c_conv2Dmul          = T.c_THByteTensor_conv2Dmul
  c_conv2Dcmul         = T.c_THByteTensor_conv2Dcmul
  c_validXCorr3Dptr    = T.c_THByteTensor_validXCorr3Dptr
  c_validConv3Dptr     = T.c_THByteTensor_validConv3Dptr
  c_fullXCorr3Dptr     = T.c_THByteTensor_fullXCorr3Dptr
  c_fullConv3Dptr      = T.c_THByteTensor_fullConv3Dptr
  c_validXCorr3DRevptr = T.c_THByteTensor_validXCorr3DRevptr
  c_conv3DRevger       = T.c_THByteTensor_conv3DRevger
  c_conv3Dger          = T.c_THByteTensor_conv3Dger
  c_conv3Dmv           = T.c_THByteTensor_conv3Dmv
  c_conv3Dmul          = T.c_THByteTensor_conv3Dmul
  c_conv3Dcmul         = T.c_THByteTensor_conv3Dcmul
  p_validXCorr2Dptr    = T.p_THByteTensor_validXCorr2Dptr
  p_validConv2Dptr     = T.p_THByteTensor_validConv2Dptr
  p_fullXCorr2Dptr     = T.p_THByteTensor_fullXCorr2Dptr
  p_fullConv2Dptr      = T.p_THByteTensor_fullConv2Dptr
  p_validXCorr2DRevptr = T.p_THByteTensor_validXCorr2DRevptr
  p_conv2DRevger       = T.p_THByteTensor_conv2DRevger
  p_conv2DRevgerm      = T.p_THByteTensor_conv2DRevgerm
  p_conv2Dger          = T.p_THByteTensor_conv2Dger
  p_conv2Dmv           = T.p_THByteTensor_conv2Dmv
  p_conv2Dmm           = T.p_THByteTensor_conv2Dmm
  p_conv2Dmul          = T.p_THByteTensor_conv2Dmul
  p_conv2Dcmul         = T.p_THByteTensor_conv2Dcmul
  p_validXCorr3Dptr    = T.p_THByteTensor_validXCorr3Dptr
  p_validConv3Dptr     = T.p_THByteTensor_validConv3Dptr
  p_fullXCorr3Dptr     = T.p_THByteTensor_fullXCorr3Dptr
  p_fullConv3Dptr      = T.p_THByteTensor_fullConv3Dptr
  p_validXCorr3DRevptr = T.p_THByteTensor_validXCorr3DRevptr
  p_conv3DRevger       = T.p_THByteTensor_conv3DRevger
  p_conv3Dger          = T.p_THByteTensor_conv3Dger
  p_conv3Dmv           = T.p_THByteTensor_conv3Dmv
  p_conv3Dmul          = T.p_THByteTensor_conv3Dmul
  p_conv3Dcmul         = T.p_THByteTensor_conv3Dcmul
