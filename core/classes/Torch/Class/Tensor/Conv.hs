module Torch.Class.Tensor.Conv where

import THTypes
import Foreign
import Foreign.C.Types

class TensorConv t where
  validXCorr2Dptr    :: Ptr (HsReal t) -> HsReal t-> Ptr (HsReal t) -> Integer -> Integer -> Ptr (HsReal t) -> Integer -> Integer -> Integer -> Integer -> IO ()
  validConv2Dptr     :: Ptr (HsReal t) -> HsReal t-> Ptr (HsReal t) -> Integer -> Integer -> Ptr (HsReal t) -> Integer -> Integer -> Integer -> Integer -> IO ()
  fullXCorr2Dptr     :: Ptr (HsReal t) -> HsReal t-> Ptr (HsReal t) -> Integer -> Integer -> Ptr (HsReal t) -> Integer -> Integer -> Integer -> Integer -> IO ()
  fullConv2Dptr      :: Ptr (HsReal t) -> HsReal t-> Ptr (HsReal t) -> Integer -> Integer -> Ptr (HsReal t) -> Integer -> Integer -> Integer -> Integer -> IO ()
  validXCorr2DRevptr :: Ptr (HsReal t) -> HsReal t-> Ptr (HsReal t) -> Integer -> Integer -> Ptr (HsReal t) -> Integer -> Integer -> Integer -> Integer -> IO ()
  conv2DRevger       :: t -> HsReal t-> HsReal t-> t -> t -> Integer -> Integer -> IO ()
  conv2DRevgerm      :: t -> HsReal t-> HsReal t-> t -> t -> Integer -> Integer -> IO ()
  conv2Dger          :: t -> HsReal t-> HsReal t-> t -> t -> Integer -> Integer -> Ptr CChar -> Ptr CChar -> IO ()
  conv2Dmv           :: t -> HsReal t-> HsReal t-> t -> t -> Integer -> Integer -> Ptr CChar -> Ptr CChar -> IO ()
  conv2Dmm           :: t -> HsReal t-> HsReal t-> t -> t -> Integer -> Integer -> Ptr CChar -> Ptr CChar -> IO ()
  conv2Dmul          :: t -> HsReal t-> HsReal t-> t -> t -> Integer -> Integer -> Ptr CChar -> Ptr CChar -> IO ()
  conv2Dcmul         :: t -> HsReal t-> HsReal t-> t -> t -> Integer -> Integer -> Ptr CChar -> Ptr CChar -> IO ()
  validXCorr3Dptr    :: Ptr (HsReal t) -> HsReal t-> Ptr (HsReal t) -> Integer -> Integer -> Integer -> Ptr (HsReal t) -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> IO ()
  validConv3Dptr     :: Ptr (HsReal t) -> HsReal t-> Ptr (HsReal t) -> Integer -> Integer -> Integer -> Ptr (HsReal t) -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> IO ()
  fullXCorr3Dptr     :: Ptr (HsReal t) -> HsReal t-> Ptr (HsReal t) -> Integer -> Integer -> Integer -> Ptr (HsReal t) -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> IO ()
  fullConv3Dptr      :: Ptr (HsReal t) -> HsReal t-> Ptr (HsReal t) -> Integer -> Integer -> Integer -> Ptr (HsReal t) -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> IO ()
  validXCorr3DRevptr :: Ptr (HsReal t) -> HsReal t-> Ptr (HsReal t) -> Integer -> Integer -> Integer -> Ptr (HsReal t) -> Integer -> Integer -> Integer -> Integer -> Integer -> Integer -> IO ()
  conv3DRevger       :: t -> HsReal t-> HsReal t-> t -> t -> Integer -> Integer -> Integer -> IO ()
  conv3Dger          :: t -> HsReal t-> HsReal t-> t -> t -> Integer -> Integer -> Integer -> Ptr CChar -> Ptr CChar -> IO ()
  conv3Dmv           :: t -> HsReal t-> HsReal t-> t -> t -> Integer -> Integer -> Integer -> Ptr CChar -> Ptr CChar -> IO ()
  conv3Dmul          :: t -> HsReal t-> HsReal t-> t -> t -> Integer -> Integer -> Integer -> Ptr CChar -> Ptr CChar -> IO ()
  conv3Dcmul         :: t -> HsReal t-> HsReal t-> t -> t -> Integer -> Integer -> Integer -> Ptr CChar -> Ptr CChar -> IO ()
