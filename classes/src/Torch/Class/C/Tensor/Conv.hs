module Torch.Class.C.Tensor.Conv where

import THTypes
import Foreign
import Foreign.C.Types
import Torch.Class.C.Internal
import GHC.Int

-- TODO: The first tensor passed in should be the return Tensor -- find out the size and return this instead of mutating it with IO ()
class TensorConv t where
  {- can't write this out with HsReal, alone
  validXCorr2Dptr    :: Ptr (HsReal t) -> HsReal t-> Ptr (HsReal t) -> Int64 -> Int64 -> Ptr (HsReal t) -> Int64 -> Int64 -> Int64 -> Int64 -> IO ()
  validConv2Dptr     :: Ptr (HsReal t) -> HsReal t-> Ptr (HsReal t) -> Int64 -> Int64 -> Ptr (HsReal t) -> Int64 -> Int64 -> Int64 -> Int64 -> IO ()
  fullXCorr2Dptr     :: Ptr (HsReal t) -> HsReal t-> Ptr (HsReal t) -> Int64 -> Int64 -> Ptr (HsReal t) -> Int64 -> Int64 -> Int64 -> Int64 -> IO ()
  fullConv2Dptr      :: Ptr (HsReal t) -> HsReal t-> Ptr (HsReal t) -> Int64 -> Int64 -> Ptr (HsReal t) -> Int64 -> Int64 -> Int64 -> Int64 -> IO ()
  validXCorr2DRevptr :: Ptr (HsReal t) -> HsReal t-> Ptr (HsReal t) -> Int64 -> Int64 -> Ptr (HsReal t) -> Int64 -> Int64 -> Int64 -> Int64 -> IO ()
  -}
  conv2DRevger       :: t -> HsReal t -> HsReal t -> t -> t -> Int64 -> Int64 -> IO ()
  conv2DRevgerm      :: t -> HsReal t -> HsReal t -> t -> t -> Int64 -> Int64 -> IO ()
  conv2Dger          :: t -> HsReal t -> HsReal t -> t -> t -> Int64 -> Int64 -> [Int8] -> [Int8] -> IO ()
  conv2Dmv           :: t -> HsReal t -> HsReal t -> t -> t -> Int64 -> Int64 -> [Int8] -> [Int8] -> IO ()
  conv2Dmm           :: t -> HsReal t -> HsReal t -> t -> t -> Int64 -> Int64 -> [Int8] -> [Int8] -> IO ()
  conv2Dmul          :: t -> HsReal t -> HsReal t -> t -> t -> Int64 -> Int64 -> [Int8] -> [Int8] -> IO ()
  conv2Dcmul         :: t -> HsReal t -> HsReal t -> t -> t -> Int64 -> Int64 -> [Int8] -> [Int8] -> IO ()
  {- can't write this out with HsReal, alone
  validXCorr3Dptr    :: Ptr (HsReal t) -> HsReal t-> Ptr (HsReal t) -> Int64 -> Int64 -> Int64 -> Ptr (HsReal t) -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> IO ()
  validConv3Dptr     :: Ptr (HsReal t) -> HsReal t-> Ptr (HsReal t) -> Int64 -> Int64 -> Int64 -> Ptr (HsReal t) -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> IO ()
  fullXCorr3Dptr     :: Ptr (HsReal t) -> HsReal t-> Ptr (HsReal t) -> Int64 -> Int64 -> Int64 -> Ptr (HsReal t) -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> IO ()
  fullConv3Dptr      :: Ptr (HsReal t) -> HsReal t-> Ptr (HsReal t) -> Int64 -> Int64 -> Int64 -> Ptr (HsReal t) -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> IO ()
  validXCorr3DRevptr :: Ptr (HsReal t) -> HsReal t-> Ptr (HsReal t) -> Int64 -> Int64 -> Int64 -> Ptr (HsReal t) -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> IO ()
  -}
  conv3DRevger       :: t -> HsReal t -> HsReal t -> t -> t -> Int64 -> Int64 -> Int64 -> IO ()
  conv3Dger          :: t -> HsReal t -> HsReal t -> t -> t -> Int64 -> Int64 -> Int64 -> [Int8] -> [Int8] -> IO ()
  conv3Dmv           :: t -> HsReal t -> HsReal t -> t -> t -> Int64 -> Int64 -> Int64 -> [Int8] -> [Int8] -> IO ()
  conv3Dmul          :: t -> HsReal t -> HsReal t -> t -> t -> Int64 -> Int64 -> Int64 -> [Int8] -> [Int8] -> IO ()
  conv3Dcmul         :: t -> HsReal t -> HsReal t -> t -> t -> Int64 -> Int64 -> Int64 -> [Int8] -> [Int8] -> IO ()

