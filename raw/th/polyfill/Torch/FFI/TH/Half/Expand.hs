{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Half.Expand where

import Foreign
import Data.Word
import Torch.Types.TH


foreign import ccall "&THHalfTensor_expand"
  c_expand_ :: Ptr C'THHalfTensor -> Ptr C'THHalfTensor -> Ptr C'THLongStorage -> IO ()

c_expand :: Ptr C'THState -> Ptr C'THHalfTensor -> Ptr C'THHalfTensor -> Ptr C'THLongStorage -> IO ()
c_expand = const c_expand_


foreign import ccall "&THHalfTensor_expandNd"
  c_expandNd_ :: Ptr (Ptr C'THHalfTensor) -> Ptr (Ptr C'THHalfTensor) -> CInt -> IO ()

c_expandNd :: Ptr C'THState -> Ptr (Ptr C'THHalfTensor) -> Ptr (Ptr C'THHalfTensor) -> CInt -> IO ()
c_expandNd = const c_expandNd_


foreign import ccall "&THHalfTensor_newExpand"
  c_newExpand_ :: Ptr C'THHalfTensor -> Ptr C'THLongStorage -> IO (Ptr C'THHalfTensor)

c_newExpand :: Ptr C'THState -> Ptr C'THHalfTensor -> Ptr C'THLongStorage -> IO (Ptr C'THHalfTensor)
c_newExpand = const c_newExpand_

