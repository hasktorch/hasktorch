{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Int.Expand where

import Foreign
import Data.Word
import Torch.Types.TH


foreign import ccall "&THIntTensor_expand"
  c_expand_ :: Ptr C'THIntTensor -> Ptr C'THIntTensor -> Ptr C'THLongStorage -> IO ()

c_expand :: Ptr C'THState -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> Ptr C'THLongStorage -> IO ()
c_expand = const c_expand_


foreign import ccall "&THIntTensor_expandNd"
  c_expandNd_ :: Ptr (Ptr C'THIntTensor) -> Ptr (Ptr C'THIntTensor) -> CInt -> IO ()

c_expandNd :: Ptr C'THState -> Ptr (Ptr C'THIntTensor) -> Ptr (Ptr C'THIntTensor) -> CInt -> IO ()
c_expandNd = const c_expandNd_


foreign import ccall "&THIntTensor_newExpand"
  c_newExpand_ :: Ptr C'THIntTensor -> Ptr C'THLongStorage -> IO (Ptr C'THIntTensor)

c_newExpand :: Ptr C'THState -> Ptr C'THIntTensor -> Ptr C'THLongStorage -> IO (Ptr C'THIntTensor)
c_newExpand = const c_newExpand_

