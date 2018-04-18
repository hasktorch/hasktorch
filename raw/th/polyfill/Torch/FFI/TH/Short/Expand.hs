{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Short.Expand where

import Foreign
import Data.Word
import Torch.Types.TH


foreign import ccall "&THShortTensor_expand"
  c_expand_ :: Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THLongStorage -> IO ()

c_expand :: Ptr C'THState -> Ptr C'THShortTensor -> Ptr C'THShortTensor -> Ptr C'THLongStorage -> IO ()
c_expand = const c_expand_


foreign import ccall "&THShortTensor_expandNd"
  c_expandNd_ :: Ptr (Ptr C'THShortTensor) -> Ptr (Ptr C'THShortTensor) -> CInt -> IO ()

c_expandNd :: Ptr C'THState -> Ptr (Ptr C'THShortTensor) -> Ptr (Ptr C'THShortTensor) -> CInt -> IO ()
c_expandNd = const c_expandNd_


foreign import ccall "&THShortTensor_newExpand"
  c_newExpand_ :: Ptr C'THShortTensor -> Ptr C'THLongStorage -> IO (Ptr C'THShortTensor)

c_newExpand :: Ptr C'THState -> Ptr C'THShortTensor -> Ptr C'THLongStorage -> IO (Ptr C'THShortTensor)
c_newExpand = const c_newExpand_

