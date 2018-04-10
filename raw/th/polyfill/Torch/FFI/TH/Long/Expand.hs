{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Long.Expand where

import Foreign
import Data.Word
import Torch.Types.TH


foreign import ccall "&THLongTensor_expand"
  c_expand_ :: Ptr C'THLongTensor -> Ptr C'THLongTensor -> Ptr C'THLongStorage -> IO ()

c_expand :: Ptr C'THState -> Ptr C'THLongTensor -> Ptr C'THLongTensor -> Ptr C'THLongStorage -> IO ()
c_expand = const c_expand_


foreign import ccall "&THLongTensor_expandNd"
  c_expandNd_ :: Ptr (Ptr C'THLongTensor) -> Ptr (Ptr C'THLongTensor) -> CInt -> IO ()

c_expandNd :: Ptr C'THState -> Ptr (Ptr C'THLongTensor) -> Ptr (Ptr C'THLongTensor) -> CInt -> IO ()
c_expandNd = const c_expandNd_


foreign import ccall "&THLongTensor_newExpand"
  c_newExpand_ :: Ptr C'THLongTensor -> Ptr C'THLongStorage -> IO (Ptr C'THLongTensor)

c_newExpand :: Ptr C'THState -> Ptr C'THLongTensor -> Ptr C'THLongStorage -> IO (Ptr C'THLongTensor)
c_newExpand = const c_newExpand_

