{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Float.Expand where

import Foreign
import Data.Word
import Torch.Types.TH


foreign import ccall "&THFloatTensor_expand"
  c_expand_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THLongStorage -> IO ()

c_expand :: Ptr C'THState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THLongStorage -> IO ()
c_expand = const c_expand_


foreign import ccall "&THFloatTensor_expandNd"
  c_expandNd_ :: Ptr (Ptr C'THFloatTensor) -> Ptr (Ptr C'THFloatTensor) -> CInt -> IO ()

c_expandNd :: Ptr C'THState -> Ptr (Ptr C'THFloatTensor) -> Ptr (Ptr C'THFloatTensor) -> CInt -> IO ()
c_expandNd = const c_expandNd_


foreign import ccall "&THFloatTensor_newExpand"
  c_newExpand_ :: Ptr C'THFloatTensor -> Ptr C'THLongStorage -> IO (Ptr C'THFloatTensor)

c_newExpand :: Ptr C'THState -> Ptr C'THFloatTensor -> Ptr C'THLongStorage -> IO (Ptr C'THFloatTensor)
c_newExpand = const c_newExpand_

