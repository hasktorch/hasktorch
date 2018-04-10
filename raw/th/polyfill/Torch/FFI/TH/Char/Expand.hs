{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Char.Expand where

import Foreign
import Data.Word
import Torch.Types.TH


foreign import ccall "&THCharTensor_expand"
  c_expand_ :: Ptr C'THCharTensor -> Ptr C'THCharTensor -> Ptr C'THLongStorage -> IO ()

c_expand :: Ptr C'THState -> Ptr C'THCharTensor -> Ptr C'THCharTensor -> Ptr C'THLongStorage -> IO ()
c_expand = const c_expand_


foreign import ccall "&THCharTensor_expandNd"
  c_expandNd_ :: Ptr (Ptr C'THCharTensor) -> Ptr (Ptr C'THCharTensor) -> CInt -> IO ()

c_expandNd :: Ptr C'THState -> Ptr (Ptr C'THCharTensor) -> Ptr (Ptr C'THCharTensor) -> CInt -> IO ()
c_expandNd = const c_expandNd_


foreign import ccall "&THCharTensor_newExpand"
  c_newExpand_ :: Ptr C'THCharTensor -> Ptr C'THLongStorage -> IO (Ptr C'THCharTensor)

c_newExpand :: Ptr C'THState -> Ptr C'THCharTensor -> Ptr C'THLongStorage -> IO (Ptr C'THCharTensor)
c_newExpand = const c_newExpand_

