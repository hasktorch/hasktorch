{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Byte.Expand where

import Foreign
import Data.Word
import Torch.Types.TH


foreign import ccall "&THByteTensor_expand"
  c_expand_ :: Ptr C'THByteTensor -> Ptr C'THByteTensor -> Ptr C'THLongStorage -> IO ()

c_expand :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THByteTensor -> Ptr C'THLongStorage -> IO ()
c_expand = const c_expand_


foreign import ccall "&THByteTensor_expandNd"
  c_expandNd_ :: Ptr (Ptr C'THByteTensor) -> Ptr (Ptr C'THByteTensor) -> CInt -> IO ()

c_expandNd :: Ptr C'THState -> Ptr (Ptr C'THByteTensor) -> Ptr (Ptr C'THByteTensor) -> CInt -> IO ()
c_expandNd = const c_expandNd_


foreign import ccall "&THByteTensor_newExpand"
  c_newExpand_ :: Ptr C'THByteTensor -> Ptr C'THLongStorage -> IO (Ptr C'THByteTensor)

c_newExpand :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THLongStorage -> IO (Ptr C'THByteTensor)
c_newExpand = const c_newExpand_

