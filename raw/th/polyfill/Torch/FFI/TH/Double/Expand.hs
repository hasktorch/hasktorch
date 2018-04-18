{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Double.Expand where

import Foreign
import Data.Word
import Torch.Types.TH


foreign import ccall "&THDoubleTensor_expand"
  c_expand_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THLongStorage -> IO ()

c_expand :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THLongStorage -> IO ()
c_expand = const c_expand_


foreign import ccall "&THDoubleTensor_expandNd"
  c_expandNd_ :: Ptr (Ptr C'THDoubleTensor) -> Ptr (Ptr C'THDoubleTensor) -> CInt -> IO ()

c_expandNd :: Ptr C'THState -> Ptr (Ptr C'THDoubleTensor) -> Ptr (Ptr C'THDoubleTensor) -> CInt -> IO ()
c_expandNd = const c_expandNd_


foreign import ccall "&THDoubleTensor_newExpand"
  c_newExpand_ :: Ptr C'THDoubleTensor -> Ptr C'THLongStorage -> IO (Ptr C'THDoubleTensor)

c_newExpand :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THLongStorage -> IO (Ptr C'THDoubleTensor)
c_newExpand = const c_newExpand_

