
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}

module Torch.Internal.Managed.Type.TensorIndex where


import Foreign.C.String
import Foreign.C.Types
import Foreign
import Torch.Internal.Type
import Torch.Internal.Class
import Torch.Internal.Cast
import Torch.Internal.Objects

import qualified Torch.Internal.Unmanaged.Type.TensorIndex as Unmanaged


newTensorIndexList :: IO (ForeignPtr (StdVector TensorIndex))
newTensorIndexList = _cast0 Unmanaged.newTensorIndexList

newTensorIndexWithInt :: CInt -> IO (ForeignPtr TensorIndex)
newTensorIndexWithInt = _cast1 Unmanaged.newTensorIndexWithInt

newTensorIndexWithBool :: CBool -> IO (ForeignPtr TensorIndex)
newTensorIndexWithBool = _cast1 Unmanaged.newTensorIndexWithBool

newTensorIndexWithSlice :: CInt -> CInt -> CInt -> IO (ForeignPtr TensorIndex)
newTensorIndexWithSlice = _cast3 Unmanaged.newTensorIndexWithSlice

newTensorIndexWithTensor :: ForeignPtr Tensor -> IO (ForeignPtr TensorIndex)
newTensorIndexWithTensor = _cast1 Unmanaged.newTensorIndexWithTensor

newTensorIndexWithEllipsis :: IO (ForeignPtr TensorIndex)
newTensorIndexWithEllipsis = _cast0 Unmanaged.newTensorIndexWithEllipsis

newTensorIndexWithNone :: IO (ForeignPtr TensorIndex)
newTensorIndexWithNone = _cast0 Unmanaged.newTensorIndexWithNone

tensorIndexList_empty :: ForeignPtr (StdVector TensorIndex) -> IO (CBool)
tensorIndexList_empty = _cast1 Unmanaged.tensorIndexList_empty

tensorIndexList_size :: ForeignPtr (StdVector TensorIndex) -> IO (CSize)
tensorIndexList_size = _cast1 Unmanaged.tensorIndexList_size

tensorIndexList_push_back :: ForeignPtr (StdVector TensorIndex) -> ForeignPtr TensorIndex -> IO ()
tensorIndexList_push_back = _cast2 Unmanaged.tensorIndexList_push_back

index :: ForeignPtr Tensor -> ForeignPtr (StdVector TensorIndex) -> IO (ForeignPtr Tensor)
index = _cast2 Unmanaged.index

index_put_ :: ForeignPtr Tensor -> ForeignPtr (StdVector TensorIndex) -> ForeignPtr Tensor -> IO (ForeignPtr Tensor)
index_put_ = _cast3 Unmanaged.index_put_
