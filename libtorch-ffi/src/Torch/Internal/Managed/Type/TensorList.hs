
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}

module Torch.Internal.Managed.Type.TensorList where


import Foreign.C.String
import Foreign.C.Types
import Foreign
import Torch.Internal.Type
import Torch.Internal.Class
import Torch.Internal.Cast
import Torch.Internal.Objects
import qualified Torch.Internal.Unmanaged.Type.TensorList as Unmanaged





newTensorList
  :: IO (ForeignPtr TensorList)
newTensorList = _cast0 Unmanaged.newTensorList

tensorList_empty
  :: ForeignPtr TensorList
  -> IO (CBool)
tensorList_empty = _cast1 Unmanaged.tensorList_empty

tensorList_size
  :: ForeignPtr TensorList
  -> IO (CSize)
tensorList_size = _cast1 Unmanaged.tensorList_size

tensorList_at_s
  :: ForeignPtr TensorList
  -> CSize
  -> IO (ForeignPtr Tensor)
tensorList_at_s = _cast2 Unmanaged.tensorList_at_s

tensorList_push_back_t
  :: ForeignPtr TensorList
  -> ForeignPtr Tensor
  -> IO (())
tensorList_push_back_t = _cast2 Unmanaged.tensorList_push_back_t

