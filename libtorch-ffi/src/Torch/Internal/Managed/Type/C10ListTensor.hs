
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}

module Torch.Internal.Managed.Type.C10ListTensor where


import Foreign.C.String
import Foreign.C.Types
import Foreign hiding (newForeignPtr)
import Foreign.Concurrent
import Torch.Internal.Type
import Torch.Internal.Class
import Torch.Internal.Cast
import Torch.Internal.Unmanaged.Type.Generator
import Torch.Internal.Unmanaged.Type.IntArray
import Torch.Internal.Unmanaged.Type.Scalar
import Torch.Internal.Unmanaged.Type.Storage
import Torch.Internal.Unmanaged.Type.Tensor
import Torch.Internal.Unmanaged.Type.TensorList
import Torch.Internal.Unmanaged.Type.TensorOptions
import Torch.Internal.Unmanaged.Type.Tuple
import Torch.Internal.Unmanaged.Type.StdString
import Torch.Internal.Unmanaged.Type.Dimname
import Torch.Internal.Unmanaged.Type.DimnameList

import qualified Torch.Internal.Unmanaged.Type.C10ListTensor as Unmanaged



newC10ListTensor
  :: IO (ForeignPtr (C10List Tensor))
newC10ListTensor = cast0 Unmanaged.newC10ListTensor





c10ListTensor_empty
  :: ForeignPtr (C10List Tensor)
  -> IO (CBool)
c10ListTensor_empty = cast1 Unmanaged.c10ListTensor_empty

c10ListTensor_size
  :: ForeignPtr (C10List Tensor)
  -> IO (CSize)
c10ListTensor_size = cast1 Unmanaged.c10ListTensor_size

c10ListTensor_at_s
  :: ForeignPtr (C10List Tensor)
  -> CSize
  -> IO (ForeignPtr Tensor)
c10ListTensor_at_s = cast2 Unmanaged.c10ListTensor_at_s

c10ListTensor_push_back_t
  :: ForeignPtr (C10List Tensor)
  -> ForeignPtr Tensor
  -> IO (())
c10ListTensor_push_back_t = cast2 Unmanaged.c10ListTensor_push_back_t



