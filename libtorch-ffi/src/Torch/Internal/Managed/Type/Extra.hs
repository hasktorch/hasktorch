
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}

module Torch.Internal.Managed.Type.Extra where


import Foreign.C.String
import Foreign.C.Types
import Foreign
import Torch.Internal.Type
import Torch.Internal.Class
import Torch.Internal.Cast
import Torch.Internal.Objects

import qualified Torch.Internal.Unmanaged.Type.Extra as Unmanaged

tensor_assign1_l
  :: ForeignPtr Tensor
  -> Int64
  -> Int64
  -> IO ()
tensor_assign1_l  = _cast3 Unmanaged.tensor_assign1_l

tensor_assign2_l
  :: ForeignPtr Tensor
  -> Int64
  -> Int64
  -> Int64
  -> IO ()
tensor_assign2_l = _cast4 Unmanaged.tensor_assign2_l

tensor_assign1_d
  :: ForeignPtr Tensor
  -> Int64
  -> CDouble
  -> IO ()
tensor_assign1_d = _cast3 Unmanaged.tensor_assign1_d

tensor_assign2_d
  :: ForeignPtr Tensor
  -> Int64
  -> Int64
  -> CDouble
  -> IO ()
tensor_assign2_d = _cast4 Unmanaged.tensor_assign2_d

tensor_assign1_t
  :: ForeignPtr Tensor
  -> Int64
  -> ForeignPtr Tensor
  -> IO ()
tensor_assign1_t  = _cast3 Unmanaged.tensor_assign1_t

tensor_assign2_t
  :: ForeignPtr Tensor
  -> Int64
  -> Int64
  -> ForeignPtr Tensor
  -> IO ()
tensor_assign2_t = _cast4 Unmanaged.tensor_assign2_t

tensor_names
  :: ForeignPtr Tensor
  -> IO (ForeignPtr DimnameList)
tensor_names = _cast1 Unmanaged.tensor_names

tensor_to_device
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
tensor_to_device = _cast2 Unmanaged.tensor_to_device

new_empty_tensor
  :: [Int]
  -> ForeignPtr TensorOptions
  -> IO (ForeignPtr Tensor)
new_empty_tensor = _cast2 Unmanaged.new_empty_tensor


tensor_dim_unsafe
  :: ForeignPtr Tensor
  -> IO (Int64)
tensor_dim_unsafe = cast1 Unmanaged.tensor_dim_unsafe

tensor_dim_c_unsafe
  :: ForeignPtr Tensor
  -> IO (Int64)
tensor_dim_c_unsafe = cast1 Unmanaged.tensor_dim_c_unsafe
                                                  
