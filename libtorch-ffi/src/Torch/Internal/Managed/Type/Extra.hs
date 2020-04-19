{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE TypeFamilies #-}

module Torch.Internal.Managed.Type.Extra where

import Foreign hiding (newForeignPtr)
import Foreign.C.String
import Foreign.C.Types
import Foreign.Concurrent
import Torch.Internal.Cast
import Torch.Internal.Class
import Torch.Internal.Type
import qualified Torch.Internal.Unmanaged.Type.Extra as Unmanaged
import Torch.Internal.Unmanaged.Type.Generator
import Torch.Internal.Unmanaged.Type.IntArray
import Torch.Internal.Unmanaged.Type.Scalar
import Torch.Internal.Unmanaged.Type.Storage
import Torch.Internal.Unmanaged.Type.Tensor
import Torch.Internal.Unmanaged.Type.TensorList
import Torch.Internal.Unmanaged.Type.TensorOptions
import Torch.Internal.Unmanaged.Type.Tuple

tensor_assign1_l ::
  ForeignPtr Tensor ->
  Int64 ->
  Int64 ->
  IO ()
tensor_assign1_l = cast3 Unmanaged.tensor_assign1_l

tensor_assign2_l ::
  ForeignPtr Tensor ->
  Int64 ->
  Int64 ->
  Int64 ->
  IO ()
tensor_assign2_l = cast4 Unmanaged.tensor_assign2_l

tensor_assign1_d ::
  ForeignPtr Tensor ->
  Int64 ->
  CDouble ->
  IO ()
tensor_assign1_d = cast3 Unmanaged.tensor_assign1_d

tensor_assign2_d ::
  ForeignPtr Tensor ->
  Int64 ->
  Int64 ->
  CDouble ->
  IO ()
tensor_assign2_d = cast4 Unmanaged.tensor_assign2_d

tensor_assign1_t ::
  ForeignPtr Tensor ->
  Int64 ->
  ForeignPtr Tensor ->
  IO ()
tensor_assign1_t = cast3 Unmanaged.tensor_assign1_t

tensor_assign2_t ::
  ForeignPtr Tensor ->
  Int64 ->
  Int64 ->
  ForeignPtr Tensor ->
  IO ()
tensor_assign2_t = cast4 Unmanaged.tensor_assign2_t
