
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}

module ATen.Managed.Type.SparseTensorRef where


import Foreign.C.String
import Foreign.C.Types
import Foreign hiding (newForeignPtr)
import Foreign.Concurrent
import LibTorch.ATen.Type
import LibTorch.ATen.Class
import LibTorch.ATen.Cast
import LibTorch.ATen.Unmanaged.Type.Generator
import LibTorch.ATen.Unmanaged.Type.IntArray
import LibTorch.ATen.Unmanaged.Type.Scalar
import ATen.Unmanaged.Type.SparseTensorRef
import LibTorch.ATen.Unmanaged.Type.Storage
import LibTorch.ATen.Unmanaged.Type.Tensor
import LibTorch.ATen.Unmanaged.Type.TensorList
import LibTorch.ATen.Unmanaged.Type.TensorOptions
import LibTorch.ATen.Unmanaged.Type.Tuple

import qualified ATen.Unmanaged.Type.SparseTensorRef as Unmanaged



newSparseTensorRef_t
  :: ForeignPtr Tensor
  -> IO (ForeignPtr SparseTensorRef)
newSparseTensorRef_t = cast1 Unmanaged.newSparseTensorRef_t







