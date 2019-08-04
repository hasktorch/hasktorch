
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}

module ATen.Managed.Type.Generator where


import Foreign.C.String
import Foreign.C.Types
import Foreign hiding (newForeignPtr)
import Foreign.Concurrent
import ATen.Type
import ATen.Class
import ATen.Cast
import ATen.Unmanaged.Type.Generator
import ATen.Unmanaged.Type.IntArray
import ATen.Unmanaged.Type.Scalar
import ATen.Unmanaged.Type.SparseTensorRef
import ATen.Unmanaged.Type.Storage
import ATen.Unmanaged.Type.Tensor
import ATen.Unmanaged.Type.TensorList
import ATen.Unmanaged.Type.TensorOptions
import ATen.Unmanaged.Type.Tuple

import qualified ATen.Unmanaged.Type.Generator as Unmanaged









