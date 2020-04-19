{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE TypeFamilies #-}

module Torch.Internal.Managed.Type.Scalar where

import Foreign hiding (newForeignPtr)
import Foreign.C.String
import Foreign.C.Types
import Foreign.Concurrent
import Torch.Internal.Cast
import Torch.Internal.Class
import Torch.Internal.Type
import Torch.Internal.Unmanaged.Type.Dimname
import Torch.Internal.Unmanaged.Type.DimnameList
import Torch.Internal.Unmanaged.Type.Generator
import Torch.Internal.Unmanaged.Type.IntArray
import Torch.Internal.Unmanaged.Type.Scalar
import qualified Torch.Internal.Unmanaged.Type.Scalar as Unmanaged
import Torch.Internal.Unmanaged.Type.StdString
import Torch.Internal.Unmanaged.Type.Storage
import Torch.Internal.Unmanaged.Type.Tensor
import Torch.Internal.Unmanaged.Type.TensorList
import Torch.Internal.Unmanaged.Type.TensorOptions
import Torch.Internal.Unmanaged.Type.Tuple

newScalar ::
  IO (ForeignPtr Scalar)
newScalar = cast0 Unmanaged.newScalar

newScalar_i ::
  CInt ->
  IO (ForeignPtr Scalar)
newScalar_i = cast1 Unmanaged.newScalar_i

newScalar_d ::
  CDouble ->
  IO (ForeignPtr Scalar)
newScalar_d = cast1 Unmanaged.newScalar_d
