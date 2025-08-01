{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE TypeFamilies #-}

module Torch.Internal.Managed.Type.StdOptional where

import Foreign
import Foreign.C.String
import Foreign.C.Types
import Torch.Internal.Cast
import Torch.Internal.Class
import Torch.Internal.Objects
import Torch.Internal.Type
import qualified Torch.Internal.Unmanaged.Type.StdOptional as Unmanaged

stdOptionalTensor_create :: ForeignPtr Tensor -> IO (ForeignPtr (StdOptional Tensor))
stdOptionalTensor_create tensor = _cast1 Unmanaged.stdOptionalTensor_create tensor

stdOptionalTensor_empty :: IO (ForeignPtr (StdOptional Tensor))
stdOptionalTensor_empty = _cast0 Unmanaged.stdOptionalTensor_empty

stdOptionalTensor_has_value :: ForeignPtr (StdOptional Tensor) -> IO CBool
stdOptionalTensor_has_value optionalTensor = _cast1 Unmanaged.stdOptionalTensor_has_value optionalTensor

stdOptionalTensor_value :: ForeignPtr (StdOptional Tensor) -> IO (ForeignPtr Tensor)
stdOptionalTensor_value optionalTensor = _cast1 Unmanaged.stdOptionalTensor_value optionalTensor
