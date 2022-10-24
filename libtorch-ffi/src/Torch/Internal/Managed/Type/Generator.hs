
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}

module Torch.Internal.Managed.Type.Generator where


import Foreign.C.String
import Foreign.C.Types
import Foreign
import Torch.Internal.Type
import Torch.Internal.Class
import Torch.Internal.Cast
import Torch.Internal.Objects

import qualified Torch.Internal.Unmanaged.Type.Generator as Unmanaged



newCUDAGenerator
  :: Word16
  -> IO (ForeignPtr Generator)
newCUDAGenerator _device_index = _cast1 Unmanaged.newCUDAGenerator _device_index

newCPUGenerator
  :: Word64
  -> IO (ForeignPtr Generator)
newCPUGenerator _seed_in = _cast1 Unmanaged.newCPUGenerator _seed_in





generator_set_current_seed
  :: ForeignPtr Generator
  -> Word64
  -> IO ()
generator_set_current_seed = _cast2 Unmanaged.generator_set_current_seed

generator_current_seed
  :: ForeignPtr Generator
  -> IO (Word64)
generator_current_seed = _cast1 Unmanaged.generator_current_seed

generator_seed
  :: ForeignPtr Generator
  -> IO (Word64)
generator_seed = _cast1 Unmanaged.generator_seed

generator_clone
  :: ForeignPtr Generator
  -> IO (ForeignPtr Generator)
generator_clone _obj = _cast1 Unmanaged.generator_clone _obj

generator_get_device
  :: ForeignPtr Generator
  -> IO Int64
generator_get_device _obj = _cast1 Unmanaged.generator_get_device _obj

generator_is_cpu
  :: ForeignPtr Generator
  -> IO CBool
generator_is_cpu _obj = _cast1 Unmanaged.generator_is_cpu _obj

generator_is_cuda
  :: ForeignPtr Generator
  -> IO CBool
generator_is_cuda _obj = _cast1 Unmanaged.generator_is_cuda _obj

generator_is_hip
  :: ForeignPtr Generator
  -> IO CBool
generator_is_hip _obj = _cast1 Unmanaged.generator_is_hip _obj
