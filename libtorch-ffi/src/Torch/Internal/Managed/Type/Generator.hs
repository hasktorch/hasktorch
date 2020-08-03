
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
newCUDAGenerator _device_index = cast1 Unmanaged.newCUDAGenerator _device_index

newCPUGenerator
  :: Word64
  -> IO (ForeignPtr Generator)
newCPUGenerator _seed_in = cast1 Unmanaged.newCPUGenerator _seed_in





generator_set_current_seed
  :: ForeignPtr Generator
  -> Word64
  -> IO ()
generator_set_current_seed = cast2 Unmanaged.generator_set_current_seed

generator_current_seed
  :: ForeignPtr Generator
  -> IO (Word64)
generator_current_seed = cast1 Unmanaged.generator_current_seed

generator_seed
  :: ForeignPtr Generator
  -> IO (Word64)
generator_seed = cast1 Unmanaged.generator_seed

generator_clone
  :: ForeignPtr Generator
  -> IO (ForeignPtr Generator)
generator_clone _obj = cast1 Unmanaged.generator_clone _obj
