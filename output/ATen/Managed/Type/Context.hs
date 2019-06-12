
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}

module ATen.Managed.Type.Context where


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

import qualified ATen.Unmanaged.Type.Context as Unmanaged









init
  :: IO (())
init = cast0 Unmanaged.init

hasCUDA
  :: IO (CBool)
hasCUDA = cast0 Unmanaged.hasCUDA

hasHIP
  :: IO (CBool)
hasHIP = cast0 Unmanaged.hasHIP

hasXLA
  :: IO (CBool)
hasXLA = cast0 Unmanaged.hasXLA

getNumGPUs
  :: IO (CSize)
getNumGPUs = cast0 Unmanaged.getNumGPUs

hasOpenMP
  :: IO (CBool)
hasOpenMP = cast0 Unmanaged.hasOpenMP

hasMKL
  :: IO (CBool)
hasMKL = cast0 Unmanaged.hasMKL

hasLAPACK
  :: IO (CBool)
hasLAPACK = cast0 Unmanaged.hasLAPACK

hasMAGMA
  :: IO (CBool)
hasMAGMA = cast0 Unmanaged.hasMAGMA

hasMKLDNN
  :: IO (CBool)
hasMKLDNN = cast0 Unmanaged.hasMKLDNN

manual_seed_L
  :: Word64
  -> IO (())
manual_seed_L = cast1 Unmanaged.manual_seed_L

