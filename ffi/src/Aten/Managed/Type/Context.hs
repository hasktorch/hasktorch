
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}

module Aten.Managed.Type.Context where


import Foreign.C.String
import Foreign.C.Types
import Foreign hiding (newForeignPtr)
import Foreign.Concurrent
import Aten.Type
import Aten.Class
import Aten.Cast
import Aten.Unmanaged.Type.Generator
import Aten.Unmanaged.Type.IntArray
import Aten.Unmanaged.Type.Scalar
import Aten.Unmanaged.Type.SparseTensorRef
import Aten.Unmanaged.Type.Storage
import Aten.Unmanaged.Type.Tensor
import Aten.Unmanaged.Type.TensorList
import Aten.Unmanaged.Type.TensorOptions
import Aten.Unmanaged.Type.Tuple

import qualified Aten.Unmanaged.Type.Context as Unmanaged









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

