
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}

module LibTorch.ATen.Managed.Type.Context where


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
import LibTorch.ATen.Unmanaged.Type.Storage
import LibTorch.ATen.Unmanaged.Type.Tensor
import LibTorch.ATen.Unmanaged.Type.TensorList
import LibTorch.ATen.Unmanaged.Type.TensorOptions
import LibTorch.ATen.Unmanaged.Type.Tuple
import LibTorch.ATen.Unmanaged.Type.StdString
import LibTorch.ATen.Unmanaged.Type.Dimname
import LibTorch.ATen.Unmanaged.Type.DimnameList

import qualified LibTorch.ATen.Unmanaged.Type.Context as Unmanaged









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

