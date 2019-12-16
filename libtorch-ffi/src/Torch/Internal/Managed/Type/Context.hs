
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}

module Torch.Internal.Managed.Type.Context where


import Foreign.C.String
import Foreign.C.Types
import Foreign hiding (newForeignPtr)
import Foreign.Concurrent
import Torch.Internal.Type
import Torch.Internal.Class
import Torch.Internal.Cast
import Torch.Internal.Unmanaged.Type.Generator
import Torch.Internal.Unmanaged.Type.IntArray
import Torch.Internal.Unmanaged.Type.Scalar
import Torch.Internal.Unmanaged.Type.Storage
import Torch.Internal.Unmanaged.Type.Tensor
import Torch.Internal.Unmanaged.Type.TensorList
import Torch.Internal.Unmanaged.Type.TensorOptions
import Torch.Internal.Unmanaged.Type.Tuple
import Torch.Internal.Unmanaged.Type.StdString
import Torch.Internal.Unmanaged.Type.Dimname
import Torch.Internal.Unmanaged.Type.DimnameList

import qualified Torch.Internal.Unmanaged.Type.Context as Unmanaged









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

