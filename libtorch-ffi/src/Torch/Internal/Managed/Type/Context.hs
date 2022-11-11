
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
import Foreign
import Torch.Internal.Type
import Torch.Internal.Class
import Torch.Internal.Cast
import Torch.Internal.Objects

import qualified Torch.Internal.Unmanaged.Type.Generator as Unmanaged
import qualified Torch.Internal.Unmanaged.Type.Context as Unmanaged









init
  :: IO (())
init = _cast0 Unmanaged.init

hasCUDA
  :: IO (CBool)
hasCUDA = _cast0 Unmanaged.hasCUDA

hasHIP
  :: IO (CBool)
hasHIP = _cast0 Unmanaged.hasHIP

hasXLA
  :: IO (CBool)
hasXLA = _cast0 Unmanaged.hasXLA

getNumGPUs
  :: IO (CSize)
getNumGPUs = _cast0 Unmanaged.getNumGPUs

hasOpenMP
  :: IO (CBool)
hasOpenMP = _cast0 Unmanaged.hasOpenMP

hasMKL
  :: IO (CBool)
hasMKL = _cast0 Unmanaged.hasMKL

hasLAPACK
  :: IO (CBool)
hasLAPACK = _cast0 Unmanaged.hasLAPACK

hasMAGMA
  :: IO (CBool)
hasMAGMA = _cast0 Unmanaged.hasMAGMA

hasMKLDNN
  :: IO (CBool)
hasMKLDNN = _cast0 Unmanaged.hasMKLDNN

manual_seed_L
  :: Word64
  -> IO (())
manual_seed_L = _cast1 Unmanaged.manual_seed_L

get_manual_seed
  :: IO (Word64)
get_manual_seed = do
  g <- Unmanaged.getDefaultCPUGenerator
  Unmanaged.generator_current_seed g
