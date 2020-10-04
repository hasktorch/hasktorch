{-# LANGUAGE DataKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE RankNTypes #-}

module Torch.GraduallyTyped.Random where

import Control.Concurrent.STM (TVar)
import Data.Word (Word64)
import Foreign.ForeignPtr (ForeignPtr)
import GHC.TypeLits (Nat)
import Torch.GraduallyTyped.Device (Device, DeviceType)
import qualified Torch.Internal.Type as ATen

data Generator (device :: Device (DeviceType Nat)) where
  UnsafeGenerator ::
    forall device.
    { seed :: Word64,
      atenGenerator :: TVar (Maybe (ForeignPtr ATen.Generator))
    } ->
    Generator device
