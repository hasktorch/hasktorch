{-# LANGUAGE TypeSynonymInstances #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Torch.Dimname where


import Foreign.ForeignPtr
import Torch.Internal.Class (Castable(..))
import qualified Torch.Internal.Const as ATen
import qualified Torch.Internal.Type as ATen
import qualified Torch.Internal.Managed.Type.IValue as ATen
import Data.String
import System.IO.Unsafe

newtype IValue = IValue (ForeignPtr ATen.IValue)

data IVLike
  = IVNone
  | IVTensor
  | IVDouble
  | IVInt
  | IVBool
  | IVTuple
  | IVIntList
  | IVDoubleList
  | IVBoolList
  | IVString
  | IVTensorList
  | IVBlob
  | IVGenericList
  | IVGenericDict
  | IVFuture
  | IVDevice
  | IVObject
  | IVUninitialized
  | IVCapsule
