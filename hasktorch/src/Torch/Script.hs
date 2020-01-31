{-# LANGUAGE TypeSynonymInstances #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE LambdaCase #-}

module Torch.Script where

import Control.Monad (forM_, forM)
import Control.Exception.Safe (throwIO)
import Foreign.ForeignPtr
import Foreign.Ptr
import Foreign.Storable
import Foreign.C.Types
import System.IO.Unsafe
import Data.Int (Int16, Int64)
import Data.Word (Word8)
import Data.List (intercalate)
import Data.Proxy
import Data.Reflection
import Numeric

import Torch.Internal.Cast
import Torch.Internal.Class (Castable(..), CppTuple2(..), CppTuple3(..), CppTuple4(..), CppObject(..))
import qualified Torch.Internal.Unmanaged.Type.Tensor as Unmanaged (tensor_data_ptr)
import qualified Torch.Internal.Managed.Type.Context as ATen
import qualified Torch.Internal.Managed.Type.Tensor as ATen
import qualified Torch.Internal.Managed.Type.TensorOptions as ATen
import qualified Torch.Internal.Managed.Type.StdArray as ATen
import qualified Torch.Internal.Managed.Type.StdString as ATen
import qualified Torch.Internal.Managed.Native as ATen
import qualified Torch.Internal.Managed.Cast as ATen
import qualified Torch.Internal.Type as ATen
import qualified Torch.Internal.Const as ATen
import Torch.Internal.Unmanaged.Type.IValue (IValueLike(..))
import Torch.Internal.Managed.Type.IValue
import qualified Torch.Internal.Managed.Type.Module as LibTorch

import Torch.Device
import Torch.DType
import Torch.Tensor (Tensor(..))
import Torch.TensorOptions

newtype Module = UnsafeModule (ForeignPtr ATen.Module)
type RawIValue = ForeignPtr ATen.IValue
newtype Blob = UnsafeBlob (ForeignPtr (ATen.C10Ptr ATen.Blob))
newtype Object = UnsafeObject (ForeignPtr (ATen.C10Ptr ATen.IVObject))
newtype Future = UnsafeFuture (ForeignPtr (ATen.C10Ptr ATen.IVFuture))
newtype Capsule = UnsafeCapsule (ForeignPtr (ATen.C10Ptr ATen.Capsule))

data IValue
  = IVNone
  | IVTensor Tensor
  | IVDouble Double
  | IVInt Int64
  | IVBool Bool
  | IVTuple [IValue]
  | IVIntList [Int64]
  | IVDoubleList [Double]
  | IVBoolList [Bool]
  | IVString String
  | IVTensorList [Tensor]
  | IVBlob Blob
  | IVGenericList [IValue]
  | IVGenericDict [(IValue,IValue)]
  | IVFuture Future
  | IVDevice Device
  | IVObject Object
  | IVUninitialized
  | IVCapsule Capsule

instance Castable Module (ForeignPtr ATen.Module) where
  cast (UnsafeModule obj) f = f obj
  uncast obj f = f $ UnsafeModule obj

{-
instance (IValueLike a (ForeignPtr ATen.IValue))
  => IValueLike a RawIValue where
  toIValue x = cast1 (toIValue :: a -> IO (ForeignPtr ATen.IValue)) x
  fromIValue x = cast1 (fromIValue :: ForeignPtr ATen.IValue -> IO a) x

instance (CppObject a, IValueLike (ForeignPtr a) (ForeignPtr ATen.IValue))
  =>  IValueLike (ForeignPtr a) RawIValue where
  toIValue x = cast1 (toIValue :: ForeignPtr a -> IO (ForeignPtr ATen.IValue)) x
  fromIValue x = cast1 (fromIValue :: ForeignPtr ATen.IValue -> IO (ForeignPtr a)) x
-}

save :: Module -> FilePath -> IO ()
save = cast2 LibTorch.save

load :: FilePath -> IO Module
load = cast1 LibTorch.load

forward :: Module -> [RawIValue] -> IO RawIValue
forward = cast2 LibTorch.forward

instance Castable IValue RawIValue where
--  cast (IVNone) f = f "None"
  cast (IVTensor (Unsafe v)) f = toIValue v>>= f
  cast (IVDouble v) f = toIValue v >>= f
  cast (IVInt v) f = toIValue v >>= f
  cast (IVBool v) f = toIValue v >>= f
  cast (IVTuple v) f = toIValue v >>= f
--  cast (IVIntList v) f = toIValue v >>= f
--  cast (IVDoubleList v) f = toIValue v >>= f
--  cast (IVBoolList v) f = toIValue v >>= f
--  cast (IVString v) f = toIValue v >>= f
--  cast (IVTensorList v) f = toIValue v >>= f
--  cast (IVBlob v) f = toIValue v >>= f
--  cast (IVGenericList v) f = toIValue v >>= f
--  cast (IVGenericDict v) f = toIValue v >>= f
--  cast (IVFuture v) f = toIValue v >>= f
--  cast (IVDevice v) f = toIValue v >>= f
--  cast (IVObject v) f = toIValue v >>= f
  --  cast (IVUninitialized) f = f (toIValue v)
--  cast (IVCapsule v) f = toIValue v >>= f
  uncast obj f = do
    iValue_isTensor obj >>= \case
      1 -> fromIValue obj >>= f.IVTensor . Unsafe
      _ -> 
        iValue_isDouble obj >>= \case
          1 -> fromIValue obj >>= f.IVDouble
          _ -> undefined
