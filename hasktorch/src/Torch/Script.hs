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
import qualified Torch.Internal.Managed.Type.Module as LibTorch

import Torch.Device
import Torch.DType
import Torch.TensorOptions

newtype Module = UnsafeModule (ForeignPtr ATen.Module)
newtype IValue = UnsafeIValue (ForeignPtr ATen.IValue)

instance Castable Module (ForeignPtr ATen.Module) where
  cast (UnsafeModule obj) f = f obj
  uncast obj f = f $ UnsafeModule obj

instance Castable IValue (ForeignPtr ATen.IValue) where
  cast (UnsafeIValue obj) f = f obj
  uncast obj f = f $ UnsafeIValue obj

instance Castable [IValue] (ForeignPtr ATen.IValueList) where
  cast xs f = do
    ptr_list <- mapM (\x -> (cast x return :: IO (ForeignPtr ATen.IValue))) xs
    cast ptr_list f
  uncast xs f = uncast xs $ \ptr_list -> do
    tensor_list <- mapM (\(x :: ForeignPtr ATen.IValue) -> uncast x return) ptr_list
    f tensor_list

instance (IValueLike a (ForeignPtr ATen.IValue))
  => IValueLike a IValue where
  toIValue x = cast1 (toIValue :: a -> IO (ForeignPtr ATen.IValue)) x
  fromIValue x = cast1 (fromIValue :: ForeignPtr ATen.IValue -> IO a) x

instance (CppObject a, IValueLike (ForeignPtr a) (ForeignPtr ATen.IValue))
  =>  IValueLike (ForeignPtr a) (ForeignPtr ATen.IValue) where
  toIValue x = cast1 (toIValue :: ForeignPtr a -> IO (ForeignPtr ATen.IValue)) x
  fromIValue x = cast1 (fromIValue :: ForeignPtr ATen.IValue -> IO (ForeignPtr a)) x

save :: Module -> FilePath -> IO ()
save = cast2 LibTorch.save

load :: FilePath -> IO Module
load = cast1 LibTorch.load

forward :: Module -> [IValue] -> IO IValue
forward = cast2 LibTorch.forward
