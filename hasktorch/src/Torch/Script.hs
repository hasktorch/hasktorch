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
import Torch.Internal.Unmanaged.Type.C10Dict
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

instance Show Blob where
  show _ = "Blob"

instance Show Future where
  show _ = "Future"

instance Show Object where
  show _ = "Object"

instance Show Capsule where
  show _ = "Capsule"

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
  | IVBlob -- Blob
  | IVGenericList [IValue]
  | IVGenericDict [(IValue,IValue)]
  | IVFuture -- Future
  | IVDevice -- Device
  | IVObject -- Object
  | IVUninitialized
  | IVCapsule -- Capsule
  deriving (Show)

instance Castable Module (ForeignPtr ATen.Module) where
  cast (UnsafeModule obj) f = f obj
  uncast obj f = f $ UnsafeModule obj

save :: Module -> FilePath -> IO ()
save = cast2 LibTorch.save

load :: FilePath -> IO Module
load = cast1 LibTorch.load

forward' :: Module -> [RawIValue] -> IO RawIValue
forward' = cast2 LibTorch.forward

forward :: Module -> [IValue] -> IO IValue
forward a b = cast2 forward' a b

instance Castable [IValue] [RawIValue] where
  cast a f = (forM a $ \v -> cast v return) >>= f
  uncast a f = (forM a $ \v -> uncast v return) >>= f

instance Castable IValue RawIValue where
  cast (IVNone) f = newIValue >>= f
  cast (IVTensor (Unsafe v)) f = toIValue v>>= f
  cast (IVDouble v) f = toIValue v >>= f
  cast (IVInt v) f = toIValue v >>= f
  cast (IVBool v) f = toIValue v >>= f
  cast (IVTuple v) f = do
    rawIValues <- cast v return :: IO [RawIValue]
    c10tuple <- cast rawIValues return :: IO (ForeignPtr (ATen.C10Ptr ATen.IVTuple))
    f =<< toIValue c10tuple
  cast (IVIntList v) f = do
    v' <- cast v return :: IO (ForeignPtr (ATen.C10List Int64))
    f =<< toIValue v'
  cast (IVDoubleList v) f = do
    cdoubles <- forM v (flip cast return) :: IO [CDouble]
    c10list <- cast cdoubles return :: IO (ForeignPtr (ATen.C10List CDouble))
    f =<< toIValue c10list
  cast (IVBoolList v) f = do
    cbools <- forM v (flip cast return) :: IO [CBool]
    c10list <- cast cbools return :: IO (ForeignPtr (ATen.C10List CBool))
    f =<< toIValue c10list
  cast (IVString v) f = do
    v' <- cast v return :: IO (ForeignPtr (ATen.StdString))
    f =<< toIValue v'
  cast (IVTensorList v) f = do
    v' <- cast v return :: IO (ForeignPtr (ATen.C10List ATen.Tensor))
    f =<< toIValue v'
  cast a f = throwIO $ userError $ "Unsupported data-type:" ++ show a
--  cast (IVBlob (UnsafeBlob v)) f = toIValue v >>= f
--  cast (IVGenericList v) f = toIValue v >>= f
--  cast (IVGenericDict v) f = toIValue v >>= f
--  cast (IVFuture (UnsafeFuture v)) f = toIValue v >>= f
--  cast (IVDevice v) f = toIValue v >>= f
--  cast (IVObject (UnsafeObject v)) f = toIValue v >>= f
--  cast (IVUninitialized) f = f (toIValue v)
--  cast (IVCapsule v) f = toIValue v >>= f
  uncast obj f =
    select
      [ (iValue_isNone obj, f IVNone)
      , (iValue_isTensor obj, fromIValue obj >>= f . IVTensor . Unsafe)
      , (iValue_isDouble obj, fromIValue obj >>= f . IVDouble)
      , (iValue_isInt obj, fromIValue obj >>= f . IVInt)
      , (iValue_isBool obj, fromIValue obj >>= f . IVBool)
      , (iValue_isString obj, do
           v <- fromIValue obj :: IO (ForeignPtr ATen.StdString)
           str <- uncast v return :: IO String
           f (IVString str)
        )
      , (iValue_isTensorList obj, do
           v' <- fromIValue obj :: IO (ForeignPtr (ATen.C10List ATen.Tensor))
           ts <- uncast v' return :: IO [Tensor]
           f (IVTensorList ts)
        )
      , (iValue_isDoubleList obj, do
           v' <- fromIValue obj :: IO (ForeignPtr (ATen.C10List CDouble))
           cdoubles <- uncast v' return :: IO [CDouble]
           doubles <- forM cdoubles (flip uncast return) :: IO [Double]
           f (IVDoubleList doubles)
        )
      , (iValue_isIntList obj, do
           v' <- fromIValue obj :: IO (ForeignPtr (ATen.C10List Int64))
           ts <- uncast v' return :: IO [Int64]
           f (IVIntList ts)
        )
      , (iValue_isBoolList obj, do
           v' <- fromIValue obj :: IO (ForeignPtr (ATen.C10List CBool))
           cbools <- uncast v' return :: IO [CBool]
           bools <- forM cbools (flip uncast return) :: IO [Bool]
           f (IVBoolList bools)
        )
      , (iValue_isTuple obj, do
           c10tuple <- fromIValue obj :: IO (ForeignPtr (ATen.C10Ptr ATen.IVTuple))
           rawIValues <- uncast c10tuple return :: IO [RawIValue]
           ts <- uncast rawIValues return :: IO [IValue]
           f (IVTuple ts)
        )
      , (iValue_isBlob obj, f IVBlob)
      , (iValue_isGenericList obj, do
           c10list <- fromIValue obj :: IO (ForeignPtr (ATen.C10List ATen.IValue))
           rawIValues <- uncast c10list return :: IO [RawIValue]
           ts <- uncast rawIValues return :: IO [IValue]
           f (IVGenericList ts)
        )
      , (iValue_isGenericDict obj, do
           c10list <- fromIValue obj :: IO (ForeignPtr (ATen.C10Dict '(ATen.IValue,ATen.IValue)))
           rawIValues <- uncast c10list return :: IO [(RawIValue,RawIValue)]
           ts <- forM rawIValues $ \(a,b) -> do
             a' <- uncast a return
             b' <- uncast b return
             return (a',b')
           f (IVGenericDict ts)
        )
      , (iValue_isFuture obj, f IVFuture)
      , (iValue_isDevice obj, f IVDevice)
      , (iValue_isObject obj, f IVObject)
      , (iValue_isCapsule obj, f IVCapsule)
      ]
    where
      select [] = throwIO $ userError "Unsupported IValue"
      select ((cond,body):xs) =
        cond >>= \case
          1 -> do
            body
          _ -> do
            select xs
