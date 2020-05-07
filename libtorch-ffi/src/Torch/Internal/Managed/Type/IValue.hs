
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE UndecidableInstances #-}



module Torch.Internal.Managed.Type.IValue where


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

import qualified Torch.Internal.Unmanaged.Type.IValue as Unmanaged
import Torch.Internal.Unmanaged.Type.IValue (IValueLike)

instance IValueLike Double (ForeignPtr IValue) where
  toIValue x = cast1 (Unmanaged.toIValue :: CDouble -> IO (Ptr IValue)) x
  fromIValue x = cast1 (Unmanaged.fromIValue :: Ptr IValue -> IO CDouble) x

instance IValueLike Int64 (ForeignPtr IValue) where
  toIValue x = cast1 (Unmanaged.toIValue :: Int64 -> IO (Ptr IValue)) x
  fromIValue x = cast1 (Unmanaged.fromIValue :: Ptr IValue -> IO Int64) x

instance IValueLike Bool (ForeignPtr IValue) where
  toIValue x = cast1 (Unmanaged.toIValue :: CBool -> IO (Ptr IValue)) x
  fromIValue x = cast1 (Unmanaged.fromIValue :: Ptr IValue -> IO CBool) x

--instance IValueLike (ForeignPtr StdString) (ForeignPtr IValue) where
--  toIValue x = cast1 (Unmanaged.toIValue :: Ptr StdString -> IO (Ptr IValue)) x
--  fromIValue x = cast1 (Unmanaged.fromIValue :: Ptr IValue -> IO (Ptr StdString)) x
--instance IValueLike String (ForeignPtr IValue) where
--  toIValue x = cast1 (Unmanaged.toIValue :: CBool -> IO (Ptr IValue)) x
--  fromIValue x = cast1 (Unmanaged.fromIValue :: Ptr IValue -> IO CBool) x

instance (CppObject a, IValueLike (Ptr a) (Ptr IValue)) => IValueLike (ForeignPtr a) (ForeignPtr IValue) where
  toIValue x = cast1 (Unmanaged.toIValue :: Ptr a -> IO (Ptr IValue)) x
  fromIValue x = cast1 (Unmanaged.fromIValue :: Ptr IValue -> IO (Ptr a)) x

newIValue
  :: IO (ForeignPtr IValue)
newIValue  = cast0 Unmanaged.newIValue

iValue_isAliasOf_V
  :: ForeignPtr IValue
  -> ForeignPtr IValue
  -> IO (CBool)
iValue_isAliasOf_V = cast2 Unmanaged.iValue_isAliasOf_V

iValue_use_count
  :: ForeignPtr IValue
  -> IO (CSize)
iValue_use_count = cast1 Unmanaged.iValue_use_count

iValue_swap_V
  :: ForeignPtr IValue
  -> ForeignPtr IValue
  -> IO (())
iValue_swap_V = cast2 Unmanaged.iValue_swap_V

iValue_isTensor
  :: ForeignPtr IValue
  -> IO (CBool)
iValue_isTensor = cast1 Unmanaged.iValue_isTensor

iValue_isBlob
  :: ForeignPtr IValue
  -> IO (CBool)
iValue_isBlob = cast1 Unmanaged.iValue_isBlob

iValue_isCapsule
  :: ForeignPtr IValue
  -> IO (CBool)
iValue_isCapsule = cast1 Unmanaged.iValue_isCapsule

iValue_isTuple
  :: ForeignPtr IValue
  -> IO (CBool)
iValue_isTuple = cast1 Unmanaged.iValue_isTuple

iValue_isDouble
  :: ForeignPtr IValue
  -> IO (CBool)
iValue_isDouble = cast1 Unmanaged.iValue_isDouble

iValue_isFuture
  :: ForeignPtr IValue
  -> IO (CBool)
iValue_isFuture = cast1 Unmanaged.iValue_isFuture

iValue_isInt
  :: ForeignPtr IValue
  -> IO (CBool)
iValue_isInt = cast1 Unmanaged.iValue_isInt

iValue_isIntList
  :: ForeignPtr IValue
  -> IO (CBool)
iValue_isIntList = cast1 Unmanaged.iValue_isIntList

iValue_isString
  :: ForeignPtr IValue
  -> IO (CBool)
iValue_isString = cast1 Unmanaged.iValue_isString

iValue_toStringRef
  :: ForeignPtr IValue
  -> IO (ForeignPtr StdString)
iValue_toStringRef = cast1 Unmanaged.iValue_toStringRef

iValue_isDoubleList
  :: ForeignPtr IValue
  -> IO (CBool)
iValue_isDoubleList = cast1 Unmanaged.iValue_isDoubleList

iValue_isBool
  :: ForeignPtr IValue
  -> IO (CBool)
iValue_isBool = cast1 Unmanaged.iValue_isBool

iValue_isObject
  :: ForeignPtr IValue
  -> IO (CBool)
iValue_isObject = cast1 Unmanaged.iValue_isObject

iValue_isBoolList
  :: ForeignPtr IValue
  -> IO (CBool)
iValue_isBoolList = cast1 Unmanaged.iValue_isBoolList

iValue_isTensorList
  :: ForeignPtr IValue
  -> IO (CBool)
iValue_isTensorList = cast1 Unmanaged.iValue_isTensorList

iValue_isList
  :: ForeignPtr IValue
  -> IO (CBool)
iValue_isList = cast1 Unmanaged.iValue_isList

iValue_isGenericDict
  :: ForeignPtr IValue
  -> IO (CBool)
iValue_isGenericDict = cast1 Unmanaged.iValue_isGenericDict

iValue_isNone
  :: ForeignPtr IValue
  -> IO (CBool)
iValue_isNone = cast1 Unmanaged.iValue_isNone

iValue_toNone
  :: ForeignPtr IValue
  -> IO (ForeignPtr StdString)
iValue_toNone = cast1 Unmanaged.iValue_toNone

iValue_isScalar
  :: ForeignPtr IValue
  -> IO (CBool)
iValue_isScalar = cast1 Unmanaged.iValue_isScalar

iValue_isDevice
  :: ForeignPtr IValue
  -> IO (CBool)
iValue_isDevice = cast1 Unmanaged.iValue_isDevice

iValue_toScalarType
  :: ForeignPtr IValue
  -> IO (ScalarType)
iValue_toScalarType = cast1 Unmanaged.iValue_toScalarType

iValue_toLayout
  :: ForeignPtr IValue
  -> IO (Layout)
iValue_toLayout = cast1 Unmanaged.iValue_toLayout

iValue_toMemoryFormat
  :: ForeignPtr IValue
  -> IO (MemoryFormat)
iValue_toMemoryFormat = cast1 Unmanaged.iValue_toMemoryFormat

iValue_toQScheme
  :: ForeignPtr IValue
  -> IO (QScheme)
iValue_toQScheme = cast1 Unmanaged.iValue_toQScheme

iValue_tagKind
  :: ForeignPtr IValue
  -> IO (ForeignPtr StdString)
iValue_tagKind = cast1 Unmanaged.iValue_tagKind

iValue_isSameIdentity_V
  :: ForeignPtr IValue
  -> ForeignPtr IValue
  -> IO (CBool)
iValue_isSameIdentity_V = cast2 Unmanaged.iValue_isSameIdentity_V

iValue_isPtrType
  :: ForeignPtr IValue
  -> IO (CBool)
iValue_isPtrType = cast1 Unmanaged.iValue_isPtrType

