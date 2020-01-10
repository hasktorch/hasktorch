
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}

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



newIValue_V
  :: ForeignPtr IValue
  -> IO (ForeignPtr IValue)
newIValue_V = cast1 Unmanaged.newIValue_V

newIValue_t
  :: ForeignPtr Tensor
  -> IO (ForeignPtr IValue)
newIValue_t = cast1 Unmanaged.newIValue_t

newIValue_l
  :: ForeignPtr TensorList
  -> IO (ForeignPtr IValue)
newIValue_l = cast1 Unmanaged.newIValue_l

newIValue_s
  :: ForeignPtr Scalar
  -> IO (ForeignPtr IValue)
newIValue_s = cast1 Unmanaged.newIValue_s

newIValue_d
  :: CDouble
  -> IO (ForeignPtr IValue)
newIValue_d = cast1 Unmanaged.newIValue_d

newIValue_l
  :: Int64
  -> IO (ForeignPtr IValue)
newIValue_l = cast1 Unmanaged.newIValue_l

newIValue_i
  :: Int32
  -> IO (ForeignPtr IValue)
newIValue_i = cast1 Unmanaged.newIValue_i

newIValue_b
  :: CBool
  -> IO (ForeignPtr IValue)
newIValue_b = cast1 Unmanaged.newIValue_b





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

iValue_toIValue
  :: ForeignPtr IValue
  -> IO (ForeignPtr IValue)
iValue_toIValue = cast1 Unmanaged.iValue_toIValue

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

iValue_toDouble
  :: ForeignPtr IValue
  -> IO (CDouble)
iValue_toDouble = cast1 Unmanaged.iValue_toDouble

iValue_isFuture
  :: ForeignPtr IValue
  -> IO (CBool)
iValue_isFuture = cast1 Unmanaged.iValue_isFuture

iValue_isInt
  :: ForeignPtr IValue
  -> IO (CBool)
iValue_isInt = cast1 Unmanaged.iValue_isInt

iValue_toInt
  :: ForeignPtr IValue
  -> IO (Int64)
iValue_toInt = cast1 Unmanaged.iValue_toInt

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

iValue_isBoolList
  :: ForeignPtr IValue
  -> IO (CBool)
iValue_isBoolList = cast1 Unmanaged.iValue_isBoolList

iValue_isTensorList
  :: ForeignPtr IValue
  -> IO (CBool)
iValue_isTensorList = cast1 Unmanaged.iValue_isTensorList

iValue_isGenericList
  :: ForeignPtr IValue
  -> IO (CBool)
iValue_isGenericList = cast1 Unmanaged.iValue_isGenericList

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

iValue_toScalar
  :: ForeignPtr IValue
  -> IO (ForeignPtr Scalar)
iValue_toScalar = cast1 Unmanaged.iValue_toScalar

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

iValue_internalToPointer
  :: ForeignPtr IValue
  -> IO (Ptr ())
iValue_internalToPointer = cast1 Unmanaged.iValue_internalToPointer

iValue_clearToNone
  :: ForeignPtr IValue
  -> IO (())
iValue_clearToNone = cast1 Unmanaged.iValue_clearToNone



