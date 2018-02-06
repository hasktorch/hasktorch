{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE TemplateHaskell #-}

module THTypes where

import Foreign
import Foreign.C.String
import Foreign.C.Types
import Foreign.Ptr
import Foreign.Storable

import TorchStructs

type CTHDescBuff = Ptr ()
type CTHAllocatorPtr = Ptr C'THAllocator
type CTHGenerator = C'THGenerator
type CTHFile = ()
type CTHHalf = CUShort

type THLongBlas = CLong
type THShortBlas = CShort
type THIntBlas = CInt
type THByteBlas = CChar
type THHalfBlas = CShort
type THFloatBlas = CFloat
type THDoubleBlas = CDouble

type THLongLapack = CLong
type THShortLapack = CShort
type THIntLapack = CInt
type THByteLapack = CChar
type THHalfLapack = CShort
type THFloatLapack = CFloat
type THDoubleLapack = CDouble

type CTHLongVector = CLong
type CTHShortVector = CShort
type CTHIntVector = CInt
type CTHByteVector = CChar
type CTHHalfVector = CShort
type CTHFloatVector = CFloat
type CTHDoubleVector = CDouble

-- in THNN.h
-- typedef void THNNState
type CTHDoubleNNState = ()
type CTHFloatNNState = ()

-- typedef int64_t THIndex_t;
-- typedef int32_t THInteger_t;
type CTHIndexTensor = CLong
type CTHIntegerTensor = CInt

-- ----------------------------------------
-- Templated types
-- ----------------------------------------

{- Byte -}

type CTHByteTensor = C'THByteTensor
type CTHByteStorage = C'THByteStorage
type CTHByteStorageCopy = CTHByteStorage


{- Char -}

type CTHCharTensor = C'THCharTensor
type CTHCharStorage = C'THCharStorage
type CTHCharStorageCopy = CTHCharStorage

{- Double -}

type CTHDoubleTensor = C'THDoubleTensor
type CTHDoubleStorage = C'THDoubleStorage
type CTHDoubleStorageCopy = CTHDoubleStorage

{- Float -}

type CTHFloatTensor = C'THFloatTensor
type CTHFloatStorage = C'THFloatStorage
type CTHFloatStorageCopy = CTHFloatStorage

{- Half -}

type CTHHalfTensor = ()
type CTHHalfStorage = ()
type CTHHalfStorageCopy = ()

{- Int -}

type CTHIntTensor = C'THIntTensor
type CTHIntStorage = C'THIntStorage
type CTHIntStorageCopy = CTHIntStorage

{- Long -}

type CTHLongTensor = C'THLongTensor
type CTHLongStorage = C'THLongStorage
type CTHLongStorageCopy = CTHLongStorage

{- Short -}

type CTHShortTensor = C'THShortTensor
type CTHShortStorage = C'THShortStorage
type CTHShortStorageCopy = CTHShortStorage
