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

-- in THNN.h
-- typedef void THNNState
type CTHDoubleNNState = Ptr ()
type CTHFloatNNState = Ptr ()

-- typedef int64_t THIndex_t;
-- typedef int32_t THInteger_t;
type CTHIndexTensor = CLong
type CTHIntegerTensor = CInt

-- bool mapping
type CBool = CInt

-- ----------------------------------------
-- Templated types
-- ----------------------------------------

{- Byte -}

type CTHByteTensor = C'THByteTensor
type CTHByteStorage = C'THByteStorage


{- Char -}

type CTHCharTensor = C'THCharTensor
type CTHCharStorage = C'THCharStorage

{- Double -}

type CTHDoubleTensor = C'THDoubleTensor
type CTHDoubleStorage = C'THDoubleStorage

{- Float -}

type CTHFloatTensor = C'THFloatTensor
type CTHFloatStorage = C'THFloatStorage

{- Half -}

type CTHHalfTensor = ()
type CTHHalfStorage = ()

{- Int -}

type CTHIntTensor = C'THIntTensor
type CTHIntStorage = C'THIntStorage

{- Long -}

type CTHLongTensor = C'THLongTensor
type CTHLongStorage = C'THLongStorage

{- Short -}

type CTHShortTensor = C'THShortTensor
type CTHShortStorage = C'THShortStorage
