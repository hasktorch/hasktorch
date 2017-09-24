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
type CTHGenerator = () -- TODO - should this be defined in terms of the pointer?
type CTHFile = ()
type CTHStorage = ()
type CTHHalf = CUShort

-- ----------------------------------------
-- Templated types
-- ----------------------------------------

{- Byte -}

type CTHByteTensor = ()
type CTHByteStorage = ()

{- Char -}

type CTHCharTensor = ()
type CTHCharStorage = ()
type CTHCharLongStorage = ()

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

type CTHLongTensor = ()
type CTHLongStorage = ()

{- Short -}

type CTHShortTensor = ()
type CTHShortStorage = ()
