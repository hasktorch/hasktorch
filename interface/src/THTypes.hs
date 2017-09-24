module THTypes where

import Foreign
import Foreign.C.Types
import Foreign.Storable

type CTHDescBuff = Ptr ()
type CTHAllocatorPtr = Ptr ()
type CTHGenerator = () -- TODO - should this be defined in terms of the pointer?
type CTHFile = ()
type CTHStorage = ()
type CTHHalf = CUShort

-- ----------------------------------------
-- Templated types
-- ----------------------------------------

-- showStruct :: MyStruct -> IO ()
-- showStruct ss = peek ss >>= print

-- data CTHIntStorage = CTHIntStorage CInt CChar
--   deriving (Show, Read, Eq)
-- type MyStruct = Ptr MyStructType

-- instance Storable MyStructType where
--   sizeOf _ = 8
--   alignment = sizeOf
--   peek ptr = do
--     a <- peekByteOff ptr 0
--     b <- peekByteOff ptr 4
--     return (MyStructType a b)

{- Byte -}

type CTHByteTensor = ()      -- THTensor / THTensor.h
type CTHByteStorage = ()     -- THStorage / THStorag
type CTHByteLongStorage = () -- THLongStorage / THStorage.h
type CTHBytePtrDiff = CInt     -- ptrdiff_t / THStorage.h

{- Char -}

type CTHCharTensor = ()      -- THTensor / THTensor.h
type CTHCharStorage = ()     -- THStorage / THStorag
type CTHCharLongStorage = () -- THLongStorage / THStorage.h
type CTHCharPtrDiff = CInt     -- ptrdiff_t / THStorage.h

{- Double -}

type CTHDoubleTensor = ()      -- THTensor / THTensor.h
type CTHDoubleStorage = ()     -- THStorage / THStorag
type CTHDoubleLongStorage = () -- THLongStorage / THStorage.h
type CTHDoublePtrDiff = CInt     -- ptrdiff_t / THStorage.h

{- Float -}

type CTHFloatTensor = ()      -- THTensor / THTensor.h
type CTHFloatStorage = ()     -- THStorage / THStorag
type CTHFloatLongStorage = () -- THLongStorage / THStorage.h
type CTHFloatPtrDiff = CInt     -- ptrdiff_t / THStorage.h

{- Half -}

type CTHHalfTensor = ()      -- THTensor / THTensor.h
type CTHHalfStorage = ()     -- THStorage / THStorag
type CTHHalfLongStorage = () -- THLongStorage / THStorage.h
type CTHHalfPtrDiff = CInt     -- ptrdiff_t / THStorage.h

{- Int -}

type CTHIntTensor = ()      -- THTensor / THTensor.h
type CTHIntStorage = ()     -- THStorage / THStorag
type CTHIntLongStorage = () -- THLongStorage / THStorage.h
type CTHIntPtrDiff = CInt     -- ptrdiff_t / THStorage.h

{- Long -}

type CTHLongTensor = ()      -- THTensor / THTensor.h
type CTHLongStorage = ()     -- THStorage / THStorag
type CTHLongLongStorage = () -- THLongStorage / THStorage.h
type CTHLongPtrDiff = CInt     -- ptrdiff_t / THStorage.h

{- Short -}

type CTHShortTensor = ()      -- THTensor / THTensor.h
type CTHShortStorage = ()     -- THStorage / THStorag
type CTHShortLongStorage = () -- THLongStorage / THStorage.h
type CTHShortPtrDiff = CInt     -- ptrdiff_t / THStorage.h
