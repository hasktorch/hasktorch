module THTypes where

import Foreign
import Foreign.C.Types

type CTHDescBuff = Ptr ()

-- ----------------------------------------
-- Templated types
-- ----------------------------------------

{- Byte -}

type CTHByteTensor = ()      -- THTensor / THTensor.h
type CTHByteStorage = ()     -- THStorage / THStorag
type CTHByteLongStorage = () -- THLongStorage / THStorage.h
-- TODO : determine appropriate type for these:
type CTHBytePtrDiff = CInt     -- ptrdiff_t / THStorage.h



{- Float -}

type CTHFloatTensor = ()      -- THTensor / THTensor.h
type CTHFloatStorage = ()     -- THStorage / THStorag
type CTHFloatLongStorage = () -- THLongStorage / THStorage.h
-- TODO : determine appropriate type for these:
type CTHFloatPtrDiff = CInt     -- ptrdiff_t / THStorage.h

{- Double -}

type CTHDoubleTensor = ()      -- THTensor / THTensor.h
type CTHDoubleStorage = ()     -- THStorage / THStorag
type CTHDoubleLongStorage = () -- THLongStorage / THStorage.h
-- TODO : determine appropriate type for these:
type CTHDoublePtrDiff = CInt     -- ptrdiff_t / THStorage.h

type CTHDoubleTensor = ()      -- THTensor / THTensor.h
type CTHDoubleStorage = ()     -- THStorage / THStorag
type CTHDoubleLongStorage = () -- THLongStorage / THStorage.h
-- TODO : determine appropriate type for these:
type CTHDoublePtrDiff = CInt     -- ptrdiff_t / THStorage.h



