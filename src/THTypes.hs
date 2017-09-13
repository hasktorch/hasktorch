module THTypes where

import Foreign
import Foreign.C.Types

{- preprocess-generated float types -}

type CTHFloatTensor = ()      -- THTensor / THTensor.h
type CTHFloatStorage = ()     -- THStorage / THStorag
type CTHFloatLongStorage = () -- THLongStorage / THStorage.h

-- TODO : determine appropriate type for these:
type CTHFloatPtrDiff = CInt     -- ptrdiff_t / THStorage.h
type CTHDescBuff = Ptr ()

