{-# LANGUAGE ForeignFunctionInterface #-}
module THByteTensor
  ( CTensor
  , CReal
  , CAccReal
  , CStorage
  ) where

import Foreign
import Foreign.C.Types
import THTypes
import Data.Word
import Data.Int

type CTensor = THTypes.CTHCharTensor
type CReal = Foreign.C.Types.CChar
type CAccReal = Foreign.C.Types.CLong
type CStorage = THTypes.CTHCharStorage