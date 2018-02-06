{-# LANGUAGE ForeignFunctionInterface #-}
module THFloatStorageCopy
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

type CTensor = THTypes.CTHFloatTensor
type CReal = Foreign.C.Types.CFloat
type CAccReal = Foreign.C.Types.CDouble
type CStorage = THTypes.CTHFloatStorage