{-# LANGUAGE ForeignFunctionInterface #-}
module THHalfBlas
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

type CTensor = THTypes.CTHHalfTensor
type CReal = THTypes.CTHHalf
type CAccReal = Foreign.C.Types.CFloat
type CStorage = THTypes.CTHHalfStorage