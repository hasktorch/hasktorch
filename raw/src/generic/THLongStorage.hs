{-# LANGUAGE ForeignFunctionInterface #-}
module THLongStorage
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

type CTensor = THTypes.CTHLongTensor
type CReal = Foreign.C.Types.CLong
type CAccReal = Foreign.C.Types.CLong
type CStorage = THTypes.CTHLongStorage