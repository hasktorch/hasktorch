{-# LANGUAGE ForeignFunctionInterface #-}
module THIntTensorLapack
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

type CTensor = THTypes.CTHIntTensor
type CReal = Foreign.C.Types.CInt
type CAccReal = Foreign.C.Types.CLong
type CStorage = THTypes.CTHIntStorage