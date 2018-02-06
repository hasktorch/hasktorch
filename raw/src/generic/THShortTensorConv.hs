{-# LANGUAGE ForeignFunctionInterface #-}
module THShortTensorConv
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

type CTensor = THTypes.CTHShortTensor
type CReal = Foreign.C.Types.CShort
type CAccReal = Foreign.C.Types.CLong
type CStorage = THTypes.CTHShortStorage