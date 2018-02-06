{-# LANGUAGE ForeignFunctionInterface #-}
module THDoubleTensorMath
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

type CTensor = THTypes.CTHDoubleTensor
type CReal = Foreign.C.Types.CDouble
type CAccReal = Foreign.C.Types.CDouble
type CStorage = THTypes.CTHDoubleStorage