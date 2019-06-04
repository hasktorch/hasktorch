{-# LANGUAGE TypeSynonymInstances #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}

module Torch.DType where

import ATen.Class (Castable(..))
import qualified ATen.Const as ATen
import qualified ATen.Type as ATen

data DType = UInt8 | Int8 | Int16 | Int32 | Int64 | Half | Float | Double
  deriving (Eq, Show)

instance Castable DType ATen.ScalarType where
  cast UInt8  f = f ATen.kByte
  cast Int8   f = f ATen.kChar
  cast Int16  f = f ATen.kShort
  cast Int32  f = f ATen.kInt
  cast Int64  f = f ATen.kLong
  cast Half   f = f ATen.kHalf
  cast Float  f = f ATen.kFloat
  cast Double f = f ATen.kDouble

  uncast x f
    | x == ATen.kByte = f UInt8
    | x == ATen.kChar = f Int8
    | x == ATen.kShort = f Int16
    | x == ATen.kInt = f Int32
    | x == ATen.kLong = f Int64
    | x == ATen.kHalf = f Half
    | x == ATen.kFloat = f Float
    | x == ATen.kDouble = f Double
