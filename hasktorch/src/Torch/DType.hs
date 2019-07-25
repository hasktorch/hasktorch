{-# LANGUAGE TypeSynonymInstances #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE AllowAmbiguousTypes #-}

module Torch.DType where

import ATen.Class (Castable(..))
import qualified ATen.Const as ATen
import qualified ATen.Type as ATen
import Data.Int
import Data.Word

data DType = UInt8 | Int8 | Int16 | Int32 | Int64 | Half | Float | Double
  deriving (Eq, Show)

class DataType a where
  dataType :: DataType a => DType

instance DataType Word8 where
  dataType = UInt8

instance DataType Int8 where
  dataType = Int8

instance DataType Int16 where
  dataType = Int16

instance DataType Int32 where
  dataType = Int32

instance DataType Int where
  dataType = Int64

instance DataType Int64 where
  dataType = Int64

instance DataType Float where
  dataType = Float

instance DataType Double where
  dataType = Double

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


isIntegral :: DType -> Bool
isIntegral UInt8  = True
isIntegral Int8   = True
isIntegral Int16  = True
isIntegral Int32  = True
isIntegral Int64  = True
isIntegral Half   = False
isIntegral Float  = False
isIntegral Double = False
