{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}

module Torch.DType where

import Torch.Internal.Class (Castable(..))
import qualified Torch.Internal.Const as ATen
import qualified Torch.Internal.Type as ATen
import Data.Int
import Data.Word
import Data.Reflection

data DType =
    Bool   -- ^ Bool
  | UInt8  -- ^ Byte
  | Int8   -- ^ Char
  | Int16  -- ^ Short
  | Int32  -- ^ Int
  | Int64  -- ^ Long
  | Half   -- ^ Half
  | Float  -- ^ Float
  | Double -- ^ Double
  deriving (Eq, Show)

instance Reifies Bool DType where
  reflect _ = Bool

instance Reifies 'Bool DType where
  reflect _ = Bool

instance Reifies Word8 DType where
  reflect _ = UInt8

instance Reifies Int8 DType where
  reflect _ = Int8

instance Reifies 'Int8 DType where
  reflect _ = Int8

instance Reifies Int16 DType where
  reflect _ = Int16

instance Reifies 'Int16 DType where
  reflect _ = Int16

instance Reifies Int32 DType where
  reflect _ = Int32

instance Reifies 'Int32 DType where
  reflect _ = Int32

instance Reifies Int DType where
  reflect _ = Int64

instance Reifies Int64 DType where
  reflect _ = Int64

instance Reifies 'Int64 DType where
  reflect _ = Int64

instance Reifies Float DType where
  reflect _ = Float

instance Reifies 'Float DType where
  reflect _ = Float

instance Reifies Double DType where
  reflect _ = Double

instance Reifies 'Double DType where
  reflect _ = Float

instance Castable DType ATen.ScalarType where
  cast Bool   f = f ATen.kBool
  cast UInt8  f = f ATen.kByte
  cast Int8   f = f ATen.kChar
  cast Int16  f = f ATen.kShort
  cast Int32  f = f ATen.kInt
  cast Int64  f = f ATen.kLong
  cast Half   f = f ATen.kHalf
  cast Float  f = f ATen.kFloat
  cast Double f = f ATen.kDouble

  uncast x f
    | x == ATen.kBool = f Bool
    | x == ATen.kByte = f UInt8
    | x == ATen.kChar = f Int8
    | x == ATen.kShort = f Int16
    | x == ATen.kInt = f Int32
    | x == ATen.kLong = f Int64
    | x == ATen.kHalf = f Half
    | x == ATen.kFloat = f Float
    | x == ATen.kDouble = f Double


isIntegral :: DType -> Bool
isIntegral Bool   = True
isIntegral UInt8  = True
isIntegral Int8   = True
isIntegral Int16  = True
isIntegral Int32  = True
isIntegral Int64  = True
isIntegral Half   = False
isIntegral Float  = False
isIntegral Double = False
