{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}

module Torch.DType where

import Data.Complex
import qualified Numeric.Half as N
import Data.Int
import Data.Reflection
import Data.Word
import Torch.Internal.Class (Castable (..))
import qualified Torch.Internal.Const as ATen
import qualified Torch.Internal.Type as ATen

data DType
  = -- | Bool
    Bool
  | -- | Byte
    UInt8
  | -- | Char
    Int8
  | -- | Short
    Int16
  | -- | Int
    Int32
  | -- | Long
    Int64
  | -- | Half
    Half
  | -- | Float
    Float
  | -- | Double
    Double
  | -- | ComplexHalf
    ComplexHalf
  | -- | ComplexFloat
    ComplexFloat
  | -- | ComplexDouble
    ComplexDouble
  | -- | QInt8
    QInt8
  | -- | QUInt8
    QUInt8
  | -- | QInt32
    QInt32
  | -- | BFloat16
    BFloat16
  deriving (Eq, Show, Read)

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

instance Reifies N.Half DType where
  reflect _ = Half

instance Reifies 'Half DType where
  reflect _ = Half

instance Reifies Float DType where
  reflect _ = Float

instance Reifies 'Float DType where
  reflect _ = Float

instance Reifies Double DType where
  reflect _ = Double

instance Reifies 'Double DType where
  reflect _ = Double

instance Reifies (Complex N.Half) DType where
  reflect _ = ComplexHalf

instance Reifies 'ComplexHalf DType where
  reflect _ = ComplexHalf

instance Reifies (Complex Float) DType where
  reflect _ = ComplexFloat

instance Reifies 'ComplexFloat DType where
  reflect _ = ComplexFloat

instance Reifies (Complex Double) DType where
  reflect _ = ComplexDouble

instance Reifies 'ComplexDouble DType where
  reflect _ = ComplexDouble

instance Castable DType ATen.ScalarType where
  cast Bool f = f ATen.kBool
  cast UInt8 f = f ATen.kByte
  cast Int8 f = f ATen.kChar
  cast Int16 f = f ATen.kShort
  cast Int32 f = f ATen.kInt
  cast Int64 f = f ATen.kLong
  cast Half f = f ATen.kHalf
  cast Float f = f ATen.kFloat
  cast Double f = f ATen.kDouble
  cast ComplexHalf f = f ATen.kComplexHalf
  cast ComplexFloat f = f ATen.kComplexFloat
  cast ComplexDouble f = f ATen.kComplexDouble
  cast QInt8 f = f ATen.kQInt8
  cast QUInt8 f = f ATen.kQUInt8
  cast QInt32 f = f ATen.kQInt32
  cast BFloat16 f = f ATen.kBFloat16

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
    | x == ATen.kComplexHalf = f ComplexHalf
    | x == ATen.kComplexFloat = f ComplexFloat
    | x == ATen.kComplexDouble = f ComplexDouble
    | x == ATen.kQInt8 = f QInt8
    | x == ATen.kQUInt8 = f QUInt8
    | x == ATen.kQInt32 = f QInt32
    | x == ATen.kBFloat16 = f BFloat16

isIntegral :: DType -> Bool
isIntegral Bool = True
isIntegral UInt8 = True
isIntegral Int8 = True
isIntegral Int16 = True
isIntegral Int32 = True
isIntegral Int64 = True
isIntegral Half = False
isIntegral Float = False
isIntegral Double = False
isIntegral ComplexHalf = False
isIntegral ComplexFloat = False
isIntegral ComplexDouble = False
isIntegral QInt8 = False
isIntegral QUInt8 = False
isIntegral QInt32 = False
isIntegral BFloat16 = False

isComplex :: DType -> Bool
isComplex Bool = False
isComplex UInt8 = False
isComplex Int8 = False
isComplex Int16 = False
isComplex Int32 = False
isComplex Int64 = False
isComplex Half = False
isComplex Float = False
isComplex Double = False
isComplex ComplexHalf = True
isComplex ComplexFloat = True
isComplex ComplexDouble = True
isComplex QInt8 = False
isComplex QUInt8 = False
isComplex QInt32 = False
isComplex BFloat16 = False

byteLength :: DType -> Int
byteLength dtype =
  case dtype of
    Bool -> 1
    UInt8 -> 1
    Int8 -> 1
    Int16 -> 2
    Int32 -> 4
    Int64 -> 8
    Half -> 2
    Float -> 4
    Double -> 8
    ComplexHalf -> 4
    ComplexFloat -> 8
    ComplexDouble -> 16
    QInt8 -> 1
    QUInt8 -> 1
    QInt32 -> 4
    BFloat16 -> 2
