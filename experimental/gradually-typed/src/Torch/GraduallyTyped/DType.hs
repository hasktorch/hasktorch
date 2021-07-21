{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE StandaloneKindSignatures #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE UndecidableSuperClasses #-}

module Torch.GraduallyTyped.DType where

import Data.Kind (Type)
import Data.Singletons (Sing)
import Data.Singletons.Prelude.Check (Check, SCheck (..), type SChecked, type SUnchecked)
import Data.Singletons.TH (genSingletons)
import Torch.GraduallyTyped.Prelude (Concat)
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

genSingletons [''DType]

deriving stock instance Show (SDType (dType :: DType))

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

type SDataType :: DType -> Type

type SDataType dType = SChecked dType

pattern SDataType :: forall (a :: DType). Sing a -> SDataType a
pattern SDataType dType = SChecked dType

type SUncheckedDataType :: Type

type SUncheckedDataType = SUnchecked DType

pattern SUncheckedDataType :: DType -> SUncheckedDataType
pattern SUncheckedDataType dType = SUnchecked dType

-- >>> :kind! GetDataTypes ('DataType 'Float)
-- GetDataTypes ('DataType 'Float) :: [DataType DType]
-- = '[ 'DataType 'Float]
-- >>> :kind! GetDataTypes '[ 'DataType 'Bool, 'DataType 'Float]
-- GetDataTypes '[ 'DataType 'Bool, 'DataType 'Float] :: [DataType
--                                                          DType]
-- = '[ 'DataType 'Bool, 'DataType 'Float]
-- >>> :kind! GetDataTypes ('Just ('DataType 'Bool))
-- GetDataTypes ('Just ('DataType 'Bool)) :: [DataType DType]
-- = '[ 'DataType 'Bool]
type GetDataTypes :: k -> [Check DType Type]
type family GetDataTypes f where
  GetDataTypes (a :: Check DType Type) = '[a]
  GetDataTypes (f g) = Concat (GetDataTypes f) (GetDataTypes g)
  GetDataTypes _ = '[]
