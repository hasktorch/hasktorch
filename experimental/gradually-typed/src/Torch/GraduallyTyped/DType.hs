{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE StandaloneKindSignatures #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}

module Torch.GraduallyTyped.DType where

import Data.Kind (Type)
import Data.Singletons (Sing, SingI (..), SingKind (..), SomeSing (..), withSomeSing)
import Data.Singletons.TH (genSingletons)
import Torch.GraduallyTyped.Prelude (Concat, IsChecked (..))
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

class KnownDType (dType :: DType) where
  dTypeVal :: DType

instance KnownDType 'Bool where
  dTypeVal = Bool

instance KnownDType 'UInt8 where
  dTypeVal = UInt8

instance KnownDType 'Int8 where
  dTypeVal = Int8

instance KnownDType 'Int16 where
  dTypeVal = Int16

instance KnownDType 'Int32 where
  dTypeVal = Int32

instance KnownDType 'Int64 where
  dTypeVal = Int64

instance KnownDType 'Half where
  dTypeVal = Half

instance KnownDType 'Float where
  dTypeVal = Float

instance KnownDType 'Double where
  dTypeVal = Double

-- | Data type to represent whether or not the tensor data type is checked, that is, known to the compiler.
data DataType (dType :: Type) where
  -- | The tensor data type is unknown to the compiler.
  UncheckedDataType :: forall dType. DataType dType
  -- | The tensor data type is known to the compiler.
  DataType :: forall dType. dType -> DataType dType
  deriving (Show)

data SDataType (dataType :: DataType DType) where
  SUncheckedDataType :: DType -> SDataType 'UncheckedDataType
  SDataType :: forall dType. SDType dType -> SDataType ('DataType dType)

deriving stock instance Show (SDataType (dataType :: DataType DType))

type instance Sing = SDataType

instance SingI dType => SingI ('DataType (dType :: DType)) where
  sing = SDataType $ sing @dType

instance SingKind (DataType DType) where
  type Demote (DataType DType) = IsChecked DType
  fromSing (SUncheckedDataType dType) = Unchecked dType
  fromSing (SDataType dType) = Checked . fromSing $ dType
  toSing (Unchecked dType) = SomeSing . SUncheckedDataType $ dType
  toSing (Checked dType) = withSomeSing dType $ SomeSing . SDataType

class KnownDataType (dataType :: DataType DType) where
  dataTypeVal :: DataType DType

instance KnownDataType 'UncheckedDataType where
  dataTypeVal = UncheckedDataType

instance
  (KnownDType dType) =>
  KnownDataType ('DataType dType)
  where
  dataTypeVal = DataType (dTypeVal @dType)

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
type GetDataTypes :: k -> [DataType DType]
type family GetDataTypes f where
  GetDataTypes (a :: DataType DType) = '[a]
  GetDataTypes (f g) = Concat (GetDataTypes f) (GetDataTypes g)
  GetDataTypes _ = '[]
