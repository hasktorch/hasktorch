{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE UndecidableSuperClasses #-}

module Torch.GraduallyTyped.DType where

import Data.Kind (Constraint, Type)
import Torch.DType (DType (..))
import Torch.GraduallyTyped.Prelude (Concat)

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

data SDataType (dType :: DataType DType) where
  SUncheckedDataType :: DType -> SDataType 'UncheckedDataType
  SDataType :: forall dType. KnownDType dType => SDataType ('DataType dType)

type family DTypeF (dataType :: DataType DType) :: DType where
  DTypeF ('DataType dType) = dType

sDType :: forall dataType. SDataType dataType -> DType
sDType (SUncheckedDataType dataType) = dataType
sDType SDataType = dTypeVal @(DTypeF dataType)

class KnownDataType (dataType :: DataType DType) where
  dataTypeVal :: DataType DType

instance KnownDataType 'UncheckedDataType where
  dataTypeVal = UncheckedDataType

instance
  (KnownDType dType) =>
  KnownDataType ( 'DataType dType)
  where
  dataTypeVal = DataType (dTypeVal @dType)

class
  -- DataTypeConstraint dataType (GetDataTypes f) =>
  WithDataTypeC (dataType :: DataType DType) (f :: Type)
  where
  type WithDataTypeF dataType f :: Type
  withDataType :: (DType -> f) -> WithDataTypeF dataType f
  withoutDataType :: WithDataTypeF dataType f -> (DType -> f)

instance
  -- DataTypeConstraint 'UncheckedDataType (GetDataTypes f) =>
  WithDataTypeC 'UncheckedDataType f
  where
  type WithDataTypeF 'UncheckedDataType f = DType -> f
  withDataType = id
  withoutDataType = id

instance
  ( -- DataTypeConstraint ( 'DataType dType) (GetDataTypes f),
    KnownDType dType
  ) =>
  WithDataTypeC ( 'DataType dType) f
  where
  type WithDataTypeF ( 'DataType dType) f = f
  withDataType f = f (dTypeVal @dType)
  withoutDataType = const

type family DataTypeConstraint (datatype :: DataType DType) (datatypes :: [DataType DType]) :: Constraint where
  DataTypeConstraint _ '[] = ()
  DataTypeConstraint datatype '[datatype'] = datatype ~ datatype'
  DataTypeConstraint _ _ = ()

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
type family GetDataTypes (f :: k) :: [DataType DType] where
  GetDataTypes (a :: DataType DType) = '[a]
  GetDataTypes (f g) = Concat (GetDataTypes f) (GetDataTypes g)
  GetDataTypes _ = '[]
