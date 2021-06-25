{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneKindSignatures #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE UndecidableSuperClasses #-}

module Torch.GraduallyTyped.DType where

import Data.Kind (Type)
import Data.Singletons (Sing, SingKind (..), SomeSing (..), withSomeSing)
import Data.Singletons.TH (genSingletons)
import Torch.DType (DType (..))
import Torch.GraduallyTyped.Prelude (Concat, IsChecked (..))

genSingletons [''DType]

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
  SDataType :: forall dType. SDType dType -> SDataType ('DataType dType)

type instance Sing = SDataType

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

class WithDataTypeC (dataType :: DataType DType) (f :: Type) where
  type WithDataTypeF dataType f :: Type
  withDataType :: (DType -> f) -> WithDataTypeF dataType f
  withoutDataType :: WithDataTypeF dataType f -> (DType -> f)

instance WithDataTypeC 'UncheckedDataType f where
  type WithDataTypeF 'UncheckedDataType f = DType -> f
  withDataType = id
  withoutDataType = id

instance
  KnownDType dType =>
  WithDataTypeC ('DataType dType) f
  where
  type WithDataTypeF ('DataType dType) f = f
  withDataType f = f (dTypeVal @dType)
  withoutDataType = const

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
