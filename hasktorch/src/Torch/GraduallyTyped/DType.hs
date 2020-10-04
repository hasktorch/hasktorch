{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE UndecidableSuperClasses #-}
{-# LANGUAGE NoStarIsType #-}

module Torch.GraduallyTyped.DType where

import Data.Int (Int16)
import Data.Kind (Type)
import Data.Proxy (Proxy (..))
import Foreign.ForeignPtr (ForeignPtr)
import GHC.TypeLits
  ( KnownNat,
    KnownSymbol,
    Nat,
    Symbol,
    natVal,
    symbolVal,
  )
import System.IO.Unsafe (unsafePerformIO)
import Torch.DType (DType (..))
import Torch.Internal.Cast (cast0, cast1)
import Torch.Internal.Class (Castable (..))
import qualified Torch.Internal.Managed.Autograd as ATen
import qualified Torch.Internal.Managed.Cast as ATen ()
import qualified Torch.Internal.Managed.Type.Context as ATen
import qualified Torch.Internal.Managed.Type.Tensor as ATen
import qualified Torch.Internal.Type as ATen (Tensor, TensorList)

data DataType (dType :: Type) where
  AnyDataType :: forall dType. DataType dType
  DataType :: forall dType. dType -> DataType dType

class KnownDataType (dataType :: DataType DType) where
  dataTypeVal :: DataType DType

instance KnownDataType 'AnyDataType where
  dataTypeVal = AnyDataType

instance KnownDataType ('DataType 'Bool) where
  dataTypeVal = DataType Bool

instance KnownDataType ('DataType 'UInt8) where
  dataTypeVal = DataType UInt8

instance KnownDataType ('DataType 'Int8) where
  dataTypeVal = DataType Int8

instance KnownDataType ('DataType 'Int16) where
  dataTypeVal = DataType Int16

instance KnownDataType ('DataType 'Int32) where
  dataTypeVal = DataType Int32

instance KnownDataType ('DataType 'Int64) where
  dataTypeVal = DataType Int64

instance KnownDataType ('DataType 'Half) where
  dataTypeVal = DataType Half

instance KnownDataType ('DataType 'Float) where
  dataTypeVal = DataType Float

instance KnownDataType ('DataType 'Double) where
  dataTypeVal = DataType Double

class WithDataTypeC (isAnyDataType :: Bool) (dataType :: DataType DType) (f :: Type) where
  type WithDataTypeF isAnyDataType f :: Type
  withDataType :: (DType -> f) -> WithDataTypeF isAnyDataType f

instance WithDataTypeC 'True dataType f where
  type WithDataTypeF 'True f = DType -> f
  withDataType = id

instance (KnownDataType dataType) => WithDataTypeC 'False dataType f where
  type WithDataTypeF 'False f = f
  withDataType f = case dataTypeVal @dataType of DataType dataType -> f dataType
