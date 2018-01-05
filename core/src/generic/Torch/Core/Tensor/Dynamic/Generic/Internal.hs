{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE FlexibleContexts #-}
module Torch.Core.Tensor.Dynamic.Generic.Internal
  ( THTensor(..)
  , THPtrType
  , GenericOps'
  , GenericMath'
  , LapackOps'
  , HaskReal'

  , TensorByte(..)
  , TensorDouble(..)
  , TensorFloat(..)

  , withManaged4
  , withManaged3
  , withManaged2
  ) where

import Control.Monad.IO.Class (liftIO)
import Control.Monad.Managed (managed, runManaged)
import Foreign (Ptr, ForeignPtr, withForeignPtr, newForeignPtr, finalizeForeignPtr)

import Torch.Core.Tensor.Generic.Internal
import Torch.Core.Tensor.Generic (GenericOps, GenericMath, LapackOps)

type LapackOps' t = (LapackOps (THPtrType t), THTensor t)
type GenericMath' t = (GenericOps (THPtrType t), GenericMath (THPtrType t), THTensor t)
type GenericOps' t = (GenericOps (THPtrType t), THTensor t)
type HaskReal' t = (HaskReal (THPtrType t))


-- FIXME: should be called "THRef" not "THTensor"
class THTensor t where
  construct :: ForeignPtr (THPtrType t) -> t
  getForeign :: t -> ForeignPtr (THPtrType t)

type family THPtrType t

-------------------------------------------------------------------------------

newtype TensorByte = TensorByte { tbTensor :: ForeignPtr CTHByteTensor }
  deriving (Show, Eq)

instance THTensor TensorByte where
  construct = TensorByte
  getForeign = tbTensor

type instance THPtrType TensorByte = CTHByteTensor
-------------------------------------------------------------------------------

newtype TensorDouble = TensorDouble { tdTensor :: ForeignPtr CTHDoubleTensor }
  deriving (Show, Eq)

instance THTensor TensorDouble where
  construct = TensorDouble
  getForeign = tdTensor

type instance THPtrType TensorDouble = CTHDoubleTensor
-------------------------------------------------------------------------------

newtype TensorFloat = TensorFloat { tfTensor :: ForeignPtr CTHFloatTensor }
  deriving (Show, Eq)

instance THTensor TensorFloat where
  construct = TensorFloat
  getForeign = tfTensor

type instance THPtrType TensorFloat = CTHFloatTensor
-------------------------------------------------------------------------------

newtype TensorInt = TensorInt { tiTensor :: ForeignPtr CTHIntTensor }
  deriving (Show, Eq)

instance THTensor TensorInt where
  construct = TensorInt
  getForeign = tiTensor

type instance THPtrType TensorInt = CTHIntTensor
-------------------------------------------------------------------------------

newtype TensorLong = TensorLong { tlTensor :: ForeignPtr CTHLongTensor }
  deriving (Show, Eq)

instance THTensor TensorLong where
  construct = TensorLong
  getForeign = tlTensor

type instance THPtrType TensorLong = CTHLongTensor
-------------------------------------------------------------------------------

newtype TensorShort = TensorShort { tsTensor :: ForeignPtr CTHShortTensor }
  deriving (Show, Eq)

instance THTensor TensorShort where
  construct = TensorShort
  getForeign = tsTensor

type instance THPtrType TensorShort = CTHShortTensor
-- ========================================================================= --
-- helper functions:

withManaged4
  :: (THTensor t)
  => (Ptr (THPtrType t) -> Ptr (THPtrType t) -> Ptr (THPtrType t) -> Ptr (THPtrType t) -> IO ())
  -> t -> t -> t -> t -> IO ()
withManaged4 fn resA resB a b = runManaged $ do
  resBRaw <- managed (withForeignPtr (getForeign resB))
  resARaw <- managed (withForeignPtr (getForeign resA))
  bRaw <- managed (withForeignPtr (getForeign b))
  aRaw <- managed (withForeignPtr (getForeign a))
  liftIO (fn resBRaw resARaw bRaw aRaw)

withManaged3
  :: (THTensor t)
  => (Ptr (THPtrType t) -> Ptr (THPtrType t) -> Ptr (THPtrType t) -> IO ())
  -> t -> t -> t -> IO ()
withManaged3 fn a b c = runManaged $ do
  a' <- managed (withForeignPtr (getForeign a))
  b' <- managed (withForeignPtr (getForeign b))
  c' <- managed (withForeignPtr (getForeign c))
  liftIO (fn a' b' c')

withManaged2
  :: (THTensor t)
  => (Ptr (THPtrType t) -> Ptr (THPtrType t) -> IO ())
  -> t -> t -> IO ()
withManaged2 fn resA a = runManaged $ do
  resARaw <- managed (withForeignPtr (getForeign resA))
  aRaw <- managed (withForeignPtr (getForeign a))
  liftIO (fn resARaw aRaw)


