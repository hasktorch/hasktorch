{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE TypeInType #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE Rank2Types #-}
{-# LANGUAGE UndecidableInstances #-}
module Torch.Core.Tensor.Types
  ( THForeignRef(..)
  , THForeignType

  , TensorByte(..)
  , TensorByteRaw
  , TensorChar(..)
  , TensorCharRaw
  , TensorDouble(..)
  , TensorDoubleRaw
  , TensorFloat(..)
  , TensorFloatRaw
  , TensorInt(..)
  , TensorIntRaw
  , TensorLong(..)
  , TensorLongRaw
  , TensorShort(..)
  , TensorShortRaw

  , withManaged4
  , withManaged3
  , withManaged2
  , withManaged1
  ) where


import Control.Monad.IO.Class (liftIO)
import Control.Monad.Managed (managed, runManaged)
import Foreign (ForeignPtr, Ptr, withForeignPtr)

import THTypes

-- ========================================================================= --

class THForeignRef t where
  construct :: ForeignPtr (THForeignType t) -> t
  getForeign :: t -> ForeignPtr (THForeignType t)

type family THForeignType t

-------------------------------------------------------------------------------

newtype TensorByte = TensorByte { tbTensor :: ForeignPtr CTHByteTensor }
  deriving (Show, Eq)

instance THForeignRef TensorByte where
  construct = TensorByte
  getForeign = tbTensor

type instance THForeignType TensorByte = CTHByteTensor
type TensorByteRaw = Ptr CTHByteTensor

-------------------------------------------------------------------------------

newtype TensorChar = TensorChar { tcTensor :: ForeignPtr CTHCharTensor }
  deriving (Show, Eq)

instance THForeignRef TensorChar where
  construct = TensorChar
  getForeign = tcTensor

type instance THForeignType TensorChar = CTHCharTensor
type TensorCharRaw   = Ptr CTHCharTensor

-------------------------------------------------------------------------------

newtype TensorDouble = TensorDouble { tdTensor :: ForeignPtr CTHDoubleTensor }
  deriving (Show, Eq)

instance THForeignRef TensorDouble where
  construct = TensorDouble
  getForeign = tdTensor

type instance THForeignType TensorDouble = CTHDoubleTensor
type TensorDoubleRaw = Ptr CTHDoubleTensor

-------------------------------------------------------------------------------

newtype TensorFloat = TensorFloat { tfTensor :: ForeignPtr CTHFloatTensor }
  deriving (Show, Eq)

instance THForeignRef TensorFloat where
  construct = TensorFloat
  getForeign = tfTensor

type instance THForeignType TensorFloat = CTHFloatTensor
type TensorFloatRaw  = Ptr CTHFloatTensor

-------------------------------------------------------------------------------

newtype TensorInt = TensorInt { tiTensor :: ForeignPtr CTHIntTensor }
  deriving (Show, Eq)

instance THForeignRef TensorInt where
  construct = TensorInt
  getForeign = tiTensor

type instance THForeignType TensorInt = CTHIntTensor
type TensorIntRaw    = Ptr CTHIntTensor

-------------------------------------------------------------------------------

newtype TensorLong = TensorLong { tlTensor :: ForeignPtr CTHLongTensor }
  deriving (Show, Eq)

instance THForeignRef TensorLong where
  construct = TensorLong
  getForeign = tlTensor

type instance THForeignType TensorLong = CTHLongTensor
type TensorLongRaw = Ptr CTHLongTensor

-------------------------------------------------------------------------------

newtype TensorShort = TensorShort { tsTensor :: ForeignPtr CTHShortTensor }
  deriving (Show, Eq)

instance THForeignRef TensorShort where
  construct = TensorShort
  getForeign = tsTensor

type instance THForeignType TensorShort = CTHShortTensor
type TensorShortRaw = Ptr CTHShortTensor

-- ========================================================================= --
-- helper functions:

withManaged4
  :: (THForeignRef t)
  => (Ptr (THForeignType t) -> Ptr (THForeignType t) -> Ptr (THForeignType t) -> Ptr (THForeignType t) -> IO ())
  -> t -> t -> t -> t -> IO ()
withManaged4 fn resA resB a b = runManaged $ do
  resBRaw <- managed (withForeignPtr (getForeign resB))
  resARaw <- managed (withForeignPtr (getForeign resA))
  bRaw <- managed (withForeignPtr (getForeign b))
  aRaw <- managed (withForeignPtr (getForeign a))
  liftIO (fn resBRaw resARaw bRaw aRaw)

withManaged3
  :: (THForeignRef t)
  => (Ptr (THForeignType t) -> Ptr (THForeignType t) -> Ptr (THForeignType t) -> IO ())
  -> t -> t -> t -> IO ()
withManaged3 fn a b c = runManaged $ do
  a' <- managed (withForeignPtr (getForeign a))
  b' <- managed (withForeignPtr (getForeign b))
  c' <- managed (withForeignPtr (getForeign c))
  liftIO (fn a' b' c')

withManaged2
  :: (THForeignRef t)
  => (Ptr (THForeignType t) -> Ptr (THForeignType t) -> IO ())
  -> t -> t -> IO ()
withManaged2 fn resA a = runManaged $ do
  resARaw <- managed (withForeignPtr (getForeign resA))
  aRaw <- managed (withForeignPtr (getForeign a))
  liftIO (fn resARaw aRaw)

withManaged1
  :: (THForeignRef t)
  => (Ptr (THForeignType t) -> IO ())
  -> t -> IO ()
withManaged1 fn a = runManaged $ do
  a' <- managed (withForeignPtr (getForeign a))
  liftIO (fn a')


