-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Indef.Dynamic.Types
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
-------------------------------------------------------------------------------
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE InstanceSigs #-}
module Torch.Indef.Types
  ( module X
  , ptrArray2hs

  , withState
  , withDynamicState, withStorageState
  , with2DynamicState
  , mkDynamic, mkStorage
  , mkDynamicIO, mkStorageIO

  , Sig.State
  , Sig.CState

  , Sig.CTensor, Sig.Tensor, Sig.Dynamic, Sig.MaskTensor, Sig.IndexTensor, Sig.CIndexTensor, Sig.longDynamicState
  , Sig.Storage, Sig.CStorage, Sig.HsReal, Sig.dynamicState, Sig.storageState
  , Sig.hs2cReal, Sig.c2hsReal

  , newForeignPtrEnv, withForeignPtr
  , Ptr, ForeignPtr
  ) where

import Foreign
import Torch.Class.Types
import qualified Foreign.Marshal.Array as FM

import qualified Torch.Sig.State as Sig
import qualified Torch.Sig.Types as Sig
import qualified Torch.Sig.Types.Global as Sig

import qualified Torch.Sig.Tensor.Memory as SigTen
import qualified Torch.Sig.Storage.Memory as SigStore

import Control.Monad.IO.Class as X
import Control.Monad.Reader.Class as X

-- helper function to work with pointer arrays
ptrArray2hs :: (Ptr a -> IO (Ptr Sig.CReal)) -> (Ptr a -> IO Int) -> ForeignPtr a -> IO [Sig.HsReal]
ptrArray2hs updPtrArray toSize fp = do
  sz <- withForeignPtr fp toSize
  creals <- withForeignPtr fp updPtrArray
  (fmap.fmap) Sig.c2hsReal (FM.peekArray sz creals)

-- working with dynamic and storage types:
withState :: Sig.State -> (Ptr Sig.CState ->IO x) -> IO x
withState s = withForeignPtr (Sig.asForeign s)

withDynamicState :: Sig.Dynamic -> (Ptr Sig.CState -> Ptr Sig.CTensor -> IO x) -> IO x
withDynamicState t fn = do
  withForeignPtr (Sig.dynamicStateRef t) $ \sref ->
    withForeignPtr (Sig.ctensor t) $ \tref ->
      fn sref tref

with2DynamicState
  :: Sig.Dynamic
  -> Sig.Dynamic
  -> (Ptr Sig.CState -> Ptr Sig.CTensor -> Ptr Sig.CTensor -> IO x)
  -> IO x
with2DynamicState t0 t1 fn = do
  withDynamicState t0 $ \s' t0' ->
    withForeignPtr (Sig.ctensor t1) $ \t1' ->
      fn s' t0' t1'

mkDynamic :: Ptr Sig.CState -> Ptr Sig.CTensor -> IO Sig.Dynamic
mkDynamic s t = Sig.dynamic
  <$> Sig.manageState s
  <*> newForeignPtrEnv SigTen.p_free s t

mkDynamicIO :: (Ptr Sig.CState -> IO (Ptr Sig.CTensor)) -> IO Sig.Dynamic
mkDynamicIO builder = Sig.newCState >>= \s ->
  builder s >>= mkDynamic s

withStorageState :: Sig.Storage -> (Ptr Sig.CState -> Ptr Sig.CStorage -> IO x) -> IO x
withStorageState t fn = do
  withForeignPtr (Sig.storageStateRef t) $ \sref ->
    withForeignPtr (Sig.cstorage t) $ \tref ->
      fn sref tref

mkStorage :: Ptr Sig.CState -> Ptr Sig.CStorage -> IO Sig.Storage
mkStorage s t = Sig.storage
  <$> Sig.manageState s
  <*> newForeignPtrEnv SigStore.p_free s t

mkStorageIO :: (Ptr Sig.CState -> IO (Ptr Sig.CStorage)) -> IO Sig.Storage
mkStorageIO builder = Sig.newCState >>= \s ->
  builder s >>= mkStorage s

-------------------------------------------------------------------------------
-- Storage type family instances

type instance Allocator    Sig.Storage = Sig.Allocator
type instance Generator    Sig.Storage = Sig.Generator
type instance DescBuff     Sig.Storage = Sig.DescBuff

type instance HsReal       Sig.Storage = Sig.HsReal
type instance HsAccReal    Sig.Storage = Sig.HsAccReal

-------------------------------------------------------------------------------
-- Dynamic type family instances

type instance AsDynamic    Sig.Dynamic = Sig.Dynamic
type instance HsStorage    Sig.Dynamic = Sig.Storage
type instance IndexTensor  Sig.Dynamic = Sig.IndexTensor
type instance IndexStorage Sig.Dynamic = Sig.IndexStorage
type instance MaskTensor   Sig.Dynamic = Sig.MaskTensor

type instance Allocator    Sig.Dynamic = Sig.Allocator
type instance Generator    Sig.Dynamic = Sig.Generator
type instance DescBuff     Sig.Dynamic = Sig.DescBuff

type instance HsReal       Sig.Dynamic = Sig.HsReal
type instance HsAccReal    Sig.Dynamic = Sig.HsAccReal


-------------------------------------------------------------------------------
-- Static type family instances

type instance AsDynamic    (Sig.Tensor d) = Sig.Dynamic
type instance HsStorage    (Sig.Tensor d) = Sig.Storage

type instance IndexTensor  (Sig.Tensor d) = Sig.IndexTensor
type instance IndexStorage (Sig.Tensor d) = Sig.IndexStorage
type instance MaskTensor   (Sig.Tensor d) = Sig.MaskTensor

type instance Allocator    (Sig.Tensor d) = Sig.Allocator
type instance Generator    (Sig.Tensor d) = Sig.Generator
type instance DescBuff     (Sig.Tensor d) = Sig.DescBuff

type instance HsReal       (Sig.Tensor d) = Sig.HsReal
type instance HsAccReal    (Sig.Tensor d) = Sig.HsAccReal

instance IsStatic (Sig.Tensor d) where
  asDynamic = Sig.asDynamic
  asStatic = Sig.asStatic

