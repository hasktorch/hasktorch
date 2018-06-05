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
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
module Torch.Indef.Types
  ( module X
  , module Sig
  , THDebug(..)

  , Step(..), Stride(..), StorageOffset(..), Size(..), KeepDim(..), fromKeepDim, keep, ignore, SortOrder(..), TopKOrder(..)
  , StorageSize(..), AllocatorContext(..), Index(..)

  , manage', joinIO

  , (.:), (..:), shuffle2, shuffle2'2, shuffle3, shuffle3'2

  , withIx, withIxStorage, withMask
  , withGen

  , withState
  , withDynamicState, withStorageState, withDynamicStateAndStorage
  , with2DynamicState
  , with3DynamicState
  , mkDynamic, mkStorage
  , mkDynamicIO, mkStorageIO
  ) where

import Foreign as X (ForeignPtr, newForeignPtrEnv, withForeignPtr, newForeignPtr, FinalizerPtr)
import Foreign.Ptr as X
import GHC.Int (Int64(..))
import Control.Monad.Managed as X
import qualified Foreign.Marshal.Array as FM

import Control.Arrow
import Control.Monad
import Control.Monad.IO.Class as X
import Control.Monad.Reader.Class as X
import Torch.Types.TH as X (C'THState)
import GHC.TypeLits

import Torch.Sig.State as Sig
import Torch.Sig.Types as Sig
import Torch.Sig.Types.Global as Sig

import Torch.Dimensions as X
import qualified Torch.Types.TH as TH
import qualified Torch.FFI.TH.Long.Storage as TH
import qualified Torch.Sig.Tensor.Memory as SigTen
import qualified Torch.Sig.Storage.Memory as SigStore


-------------------------------------------------------------------------------

-- Maybe better served as a newtype of Foreign.C.Types.CLLong
newtype Stride = Stride Int64
  deriving (Bounded, Enum, Eq, Integral, Num, Ord, Read, Real, Show)

-- Maybe better served as a newtype of Foreign.C.Types.CLLong
newtype Size = Size Int64
  deriving (Bounded, Enum, Eq, Integral, Num, Ord, Read, Real, Show)

-- Maybe better served as a newtype of Foreign.C.Types.CPtrDiff
newtype StorageOffset = StorageOffset Int64
  deriving (Bounded, Enum, Eq, Integral, Num, Ord, Read, Real, Show)

-- Maybe better served as a newtype of Foreign.C.Types.CLong
newtype Step = Step Int64
  deriving (Bounded, Enum, Eq, Integral, Num, Ord, Read, Real, Show)

-- haskell representation of a CInt which determines whether or not to return dimensions
newtype KeepDim = KeepDim { keepIt :: Bool }
  deriving (Bounded, Enum, Eq, Ord, Read, Show)

-- don't bind the @i@ in case there are some differences between THC and TH
fromKeepDim :: Integral i => Maybe KeepDim -> i
fromKeepDim = maybe 0 (fromIntegral . fromEnum)

-- smart constructors for keepdim since we don't get inference for free like Num
keep,  ignore :: KeepDim
(keep, ignore) = (KeepDim True, KeepDim False)

data SortOrder = Ascending | Descending
  deriving (Eq, Show, Ord, Enum, Bounded)

-- https://github.com/torch/torch7/blob/75a86469aa9e2f5f04e11895b269ec22eb0e4687/lib/TH/generic/THTensorMath.c#L2545
data TopKOrder = KAscending | KNone | KDescending
  deriving (Eq, Show, Ord, Enum, Bounded)

-- should be CPtrdiff
newtype StorageSize = StorageSize Int64
  deriving (Bounded, Enum, Eq, Integral, Num, Ord, Read, Real, Show)

newtype AllocatorContext = AllocatorContext (Ptr ())

newtype Index = Index Int64
  deriving (Bounded, Enum, Eq, Integral, Num, Ord, Read, Real, Show)

-------------------------------------------------------------------------------


-- Try to replace a lot of the below with these functions:
manage' :: (c -> ForeignPtr a) -> c -> Managed (Ptr a)
manage' fn c = managed (withForeignPtr (fn c))

joinIO :: MonadIO m => m (IO x) -> m x
joinIO c = join (liftIO <$> c)

-- (stites): This happens often enough that I'm pulling in the blackbird
(.:) :: (b -> c) -> (a0 -> a1 -> b) -> a0 -> a1 -> c
(.:) = (.) . (.)
infixl 5 .:

(..:) :: (b -> c) -> (a0 -> a1 -> a2 -> b) -> a0 -> a1 -> a2 -> c
(..:) = (.) . (.) . (.)
infixl 5 ..:

shuffle2 :: (a -> b -> c -> d) -> c -> a -> b -> d
shuffle2 fn c a b = fn a b c

shuffle2'2 :: (a -> b -> c -> d -> e) -> c -> d -> a -> b -> e
shuffle2'2 fn c d a b = fn a b c d

shuffle3 :: (a -> b -> c -> d -> e) -> d -> a -> b -> c -> e
shuffle3 fn d a b c = fn a b c d

shuffle3'2 :: (a -> b -> c -> d -> e -> f) -> d -> e -> a -> b -> c -> f
shuffle3'2 fn d e a b c = fn a b c d e

withGen :: Sig.Generator -> (Ptr CGenerator -> IO x) -> IO x
withGen g fn = withForeignPtr (Sig.rng g) fn

withIx :: Sig.IndexDynamic -> (Ptr CIndexTensor -> IO x) -> IO x
withIx ix fn = withForeignPtr (snd $ Sig.longDynamicState ix) fn

withIxStorage :: Sig.IndexStorage -> (Ptr CLongStorage -> IO x) -> IO x
withIxStorage ix fn = withForeignPtr (snd $ Sig.longStorageState ix) fn

withMask :: Sig.MaskDynamic -> (Ptr CMaskTensor -> IO x) -> IO x
withMask ix fn = withForeignPtr (snd $ Sig.byteDynamicState ix) fn

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

with3DynamicState
  :: Sig.Dynamic
  -> Sig.Dynamic
  -> Sig.Dynamic
  -> (Ptr Sig.CState -> Ptr Sig.CTensor -> Ptr Sig.CTensor -> Ptr Sig.CTensor -> IO x)
  -> IO x
with3DynamicState t0 t1 t2 fn = do
  with2DynamicState t0 t1 $ \s' t0' t1' ->
    withForeignPtr (Sig.ctensor t2) $ \t2' ->
      fn s' t0' t1' t2'

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

withDynamicStateAndStorage :: Sig.Dynamic -> Sig.Storage -> (Ptr Sig.CState -> Ptr Sig.CTensor -> Ptr Sig.CStorage -> IO x) -> IO x
withDynamicStateAndStorage t s fn =
  withDynamicState t $ \state' t' ->
    withForeignPtr (Sig.cstorage s) (fn state' t')

-- -------------------------------------------------------------------------------

class THDebug t where
  printRefs :: t -> IO ()

instance THDebug Sig.Storage where
  printRefs t = do
    let (s, c) = Sig.storageState t
    putStrLn $ "State reference   : " ++ show s
    putStrLn $ "CStorage reference: " ++ show s

instance THDebug Sig.Dynamic where
  printRefs t = do
    let (s, c) = Sig.dynamicState t
    putStrLn $ "State reference  : " ++ show s
    putStrLn $ "CTensor reference: " ++ show s

instance THDebug (Sig.Tensor d) where
  printRefs = printRefs . Sig.asDynamic

