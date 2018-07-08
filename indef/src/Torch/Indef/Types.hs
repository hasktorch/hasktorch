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
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
module Torch.Indef.Types
  ( module Sig
  , THDebug(..)

  , DimVal(..)

  , Step(..), Stride(..), StorageOffset(..), Size(..), KeepDim(..), fromKeepDim, keep, ignore, SortOrder(..), TopKOrder(..)
  , StorageSize(..), AllocatorContext(..), Index(..)

  , manage', joinIO

  , (.:), (..:), shuffle2, shuffle2'2, shuffle3, shuffle3'2

  , withGen

  , withState
  , withDynamicState, withStorageState
  , with2DynamicState
  , with3DynamicState
  , mkDynamic, mkStorage
  , mkDynamicIO, mkStorageIO
  ) where

import Foreign
import Foreign.C.Types
import Foreign.Ptr
import GHC.Int (Int64(..), Int32(..))
import Control.Monad.Managed
import Numeric.Dimensions
import qualified Foreign.Marshal.Array as FM

import Control.Arrow
import Control.Monad
import Control.Monad.IO.Class
import Control.Monad.Reader.Class
import Torch.Types.TH (C'THState)
import GHC.TypeLits

import Torch.Sig.State as Sig
import Torch.Sig.Types as Sig
import Torch.Sig.Types.Global as Sig

import qualified Numeric.Dimensions (KnownDim)
import qualified Torch.Types.TH as TH
import qualified Torch.FFI.TH.Long.Storage as TH
import qualified Torch.Sig.Tensor.Memory as SigTen
import qualified Torch.Sig.Storage.Memory as SigStore


-------------------------------------------------------------------------------

-- | From
-- https://github.com/pytorch/pytorch/blob/c61f0217a536d19c9ff3290067ddcbb9ce3a5c6a/aten/src/THNN/Reduction.h
--
-- NB: Keep this in sync with Reduction class in torch/nn/modules/functional.py
-- These constants control the reduction behavior of loss functions.
-- Ideally, this would be a scoped enum, but jit doesn't support that
data Reduction
  = NoReduce         -- ^ Do not reduce
  | ElementwiseMean  -- ^ Sum losses and take mean over each individually computed loss element
  | Sum              -- ^ Sum losses

-------------------------------------------------------------------------------
-- helpers for dimensions:

-- | term-level representation of an index.
newtype DimVal = DimVal Int32
  deriving (Bounded, Enum, Eq, Integral, Num, Ord, Read, Real, Show)

-- | term-level representation of an index.
newtype Index = Index Int64
  deriving (Bounded, Enum, Eq, Integral, Num, Ord, Read, Real, Show)

{-# DEPRECATED DimVal, Index "Use dimensions package's Idx instead" #-}

{-
transferDims :: Proxy (ds::[Nat]) -> Dim ds
transferDims p = undefined
 where

go :: forall f m . Proxy (m::[Nat]) -> Dim (f :: [Nat])
go _ =
  if null (fromSing (sing :: Sing m))
  then (D  :: Dim f)
  else (Dn :: (x:xs) ~ m => Dim (x::Nat)) :* (go (Proxy :: (x:xs) ~ m => Proxy xs))
-- -}

-- Helper function to debug dimensions package. We return @Integral i@ in case we need to cast directly to C-level types.



-------------------------------------------------------------------------------

-- | newtype wrapper around the C-level representation of a tensor's internal
-- 'Storage' stride for each dimension.
newtype Stride = Stride CLLong
  deriving (Bounded, Enum, Eq, Integral, Num, Ord, Read, Real, Show)

-- | newtype wrapper around the C-level representation of a dimension's size
newtype Size = Size CLLong
  deriving (Bounded, Enum, Eq, Integral, Num, Ord, Read, Real, Show)

-- | newtype wrapper around the C-level representation of a storage offset
newtype StorageOffset = StorageOffset CPtrdiff
  deriving (Bounded, Enum, Eq, Integral, Num, Ord, Read, Real, Show)

-- | Represents the size of storage, should be CPtrdiff to match with the C internals
newtype StorageSize = StorageSize CPtrdiff
  deriving (Bounded, Enum, Eq, Integral, Num, Ord, Read, Real, Show)

-- | newtype wrapper around the C-level representation of a step size
newtype Step = Step CLong
  deriving (Bounded, Enum, Eq, Integral, Num, Ord, Read, Real, Show)

-- | haskell representation of a CInt which determines whether or not to return dimensions
newtype KeepDim = KeepDim { keepIt :: Bool }
  deriving (Bounded, Enum, Eq, Ord, Read, Show)

-- | cast a 'KeepDim' to a numerical representation.
--
-- NOTE: don't bind the @i@ in case there are some differences between THC and TH
fromKeepDim :: Integral i => Maybe KeepDim -> i
fromKeepDim = maybe 0 (fromIntegral . fromEnum)

-- | smart constructors for keepdim since we don't get inference for free like Num
keep,  ignore :: KeepDim
(keep, ignore) = (KeepDim True, KeepDim False)

-- | Simple datatype to represent sort order arguments which torch provides to us.
data SortOrder = Ascending | Descending
  deriving (Eq, Show, Ord, Enum, Bounded)

-- | Simple datatype to represent arguments for a topk function.
--
-- See https://github.com/torch/torch7/blob/75a86469aa9e2f5f04e11895b269ec22eb0e4687/lib/TH/generic/THTensorMath.c#L2545
data TopKOrder = KAscending | KNone | KDescending
  deriving (Eq, Show, Ord, Enum, Bounded)

-- | this is supposed to represent the AllocatorContext, but it should not be exposed to a user.
newtype AllocatorContext = AllocatorContext (Ptr ())
{-# WARNING AllocatorContext "this should not be used or referenced -- we are still figuring out what to do with this." #-}

-------------------------------------------------------------------------------


-- | helper functions to start using the Managed Monad more.
--
-- FIXME: Try to replace a lot of the below with this function, but ultimately try to remove this helper.
manage' :: (c -> ForeignPtr a) -> c -> Managed (Ptr a)
manage' fn c = managed (withForeignPtr (fn c))

-- | helper function to join MonadIO and IO.
joinIO :: MonadIO m => m (IO x) -> m x
joinIO c = join (liftIO <$> c)

-- | The blackbird combinator.
--
-- (stites): This happens often enough that I'm pulling in the blackbird
--
-- FIXME(stites): remove this
(.:) :: (b -> c) -> (a0 -> a1 -> b) -> a0 -> a1 -> c
(.:) = (.) . (.)
infixl 5 .:

-- | even more blackbird
--
-- FIXME(stites): remove this
(..:) :: (b -> c) -> (a0 -> a1 -> a2 -> b) -> a0 -> a1 -> a2 -> c
(..:) = (.) . (.) . (.)
infixl 5 ..:

-- | shuffle 2 arguments for pointfree raw-ffi functions.
--
-- FIXME(stites): remove this
shuffle2 :: (a -> b -> c -> d) -> c -> a -> b -> d
shuffle2 fn c a b = fn a b c

-- | shuffle the first two arguments two positions to the right for pointfree raw-ffi functions.
--
-- FIXME(stites): remove this
shuffle2'2 :: (a -> b -> c -> d -> e) -> c -> d -> a -> b -> e
shuffle2'2 fn c d a b = fn a b c d

-- | shuffle the first three arguments to the right for pointfree raw-ffi functions.
--
-- FIXME(stites): remove this
shuffle3 :: (a -> b -> c -> d -> e) -> d -> a -> b -> c -> e
shuffle3 fn d a b c = fn a b c d

-- | shuffle the first three arguments two positions to the right for pointfree raw-ffi functions.
--
-- FIXME(stites): remove this
shuffle3'2 :: (a -> b -> c -> d -> e -> f) -> d -> e -> a -> b -> c -> f
shuffle3'2 fn d e a b c = fn a b c d e

-- | run a function against the internal reference of a torch generator.
withGen :: Sig.Generator -> (Ptr CGenerator -> IO x) -> IO x
withGen g fn = withForeignPtr (Sig.rng g) fn

-- | run a function with a managed state's raw internal pointer.
withState :: Sig.State -> (Ptr Sig.CState ->IO x) -> IO x
withState s = withForeignPtr (Sig.asForeign s)

-- | run a function with access to a tensor's underlying state and C-tensor.
withDynamicState :: Sig.Dynamic -> (Ptr Sig.CState -> Ptr Sig.CTensor -> IO x) -> IO x
withDynamicState t fn = do
  withForeignPtr (Sig.dynamicStateRef t) $ \sref ->
    withForeignPtr (Sig.ctensor t) $ \tref ->
      fn sref tref

-- | run a function with two tensors with reference to the first tensor's underlying state.
with2DynamicState
  :: Sig.Dynamic
  -> Sig.Dynamic
  -> (Ptr Sig.CState -> Ptr Sig.CTensor -> Ptr Sig.CTensor -> IO x)
  -> IO x
with2DynamicState t0 t1 fn = do
  withDynamicState t0 $ \s' t0' ->
    withForeignPtr (Sig.ctensor t1) $ \t1' ->
      fn s' t0' t1'

-- | run a function with three tensors with reference to the first tensor's underlying state.
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

-- | smart constructor for a 'Sig.Dynamic' tensor
mkDynamic :: Ptr Sig.CState -> Ptr Sig.CTensor -> IO Sig.Dynamic
mkDynamic s t = Sig.dynamic
  <$> Sig.manageState s
  <*> newForeignPtrEnv SigTen.p_free s t

-- | smart constructor for a 'Sig.Dynamic' tensor with a given builder function.
mkDynamicIO :: (Ptr Sig.CState -> IO (Ptr Sig.CTensor)) -> IO Sig.Dynamic
mkDynamicIO builder = Sig.newCState >>= \s ->
  builder s >>= mkDynamic s

-- | run a function with access to a 'Sig.Storage's underlying state and C-reference.
withStorageState :: Sig.Storage -> (Ptr Sig.CState -> Ptr Sig.CStorage -> IO x) -> IO x
withStorageState t fn = do
  withForeignPtr (Sig.storageStateRef t) $ \sref ->
    withForeignPtr (Sig.cstorage t) $ \tref ->
      fn sref tref

-- | smart constructor for 'Sig.Storage'
mkStorage :: Ptr Sig.CState -> Ptr Sig.CStorage -> IO Sig.Storage
mkStorage s t = Sig.storage
  <$> Sig.manageState s
  <*> newForeignPtrEnv SigStore.p_free s t

-- | smart constructor for 'Sig.Storage' with a given builder function.
mkStorageIO :: (Ptr Sig.CState -> IO (Ptr Sig.CStorage)) -> IO Sig.Storage
mkStorageIO builder = Sig.newCState >>= \s ->
  builder s >>= mkStorage s

-- -------------------------------------------------------------------------------

-- | Class to print out C-references when working debugging segfaults.
class THDebug t where
  -- | print out all possible references we can find.
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

