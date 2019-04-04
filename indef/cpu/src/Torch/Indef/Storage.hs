module Torch.Indef.Storage
  ( module X
  , storagedata
  , newWithData
  , IsList(fromList, toList)
  ) where

import Torch.Indef.Storage.Copy as X
import Torch.Indef.Storage.Internal as X

import Control.Monad.Managed
import Foreign hiding (with, new)
import GHC.Exts (IsList(..))
import System.IO.Unsafe
import qualified Foreign.Marshal.Array as FM

import Torch.Indef.Types
import qualified Torch.Sig.Storage as Sig

-- | return the internal data of 'Storage' as a list of haskell values.
storagedata :: Storage -> [HsReal]
storagedata s =
  unsafeDupablePerformIO
    . flip with (pure . fmap c2hsReal)
    $ do
        st <- managedState
        s' <- managedStorage s
        liftIO $ do
          -- a strong dose of paranoia
          sz     <- fromIntegral <$> Sig.c_size st s'
          tmp    <- FM.mallocArray sz

          creals <- Sig.c_data st s'
          FM.copyArray tmp creals sz
          FM.peekArray sz tmp
 where
  arrayLen :: Ptr CState -> Ptr CStorage -> IO Int
  arrayLen st p = fromIntegral <$> Sig.c_size st p
{-# NOINLINE storagedata #-}

-- | make a new 'Storage' from a given list and 'StorageSize'.
--
-- FIXME: find out if 'StorageSize' always corresponds to the length of the list. If so,
-- remove it!
newWithData
  :: [HsReal]
  -> Word64   -- ^ storage size
  -> Storage
newWithData pr pd =
  unsafeDupablePerformIO
    .   withStorage
    $   Sig.c_newWithData
    <$> managedState
    <*> liftIO (FM.newArray (hs2cReal <$> pr))
    <*> pure (fromIntegral pd)
{-# NOINLINE newWithData #-}


-- -- FIXME: reintroduce these???
--
-- instance Class.CPUStorage Storage where
--   newWithAllocator :: StorageSize -> (CTHAllocatorPtr, AllocatorContext) -> IO Storage
--   newWithAllocator pd (alloc, AllocatorContext ctx) = Sig.c_newWithAllocator (fromIntegral pd) alloc ctx >>= mkStorage
--
--   newWithDataAndAllocator :: [HsReal] -> StorageSize -> (CTHAllocatorPtr, AllocatorContext) -> IO Storage
--   newWithDataAndAllocator pr pd (alloc, AllocatorContext ctx) = do
--     pr' <- FM.withArray (hs2cReal <$> pr) pure
--     s <- Sig.c_newWithDataAndAllocator pr' (fromIntegral pd) alloc ctx {-seems like it's fine to pass nullPtr-}
--     mkStorage s
--
--   swap :: Storage -> Storage -> IO ()
--   swap s0 s1 =
--     withForeignPtr (storage s0) $ \s0' ->
--       withForeignPtr (storage s1) $ \s1' ->
--         Sig.c_swap s0' s1'

instance IsList Storage where
  type Item Storage = HsReal
  toList = storagedata
  fromList pr = newWithData pr (fromIntegral $ length pr)

instance Show Storage where
  show = show . storagedata

