module Torch.Indef.Storage
  ( module X
  , storagedata
  , newWithData
  ) where

import Torch.Indef.Storage.Copy as X
import Torch.Indef.Storage.Internal as X

import Control.Monad.Managed
import Foreign hiding (with, new)
import GHC.Exts (IsList(..))
import System.IO.Unsafe
import qualified Foreign.CUDA.Ptr             as CUDA
import qualified Foreign.CUDA.Runtime.Marshal as CUDA

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
          sz     <- fromIntegral <$> Sig.c_size st s'
          creals <- CUDA.DevicePtr <$> Sig.c_data st s'
          CUDA.peekListArray sz creals
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
    <*> liftIO (CUDA.useDevicePtr <$> CUDA.newListArray (hs2cReal <$> pr))
    <*> pure (fromIntegral pd)
{-# NOINLINE newWithData #-}


instance IsList Storage where
  type Item Storage = HsReal
  toList = storagedata
  fromList pr = newWithData pr (fromIntegral $ length pr)

instance Show Storage where
  show = show . storagedata

{-
instance Class.GPUStorage t where
  c_getDevice :: t -> io Int
-}
