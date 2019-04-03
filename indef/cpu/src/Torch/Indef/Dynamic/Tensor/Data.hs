Loaded package environment from /home/stites/git/hasktorch/.ghc.environment.x86_64-linux-8.4.4
Loaded package environment from /home/stites/git/hasktorch/.ghc.environment.x86_64-linux-8.4.4
module Indef.Dynamic.Tensor.Data
  ( tensordata
  ) where

import qualified Foreign.Marshal.Array     as FM

-- | Get the underlying data as a haskell list from the tensor
--
-- NOTE: This _cannot_ use a Tensor's storage size because ATen's Storage
-- allocates up to the next 64-byte line on the CPU (needs reference, this
-- is the unofficial response from \@soumith in slack).
tensordata :: Dynamic -> [HsReal]
tensordata t =
  case shape t of
    [] -> []
    ds ->
      unsafeDupablePerformIO . flip with (pure . fmap c2hsReal) $ do
        st <- managedState
        t' <- managedTensor t
        liftIO $ do
          let sz = fromIntegral (product ds)
          -- a strong dose of paranoia
          tmp <- FM.mallocArray sz
          creals <- Sig.c_data st t'
          FM.copyArray tmp creals sz
          FM.peekArray sz tmp
{-# NOINLINE tensordata #-}

-- | Set the storage of a tensor, referencing any number of dimensions of storage
setStorageNd_
  :: Dynamic       -- ^ tensor to mutate, inplace
  -> Storage       -- ^ storage to set
  -> StorageOffset -- ^ offset of the storage to start from
  -> Word          -- ^ dimension... to operate over? to start from? (TODO: allow for "unset" dimension)
  -> [Size]        -- ^ sizes to use with the storage
  -> [Stride]      -- ^ strides to use with the storage
  -> IO ()
setStorageNd_ t s a b hsc hsd = withLift $ Sig.c_setStorageNd
  <$> managedState
  <*> managedTensor t
  <*> managed (withForeignPtr (Sig.cstorage s))
  <*> pure (fromIntegral a)
  <*> pure (fromIntegral b)
  <*> liftIO (FM.newArray (coerce hsc :: [CLLong]))
  <*> liftIO (FM.newArray (coerce hsd :: [CLLong]))
{-# WARNING setStorageNd_ "mutating a tensor's storage can make your program unsafe. You are warned." #-}

-- | FIXME: doublecheck what this does.
_expandNd  :: NonEmpty Dynamic -> NonEmpty Dynamic -> Int -> IO ()
_expandNd (rets@(s:|_)) ops i = runManaged $ do
  st    <- managedState
  rets' <- mngNonEmpty rets
  ops'  <- mngNonEmpty ops
  liftIO $ Sig.c_expandNd st rets' ops' (fromIntegral i)
 where
  mngNonEmpty :: NonEmpty Dynamic -> Managed (Ptr (Ptr CTensor))
  mngNonEmpty = mapM toMPtr . NE.toList >=> mWithArray

  mWithArray :: [Ptr a] -> Managed (Ptr (Ptr a))
  mWithArray as = managed (FM.withArray as)

  toMPtr :: Dynamic -> Managed (Ptr CTensor)
  toMPtr d = managed (withForeignPtr (Sig.ctensor d))


