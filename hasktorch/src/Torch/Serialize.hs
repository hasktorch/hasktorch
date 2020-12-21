{-# LANGUAGE FlexibleContexts #-}

module Torch.Serialize where

import Control.Exception.Safe
  ( SomeException (..),
    throwIO,
    try,
  )
import Control.Monad (when)
import qualified Data.ByteString as BS
import qualified Data.ByteString.Internal as BSI
import qualified Foreign.ForeignPtr as F
import qualified Foreign.Ptr as F
import System.IO
import Torch.Autograd
import Torch.DType
import Torch.Functional
import Torch.Internal.Cast
import qualified Torch.Internal.Managed.Serialize as S
import Torch.NN
import Torch.Tensor
import Torch.Traversable

save ::
  -- | inputs
  [Tensor] ->
  -- | file
  FilePath ->
  -- | output
  IO ()
save = cast2 S.save

load ::
  -- | file
  FilePath ->
  -- | output
  IO [Tensor]
load = cast1 S.load

saveParams ::
  GTraversable Parameter f =>
  -- | model
  f ->
  -- | filepath
  FilePath ->
  -- | output
  IO ()
saveParams model filePath = do
  let params = map toDependent $ flattenParameters model
  save params filePath

loadParams ::
  GTraversable Parameter b =>
  -- | model
  b ->
  -- | filepath
  FilePath ->
  -- | output
  IO b
loadParams model filePath = do
  tensors <- load filePath
  let params = map IndependentTensor tensors
  pure $ replaceParameters model params

class RawFile a where
  loadBinary :: Handle -> a -> IO a
  saveBinary :: Handle -> a -> IO ()

instance RawFile Tensor where
  loadBinary handle tensor = do
    let len = (byteLength (dtype tensor)) * product (shape tensor)
    v <- BS.hGet handle len
    t <- clone tensor
    withTensor t $ \ptr1 -> do
      let (BSI.PS fptr _ len') = v
      when (len' < len) $ do
        throwIO $ userError $ "Read data's size is less than input tensor's one(" <> show len <> ")."
      F.withForeignPtr fptr $ \ptr2 -> do
        BSI.memcpy (F.castPtr ptr1) (F.castPtr ptr2) (Prelude.min len len')
        return t

  saveBinary handle tensor = do
    let len = (byteLength (dtype tensor)) * product (shape tensor)
    t <- clone tensor
    withTensor tensor $ \ptr1 -> do
      hPutBuf handle (F.castPtr ptr1) len
