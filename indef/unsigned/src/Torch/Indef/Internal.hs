{-# LANGUAGE TypeFamilies #-}
module Torch.Indef.Internal
  ( ptrArray2hs
  , withDynTensor
  , with2DynTensors
  , with3DynTensors
  , with4DynTensors
  , with5DynTensors
  , _withDynTensor
  , _with2DynTensors
  , _with3DynTensors
  , _with4DynTensors
  , _with5DynTensors
  , module Sig
  ) where

import Foreign
import GHC.ForeignPtr (ForeignPtr)
import qualified Foreign.Marshal.Array as FM

import Torch.Sig.Types as Sig
import qualified Torch.Class.Types as TypeFams


ptrArray2hs :: (Ptr a -> IO (Ptr CReal)) -> (Ptr a -> IO Int) -> ForeignPtr a -> IO [HsReal]
ptrArray2hs updPtrArray toSize fp = do
  sz <- withForeignPtr fp toSize
  creals <- withForeignPtr fp updPtrArray
  (fmap.fmap) c2hsReal (FM.peekArray sz creals)

-- ========================================================================= --

_withDynTensor :: DynTensor -> (Ptr CTensor -> IO x) -> IO x
_withDynTensor t0 fn = withForeignPtr (tensor t0) fn

withDynTensor = flip _withDynTensor

_with2DynTensors
  :: DynTensor -> DynTensor
  -> (Ptr CTensor -> Ptr CTensor -> IO x)
  -> IO x
_with2DynTensors t0 t1 fn =
  _withDynTensor t0 $ \t0' ->
    _withDynTensor t1 $ \t1' ->
      fn t0' t1'

with2DynTensors fn t0 t1 = _with2DynTensors t0 t1 fn

_with3DynTensors
  :: DynTensor -> DynTensor -> DynTensor
  -> (Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> IO x)
  -> IO x
_with3DynTensors t0 t1 t2 fn =
  _with2DynTensors t0 t1 $ \t0' t1' ->
    _withDynTensor t2 $ \t2' ->
      fn t0' t1' t2'

with3DynTensors fn t0 t1 t2 = _with3DynTensors t0 t1 t2 fn

_with4DynTensors
  :: DynTensor -> DynTensor -> DynTensor -> DynTensor
  -> (Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> IO x)
  -> IO x
_with4DynTensors t0 t1 t2 t3 fn =
  _with3DynTensors t0 t1 t2 $ \t0' t1' t2' ->
    _withDynTensor t3 $ \t3' ->
      fn t0' t1' t2' t3'

with4DynTensors fn t0 t1 t2 t3 = _with4DynTensors t0 t1 t2 t3 fn

_with5DynTensors
  :: DynTensor -> DynTensor -> DynTensor -> DynTensor -> DynTensor
  -> (Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> IO x)
  -> IO x
_with5DynTensors t0 t1 t2 t3 t4 fn =
  _with4DynTensors t0 t1 t2 t3 $ \t0' t1' t2' t3' ->
    _withDynTensor t4 $ \t4' ->
      fn t0' t1' t2' t3' t4'

with5DynTensors fn t0 t1 t2 t3 t4 = _with5DynTensors t0 t1 t2 t3 t4 fn

-- ========================================================================= --
-- TODO: bring back the managed versions of the above
--
-- withManaged4
--   :: (THForeignRef t)
--   => (Ptr (THForeignType t) -> Ptr (THForeignType t) -> Ptr (THForeignType t) -> Ptr (THForeignType t) -> IO ())
--   -> t -> t -> t -> t -> IO ()
-- withManaged4 fn resA resB a b = runManaged $ do
--   resBRaw <- managed (withForeignPtr (getForeign resB))
--   resARaw <- managed (withForeignPtr (getForeign resA))
--   bRaw <- managed (withForeignPtr (getForeign b))
--   aRaw <- managed (withForeignPtr (getForeign a))
--   liftIO (fn resBRaw resARaw bRaw aRaw)
--
-- withManaged3
--   :: (THForeignRef t)
--   => (Ptr (THForeignType t) -> Ptr (THForeignType t) -> Ptr (THForeignType t) -> IO ())
--   -> t -> t -> t -> IO ()
-- withManaged3 fn a b c = runManaged $ do
--   a' <- managed (withForeignPtr (getForeign a))
--   b' <- managed (withForeignPtr (getForeign b))
--   c' <- managed (withForeignPtr (getForeign c))
--   liftIO (fn a' b' c')
--
-- withManaged2
--   :: (THForeignRef t)
--   => (Ptr (THForeignType t) -> Ptr (THForeignType t) -> IO ())
--   -> t -> t -> IO ()
-- withManaged2 fn resA a = runManaged $ do
--   resARaw <- managed (withForeignPtr (getForeign resA))
--   aRaw <- managed (withForeignPtr (getForeign a))
--   liftIO (fn resARaw aRaw)
--
-- withManaged1
--   :: (THForeignRef t)
--   => (Ptr (THForeignType t) -> IO ())
--   -> t -> IO ()
-- withManaged1 fn a = runManaged $ do
--   a' <- managed (withForeignPtr (getForeign a))
--   liftIO (fn a')


