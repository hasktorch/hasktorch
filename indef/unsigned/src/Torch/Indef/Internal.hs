{-# LANGUAGE TypeFamilies #-}
module Torch.Indef.Internal
  ( ptrArray2hs
  , withDynamic
  , with2Dynamics
  , with3Dynamics
  , with4Dynamics
  , with5Dynamics
  , _withDynamic
  , _with2Dynamics
  , _with3Dynamics
  , _with4Dynamics
  , _with5Dynamics
  ) where

import Foreign
import GHC.ForeignPtr (ForeignPtr)
import qualified Foreign.Marshal.Array as FM

import Torch.Sig.Types
import qualified Torch.Class.Types as TypeFams


ptrArray2hs :: (Ptr a -> IO (Ptr CReal)) -> (Ptr a -> IO Int) -> ForeignPtr a -> IO [HsReal]
ptrArray2hs updPtrArray toSize fp = do
  sz <- withForeignPtr fp toSize
  creals <- withForeignPtr fp updPtrArray
  (fmap.fmap) c2hsReal (FM.peekArray sz creals)

-- ========================================================================= --

_withDynamic :: Dynamic -> (Ptr CTensor -> IO x) -> IO x
_withDynamic t0 fn = withForeignPtr (ctensor t0) fn

withDynamic = flip _withDynamic

_with2Dynamics
  :: Dynamic -> Dynamic
  -> (Ptr CTensor -> Ptr CTensor -> IO x)
  -> IO x
_with2Dynamics t0 t1 fn =
  _withDynamic t0 $ \t0' ->
    _withDynamic t1 $ \t1' ->
      fn t0' t1'

with2Dynamics fn t0 t1 = _with2Dynamics t0 t1 fn

_with3Dynamics
  :: Dynamic -> Dynamic -> Dynamic
  -> (Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> IO x)
  -> IO x
_with3Dynamics t0 t1 t2 fn =
  _with2Dynamics t0 t1 $ \t0' t1' ->
    _withDynamic t2 $ \t2' ->
      fn t0' t1' t2'

with3Dynamics fn t0 t1 t2 = _with3Dynamics t0 t1 t2 fn

_with4Dynamics
  :: Dynamic -> Dynamic -> Dynamic -> Dynamic
  -> (Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> IO x)
  -> IO x
_with4Dynamics t0 t1 t2 t3 fn =
  _with3Dynamics t0 t1 t2 $ \t0' t1' t2' ->
    _withDynamic t3 $ \t3' ->
      fn t0' t1' t2' t3'

with4Dynamics fn t0 t1 t2 t3 = _with4Dynamics t0 t1 t2 t3 fn

_with5Dynamics
  :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic
  -> (Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> IO x)
  -> IO x
_with5Dynamics t0 t1 t2 t3 t4 fn =
  _with4Dynamics t0 t1 t2 t3 $ \t0' t1' t2' t3' ->
    _withDynamic t4 $ \t4' ->
      fn t0' t1' t2' t3' t4'

with5Dynamics fn t0 t1 t2 t3 t4 = _with5Dynamics t0 t1 t2 t3 t4 fn

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


