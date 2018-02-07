{-# LANGUAGE TypeFamilies #-}
module Torch.Core.Types
  ( Storage(..)
  , Tensor(..)

  , ptrArray2hs
  , withTensor
  , with2Tensors
  , with3Tensors
  , with4Tensors
  , with5Tensors
  , _withTensor
  , _with2Tensors
  , _with3Tensors
  , _with4Tensors
  , _with5Tensors


  , module Sig
  ) where

import Foreign
import GHC.ForeignPtr (ForeignPtr)
import qualified Foreign.Marshal.Array as FM

import SigTypes as Sig
import qualified Torch.Class.Internal as TypeFamilies

newtype Storage = Storage { storage :: ForeignPtr Sig.CStorage }
  deriving (Eq, Show)

newtype Tensor = Tensor { tensor :: ForeignPtr Sig.CTensor }
  deriving (Show, Eq)

type instance TypeFamilies.HsReal    Tensor  = Sig.HsReal
type instance TypeFamilies.HsReal    Storage = Sig.HsReal

type instance TypeFamilies.HsAccReal Tensor  = Sig.HsAccReal
type instance TypeFamilies.HsAccReal Storage = Sig.HsAccReal

type instance TypeFamilies.HsStorage Tensor  = Storage


ptrArray2hs :: (Ptr a -> IO (Ptr Sig.CReal)) -> (Ptr a -> IO Int) -> ForeignPtr a -> IO [Sig.HsReal]
ptrArray2hs updPtrArray toSize fp = do
  sz <- withForeignPtr fp toSize
  creals <- withForeignPtr fp updPtrArray
  (fmap.fmap) c2hsReal (FM.peekArray sz creals)

-- ========================================================================= --

_withTensor :: Tensor -> (Ptr CTensor -> IO x) -> IO x
_withTensor t0 fn = withForeignPtr (tensor t0) fn

withTensor = flip _withTensor

_with2Tensors
  :: Tensor -> Tensor
  -> (Ptr CTensor -> Ptr CTensor -> IO x)
  -> IO x
_with2Tensors t0 t1 fn =
  _withTensor t0 $ \t0' ->
    _withTensor t1 $ \t1' ->
      fn t0' t1'

with2Tensors fn t0 t1 = _with2Tensors t0 t1 fn

_with3Tensors
  :: Tensor -> Tensor -> Tensor
  -> (Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> IO x)
  -> IO x
_with3Tensors t0 t1 t2 fn =
  _with2Tensors t0 t1 $ \t0' t1' ->
    _withTensor t2 $ \t2' ->
      fn t0' t1' t2'

with3Tensors fn t0 t1 t2 = _with3Tensors t0 t1 t2 fn

_with4Tensors
  :: Tensor -> Tensor -> Tensor -> Tensor
  -> (Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> IO x)
  -> IO x
_with4Tensors t0 t1 t2 t3 fn =
  _with3Tensors t0 t1 t2 $ \t0' t1' t2' ->
    _withTensor t3 $ \t3' ->
      fn t0' t1' t2' t3'

with4Tensors fn t0 t1 t2 t3 = _with4Tensors t0 t1 t2 t3 fn

_with5Tensors
  :: Tensor -> Tensor -> Tensor -> Tensor -> Tensor
  -> (Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> IO x)
  -> IO x
_with5Tensors t0 t1 t2 t3 t4 fn =
  _with4Tensors t0 t1 t2 t3 $ \t0' t1' t2' t3' ->
    _withTensor t4 $ \t4' ->
      fn t0' t1' t2' t3' t4'

with5Tensors fn t0 t1 t2 t3 t4 = _with5Tensors t0 t1 t2 t3 t4 fn

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


