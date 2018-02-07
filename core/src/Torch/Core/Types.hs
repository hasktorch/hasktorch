{-# LANGUAGE TypeFamilies #-}
module Torch.Core.Types
  ( Storage(..)
  , Tensor(..)

  , ptrArray2hs

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


