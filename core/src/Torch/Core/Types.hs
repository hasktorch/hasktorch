{-# LANGUAGE TypeFamilies #-}
module Torch.Core.Types
  ( Storage(..)
  , Tensor(..)
  , module Sig
  ) where

import GHC.ForeignPtr (ForeignPtr)

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


