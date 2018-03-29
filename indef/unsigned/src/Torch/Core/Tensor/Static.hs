-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Core.Tensor.Static
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
--
-- Tensors with dimensional phantom types.
--
-- Be aware of https://ghc.haskell.org/trac/ghc/wiki/Roles but since Dynamic
-- and static tensors are the same (minus the dimension operators in the
-- phantom type), I (@stites) don't think we need to be too concerned.
-------------------------------------------------------------------------------
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
module Torch.Core.Tensor.Static
  ( Tensor
  , Sig.dynamic
  , tensor
  , Dynamic.IsTensor(..)
  , Dynamic.TensorCopy(..)
  , Dynamic.TensorConv(..)
  , Dynamic.TensorMath(..)
  , module Random
  ) where

import SigTypes hiding (tensor)
import Data.Coerce (coerce)
import Foreign.ForeignPtr (ForeignPtr)

import GHC.TypeLits (Nat)
import Torch.Core.Types (Storage)
import Torch.Class.C.Tensor.Static (IsStatic(..))
import qualified SigTypes as Sig
import qualified Torch.Class.C.Internal as TypeFamilies
import qualified Torch.Core.Tensor.Dynamic as Dynamic
import Torch.Core.Tensor.Static.Random ()
import Torch.Class.C.Tensor.Random as Random

-- instance Dynamic.IsTensor (Tensor (ds :: [Nat]))
-- instance Dynamic.TensorCopy (Tensor (ds :: [Nat]))
-- instance Dynamic.TensorConv (Tensor (ds :: [Nat]))
-- instance Dynamic.TensorMath (Tensor (ds :: [Nat]))
-- instance Dynamic.TensorRandom (Tensor (ds :: [Nat]))

tensor :: Tensor (ds :: [Nat]) -> ForeignPtr CTensor
tensor = Sig.tensor . Sig.dynamic

type instance TypeFamilies.HsReal    (Tensor (ds::[Nat])) = HsReal
type instance TypeFamilies.HsAccReal (Tensor (ds::[Nat])) = HsAccReal
type instance TypeFamilies.HsStorage (Tensor (ds::[Nat])) = Storage
type instance TypeFamilies.AsDynamic (Tensor (ds::[Nat])) = Dynamic.Tensor

instance IsStatic (Tensor ds) where
  asDynamic = Sig.dynamic
  asStatic = Sig.asStatic

