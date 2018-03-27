-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Indef.Dynamic.Types
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
-------------------------------------------------------------------------------
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE InstanceSigs #-}
module Torch.Indef.Types
  ( Torch
  , module X
  ) where

import Torch.Class.Types
import qualified Torch.Sig.Types as Sig
import qualified Torch.Sig.State as Sig

import Torch.Indef.Internal as X

type Torch = Torch' Sig.State

-------------------------------------------------------------------------------
-- Storage type family instances

type instance Allocator    Sig.Storage = Sig.HsAllocator
type instance Generator    Sig.Storage = Sig.HsGenerator
type instance DescBuff     Sig.Storage = Sig.HsDescBuff

type instance HsReal       Sig.Storage = Sig.HsReal
type instance HsAccReal    Sig.Storage = Sig.HsAccReal

-------------------------------------------------------------------------------
-- Dynamic type family instances

type instance AsDynamic    Sig.DynTensor = Sig.DynTensor
type instance HsStorage    Sig.DynTensor = Sig.Storage
type instance IndexTensor  Sig.DynTensor = Sig.HsIndexTensor
type instance IndexStorage Sig.DynTensor = Sig.HsIndexStorage
type instance MaskTensor   Sig.DynTensor = Sig.HsMaskTensor

type instance Allocator    Sig.DynTensor = Sig.HsAllocator
type instance Generator    Sig.DynTensor = Sig.HsGenerator
type instance DescBuff     Sig.DynTensor = Sig.HsDescBuff

type instance HsReal       Sig.DynTensor = Sig.HsReal
type instance HsAccReal    Sig.DynTensor = Sig.HsAccReal


-------------------------------------------------------------------------------
-- Static type family instances

type instance AsDynamic    (Sig.Tensor d) = Sig.DynTensor
type instance HsStorage    (Sig.Tensor d) = Sig.Storage

type instance IndexTensor  (Sig.Tensor d) = Sig.HsIndexTensor
type instance IndexStorage (Sig.Tensor d) = Sig.HsIndexStorage
type instance MaskTensor   (Sig.Tensor d) = Sig.HsMaskTensor

type instance Allocator    (Sig.Tensor d) = Sig.HsAllocator
type instance Generator    (Sig.Tensor d) = Sig.HsGenerator
type instance DescBuff     (Sig.Tensor d) = Sig.HsDescBuff

type instance HsReal       (Sig.Tensor d) = Sig.HsReal
type instance HsAccReal    (Sig.Tensor d) = Sig.HsAccReal

instance IsStatic (Sig.Tensor d) where
  asDynamic = Sig.dynamic
  asStatic = Sig.asStatic

