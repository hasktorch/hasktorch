-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Class.Internal
-- Copyright :  (c) Sam Stites 2017
-- License   :  MIT
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
--
-- Should be "Torch.Class.Types"
-------------------------------------------------------------------------------
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
module Torch.Class.Internal where

import GHC.Int (Int64)
import qualified Torch.Types.TH.Long as Long (Tensor, Storage)

type family HsReal t
type family HsAccReal t
type family HsStorage t
type family AsDynamic t

-- Maybe better served as a newtype of Foreign.C.Types.CLLong
newtype Stride = Stride Int64
  deriving (Bounded, Enum, Eq, Integral, Num, Ord, Read, Real, Show)

-- Maybe better served as a newtype of Foreign.C.Types.CLLong
newtype Size = Size Int64
  deriving (Bounded, Enum, Eq, Integral, Num, Ord, Read, Real, Show)

-- Maybe better served as a newtype of Foreign.C.Types.CPtrDiff
newtype StorageOffset = StorageOffset Int64
  deriving (Bounded, Enum, Eq, Integral, Num, Ord, Read, Real, Show)

-- Maybe better served as a newtype of Foreign.C.Types.CLong
newtype Step = Step Int64
  deriving (Bounded, Enum, Eq, Integral, Num, Ord, Read, Real, Show)

type SizesStorage = Long.Storage
type StridesStorage = Long.Storage
