-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Class.Types
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
--
-------------------------------------------------------------------------------
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE TypeSynonymInstances #-}
{-# LANGUAGE ConstraintKinds #-}
module Torch.Class.Types where

import Foreign
import Torch.Types.TH (C'THState, State)
import GHC.Int (Int64)
import Control.Monad.Trans.Reader (ReaderT)
import Control.Monad.Reader.Class (MonadReader)
import Control.Monad.IO.Class (MonadIO)
import Control.Exception.Safe (MonadThrow)
import GHC.TypeLits
import Torch.Dimensions

type family HsReal t
type family HsAccReal t
type family HsStorage t
type family AsDynamic t
type family Allocator t
type family Generator t
type family IndexTensor t (d :: [Nat])
type family IndexDynamic t
type family IndexStorage t
type family MaskTensor t (d :: [Nat])
type family MaskDynamic t
type family DescBuff t
type family DimReal t

type family SizesStorage t
type family StridesStorage t

impossible = error

type (Dimensions2 d d') = (Dimensions d, Dimensions d')
type (Dimensions3 d d' d'' ) = (Dimensions2 d d', Dimensions d'')
type (Dimensions4 d d' d'' d''') = (Dimensions2 d d', Dimensions2 d'' d''')
type (Dimensions5 d d' d'' d''' d'''') = (Dimensions4 d d' d'' d''', Dimensions d'''')

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

-- haskell representation of a CInt which determines whether or not to return dimensions
newtype KeepDim = KeepDim { keepIt :: Bool }
  deriving (Bounded, Enum, Eq, Ord, Read, Show)

-- don't bind the @i@ in case there are some differences between THC and TH
fromKeepDim :: Integral i => Maybe KeepDim -> i
fromKeepDim = maybe 0 (fromIntegral . fromEnum)

-- smart constructors for keepdim since we don't get inference for free like Num
keep,  ignore :: KeepDim
(keep, ignore) = (KeepDim True, KeepDim False)

data SortOrder = Ascending | Descending
  deriving (Eq, Show, Ord, Enum, Bounded)

-- for working with a unified conversion of static and dynamic types
class IsStatic t where
  asDynamic :: t -> AsDynamic t
  asStatic :: AsDynamic t -> t

