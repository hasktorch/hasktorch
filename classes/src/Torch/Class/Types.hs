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
module Torch.Class.Types where

import Foreign
import Torch.Types.TH (C'THState, State)
import GHC.Int (Int64)
import Control.Monad.Trans.Reader (ReaderT)
import Control.Monad.Reader.Class (MonadReader)
import Control.Monad.IO.Class (MonadIO)
import Control.Exception.Safe (MonadThrow)

type family HsReal t
type family HsAccReal t
type family HsStorage t
type family AsDynamic t
type family Allocator t
type family Generator t
type family IndexTensor t
type family IndexStorage t
type family MaskTensor t
type family DescBuff t
type family DimReal t

type SizesStorage t = IndexStorage t
type StridesStorage t = IndexStorage t

class MonadIO (m s) => HasState m s where
  getTHState :: m s s

-- come up with an escape hatch for the statefulness
-- instance MonadIO (ReaderT C'THState IO) => HasState (ReaderT C'THState IO) C'THState where
--   getTHState = pure ()
-- 
-- liftIO :: IO x -> (ReaderT C'THState IO x)

-- instance HasState ReadIO State where
--   getTHState = pure (impossible "a TH state will never be evaluated")

impossible = error

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

-- for working with a unified conversion of static and dynamic types
class IsStatic t where
  asDynamic :: t -> AsDynamic t
  asStatic :: AsDynamic t -> t

