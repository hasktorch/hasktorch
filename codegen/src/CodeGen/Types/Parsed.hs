{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveDataTypeable #-}
module CodeGen.Types.Parsed where

import CodeGen.Prelude
import CodeGen.Types.CLI
import Control.Monad
import Data.Data
import Data.Typeable
import qualified Data.HashSet as HS


-- ----------------------------------------
-- Parsed types
-- ----------------------------------------
data Parsable
  = Ptr Parsable
  | TenType TenType
  -- | NNType NNType
  | CType CType
  deriving (Eq, Show, Generic, Hashable)

data CType
  = CBool
  | CVoid
  | CPtrdiff
  | CFloat
  | CDouble
  | CLong

  | CUInt64
  | CUInt32
  | CUInt16
  | CUInt8

  | CInt64
  | CInt32
  | CInt16
  | CInt8

  | CInt -- must come _after_ all the other int types

  | CSize
  | CChar
  | CShort
  deriving (Eq, Show, Generic, Hashable, Bounded, Enum)

newtype TenType = Pair { unTenType :: (RawTenType, LibType) }
  deriving (Eq, Show, Generic, Hashable)

data RawTenType
  = Tensor
  | ByteTensor
  | CharTensor
  | ShortTensor
  | IntTensor
  | LongTensor
  | FloatTensor
  | DoubleTensor
  | HalfTensor

  | Storage
  | ByteStorage
  | CharStorage
  | ShortStorage
  | IntStorage
  | LongStorage
  | FloatStorage
  | DoubleStorage
  | HalfStorage

  | DescBuff
  | Generator
  | Allocator
  | File
  | Half
  | State

  -- NN types
  | IndexTensor
  | IntegerTensor

  -- real and accreal are parameterized occasionally differ by library, but I don't think this exists to date.
  | Real
  | AccReal

  -- FIXME: I don't think we need to enable THCThreadLocal, yet. But we would need to include a
  -- wrapper of ThreadId from the pthread package: https://hackage.haskell.org/package/pthread-0.2.0
  -- | ThreadLocal  -- THC-specific

  -- FIXME: while we can add this to codegen now, we need access to cudaStream_t in cuda_runtime_api
  | Stream  -- THC-specific
  deriving (Eq, Show, Generic, Hashable, Bounded, Enum)


isConcreteCudaPrefixed :: TenType -> Bool
isConcreteCudaPrefixed (Pair (t, lib)) = (lib == THC || lib == THCUNN) && t `HS.member` HS.fromList
  [ ByteTensor
  , CharTensor
  , ShortTensor
  , IntTensor
  , LongTensor
  , FloatTensor
  , DoubleTensor
  , HalfTensor
  -- , IndexTensor
  ]

allTenTypes :: [TenType]
allTenTypes = Pair <$> ((,) <$> [minBound..maxBound] <*> [minBound..maxBound])

-- data NNType
--   = IndexTensor
--   | IntegerTensor
--   deriving (Eq, Show, Generic, Hashable, Bounded, Enum)


data Arg = Arg
  { argType :: Parsable
  , argName :: Text
  } deriving (Eq, Show, Generic, Hashable)


data Function = Function
  { funPrefix :: Maybe (LibType, Text)
  , funName   :: Text
  , funArgs   :: [Arg]
  , funReturn :: Parsable
  } deriving (Eq, Show, Generic, Hashable)

type Parser = Parsec Void String


