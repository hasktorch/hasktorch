{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Long.TensorRandom
  ( c_random
  , c_clampedRandom
  , c_cappedRandom
  , c_geometric
  , c_bernoulli
  , c_bernoulli_FloatTensor
  , c_bernoulli_DoubleTensor
  , p_random
  , p_clampedRandom
  , p_cappedRandom
  , p_geometric
  , p_bernoulli
  , p_bernoulli_FloatTensor
  , p_bernoulli_DoubleTensor
  ) where

import Foreign
import Foreign.C.Types
import THTypes
import Data.Word
import Data.Int

-- | c_random :  self _generator -> void
foreign import ccall "THTensorRandom.h c_THTensorLong_random"
  c_random :: Ptr (CTHLongTensor) -> Ptr (CTHGenerator) -> IO (())

-- | c_clampedRandom :  self _generator min max -> void
foreign import ccall "THTensorRandom.h c_THTensorLong_clampedRandom"
  c_clampedRandom :: Ptr (CTHLongTensor) -> Ptr (CTHGenerator) -> CLLong -> CLLong -> IO (())

-- | c_cappedRandom :  self _generator max -> void
foreign import ccall "THTensorRandom.h c_THTensorLong_cappedRandom"
  c_cappedRandom :: Ptr (CTHLongTensor) -> Ptr (CTHGenerator) -> CLLong -> IO (())

-- | c_geometric :  self _generator p -> void
foreign import ccall "THTensorRandom.h c_THTensorLong_geometric"
  c_geometric :: Ptr (CTHLongTensor) -> Ptr (CTHGenerator) -> CDouble -> IO (())

-- | c_bernoulli :  self _generator p -> void
foreign import ccall "THTensorRandom.h c_THTensorLong_bernoulli"
  c_bernoulli :: Ptr (CTHLongTensor) -> Ptr (CTHGenerator) -> CDouble -> IO (())

-- | c_bernoulli_FloatTensor :  self _generator p -> void
foreign import ccall "THTensorRandom.h c_THTensorLong_bernoulli_FloatTensor"
  c_bernoulli_FloatTensor :: Ptr (CTHLongTensor) -> Ptr (CTHGenerator) -> Ptr (CTHFloatTensor) -> IO (())

-- | c_bernoulli_DoubleTensor :  self _generator p -> void
foreign import ccall "THTensorRandom.h c_THTensorLong_bernoulli_DoubleTensor"
  c_bernoulli_DoubleTensor :: Ptr (CTHLongTensor) -> Ptr (CTHGenerator) -> Ptr (CTHDoubleTensor) -> IO (())

-- | p_random : Pointer to function : self _generator -> void
foreign import ccall "THTensorRandom.h &p_THTensorLong_random"
  p_random :: FunPtr (Ptr (CTHLongTensor) -> Ptr (CTHGenerator) -> IO (()))

-- | p_clampedRandom : Pointer to function : self _generator min max -> void
foreign import ccall "THTensorRandom.h &p_THTensorLong_clampedRandom"
  p_clampedRandom :: FunPtr (Ptr (CTHLongTensor) -> Ptr (CTHGenerator) -> CLLong -> CLLong -> IO (()))

-- | p_cappedRandom : Pointer to function : self _generator max -> void
foreign import ccall "THTensorRandom.h &p_THTensorLong_cappedRandom"
  p_cappedRandom :: FunPtr (Ptr (CTHLongTensor) -> Ptr (CTHGenerator) -> CLLong -> IO (()))

-- | p_geometric : Pointer to function : self _generator p -> void
foreign import ccall "THTensorRandom.h &p_THTensorLong_geometric"
  p_geometric :: FunPtr (Ptr (CTHLongTensor) -> Ptr (CTHGenerator) -> CDouble -> IO (()))

-- | p_bernoulli : Pointer to function : self _generator p -> void
foreign import ccall "THTensorRandom.h &p_THTensorLong_bernoulli"
  p_bernoulli :: FunPtr (Ptr (CTHLongTensor) -> Ptr (CTHGenerator) -> CDouble -> IO (()))

-- | p_bernoulli_FloatTensor : Pointer to function : self _generator p -> void
foreign import ccall "THTensorRandom.h &p_THTensorLong_bernoulli_FloatTensor"
  p_bernoulli_FloatTensor :: FunPtr (Ptr (CTHLongTensor) -> Ptr (CTHGenerator) -> Ptr (CTHFloatTensor) -> IO (()))

-- | p_bernoulli_DoubleTensor : Pointer to function : self _generator p -> void
foreign import ccall "THTensorRandom.h &p_THTensorLong_bernoulli_DoubleTensor"
  p_bernoulli_DoubleTensor :: FunPtr (Ptr (CTHLongTensor) -> Ptr (CTHGenerator) -> Ptr (CTHDoubleTensor) -> IO (()))