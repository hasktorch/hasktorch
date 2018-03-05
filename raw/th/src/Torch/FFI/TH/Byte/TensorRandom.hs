{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Byte.TensorRandom
  ( c_random
  , c_clampedRandom
  , c_cappedRandom
  , c_geometric
  , c_bernoulli
  , c_bernoulli_FloatTensor
  , c_bernoulli_DoubleTensor
  , c_getRNGState
  , c_setRNGState
  , p_random
  , p_clampedRandom
  , p_cappedRandom
  , p_geometric
  , p_bernoulli
  , p_bernoulli_FloatTensor
  , p_bernoulli_DoubleTensor
  , p_getRNGState
  , p_setRNGState
  ) where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_random :  self _generator -> void
foreign import ccall "THTensorRandom.h c_THTensorByte_random"
  c_random :: Ptr (CTHByteTensor) -> Ptr (CTHGenerator) -> IO (())

-- | c_clampedRandom :  self _generator min max -> void
foreign import ccall "THTensorRandom.h c_THTensorByte_clampedRandom"
  c_clampedRandom :: Ptr (CTHByteTensor) -> Ptr (CTHGenerator) -> CLLong -> CLLong -> IO (())

-- | c_cappedRandom :  self _generator max -> void
foreign import ccall "THTensorRandom.h c_THTensorByte_cappedRandom"
  c_cappedRandom :: Ptr (CTHByteTensor) -> Ptr (CTHGenerator) -> CLLong -> IO (())

-- | c_geometric :  self _generator p -> void
foreign import ccall "THTensorRandom.h c_THTensorByte_geometric"
  c_geometric :: Ptr (CTHByteTensor) -> Ptr (CTHGenerator) -> CDouble -> IO (())

-- | c_bernoulli :  self _generator p -> void
foreign import ccall "THTensorRandom.h c_THTensorByte_bernoulli"
  c_bernoulli :: Ptr (CTHByteTensor) -> Ptr (CTHGenerator) -> CDouble -> IO (())

-- | c_bernoulli_FloatTensor :  self _generator p -> void
foreign import ccall "THTensorRandom.h c_THTensorByte_bernoulli_FloatTensor"
  c_bernoulli_FloatTensor :: Ptr (CTHByteTensor) -> Ptr (CTHGenerator) -> Ptr (CTHFloatTensor) -> IO (())

-- | c_bernoulli_DoubleTensor :  self _generator p -> void
foreign import ccall "THTensorRandom.h c_THTensorByte_bernoulli_DoubleTensor"
  c_bernoulli_DoubleTensor :: Ptr (CTHByteTensor) -> Ptr (CTHGenerator) -> Ptr (CTHDoubleTensor) -> IO (())

-- | c_getRNGState :  _generator self -> void
foreign import ccall "THTensorRandom.h c_THTensorByte_getRNGState"
  c_getRNGState :: Ptr (CTHGenerator) -> Ptr (CTHByteTensor) -> IO (())

-- | c_setRNGState :  _generator self -> void
foreign import ccall "THTensorRandom.h c_THTensorByte_setRNGState"
  c_setRNGState :: Ptr (CTHGenerator) -> Ptr (CTHByteTensor) -> IO (())

-- | p_random : Pointer to function : self _generator -> void
foreign import ccall "THTensorRandom.h &p_THTensorByte_random"
  p_random :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHGenerator) -> IO (()))

-- | p_clampedRandom : Pointer to function : self _generator min max -> void
foreign import ccall "THTensorRandom.h &p_THTensorByte_clampedRandom"
  p_clampedRandom :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHGenerator) -> CLLong -> CLLong -> IO (()))

-- | p_cappedRandom : Pointer to function : self _generator max -> void
foreign import ccall "THTensorRandom.h &p_THTensorByte_cappedRandom"
  p_cappedRandom :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHGenerator) -> CLLong -> IO (()))

-- | p_geometric : Pointer to function : self _generator p -> void
foreign import ccall "THTensorRandom.h &p_THTensorByte_geometric"
  p_geometric :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHGenerator) -> CDouble -> IO (()))

-- | p_bernoulli : Pointer to function : self _generator p -> void
foreign import ccall "THTensorRandom.h &p_THTensorByte_bernoulli"
  p_bernoulli :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHGenerator) -> CDouble -> IO (()))

-- | p_bernoulli_FloatTensor : Pointer to function : self _generator p -> void
foreign import ccall "THTensorRandom.h &p_THTensorByte_bernoulli_FloatTensor"
  p_bernoulli_FloatTensor :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHGenerator) -> Ptr (CTHFloatTensor) -> IO (()))

-- | p_bernoulli_DoubleTensor : Pointer to function : self _generator p -> void
foreign import ccall "THTensorRandom.h &p_THTensorByte_bernoulli_DoubleTensor"
  p_bernoulli_DoubleTensor :: FunPtr (Ptr (CTHByteTensor) -> Ptr (CTHGenerator) -> Ptr (CTHDoubleTensor) -> IO (()))

-- | p_getRNGState : Pointer to function : _generator self -> void
foreign import ccall "THTensorRandom.h &p_THTensorByte_getRNGState"
  p_getRNGState :: FunPtr (Ptr (CTHGenerator) -> Ptr (CTHByteTensor) -> IO (()))

-- | p_setRNGState : Pointer to function : _generator self -> void
foreign import ccall "THTensorRandom.h &p_THTensorByte_setRNGState"
  p_setRNGState :: FunPtr (Ptr (CTHGenerator) -> Ptr (CTHByteTensor) -> IO (()))