{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Int.TensorRandom
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
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_random :  self _generator -> void
foreign import ccall "THTensorRandom.h c_THTensorInt_random"
  c_random :: Ptr (CTHIntTensor) -> Ptr (CTHGenerator) -> IO (())

-- | c_clampedRandom :  self _generator min max -> void
foreign import ccall "THTensorRandom.h c_THTensorInt_clampedRandom"
  c_clampedRandom :: Ptr (CTHIntTensor) -> Ptr (CTHGenerator) -> CLLong -> CLLong -> IO (())

-- | c_cappedRandom :  self _generator max -> void
foreign import ccall "THTensorRandom.h c_THTensorInt_cappedRandom"
  c_cappedRandom :: Ptr (CTHIntTensor) -> Ptr (CTHGenerator) -> CLLong -> IO (())

-- | c_geometric :  self _generator p -> void
foreign import ccall "THTensorRandom.h c_THTensorInt_geometric"
  c_geometric :: Ptr (CTHIntTensor) -> Ptr (CTHGenerator) -> CDouble -> IO (())

-- | c_bernoulli :  self _generator p -> void
foreign import ccall "THTensorRandom.h c_THTensorInt_bernoulli"
  c_bernoulli :: Ptr (CTHIntTensor) -> Ptr (CTHGenerator) -> CDouble -> IO (())

-- | c_bernoulli_FloatTensor :  self _generator p -> void
foreign import ccall "THTensorRandom.h c_THTensorInt_bernoulli_FloatTensor"
  c_bernoulli_FloatTensor :: Ptr (CTHIntTensor) -> Ptr (CTHGenerator) -> Ptr (CTHFloatTensor) -> IO (())

-- | c_bernoulli_DoubleTensor :  self _generator p -> void
foreign import ccall "THTensorRandom.h c_THTensorInt_bernoulli_DoubleTensor"
  c_bernoulli_DoubleTensor :: Ptr (CTHIntTensor) -> Ptr (CTHGenerator) -> Ptr (CTHDoubleTensor) -> IO (())

-- | p_random : Pointer to function : self _generator -> void
foreign import ccall "THTensorRandom.h &p_THTensorInt_random"
  p_random :: FunPtr (Ptr (CTHIntTensor) -> Ptr (CTHGenerator) -> IO (()))

-- | p_clampedRandom : Pointer to function : self _generator min max -> void
foreign import ccall "THTensorRandom.h &p_THTensorInt_clampedRandom"
  p_clampedRandom :: FunPtr (Ptr (CTHIntTensor) -> Ptr (CTHGenerator) -> CLLong -> CLLong -> IO (()))

-- | p_cappedRandom : Pointer to function : self _generator max -> void
foreign import ccall "THTensorRandom.h &p_THTensorInt_cappedRandom"
  p_cappedRandom :: FunPtr (Ptr (CTHIntTensor) -> Ptr (CTHGenerator) -> CLLong -> IO (()))

-- | p_geometric : Pointer to function : self _generator p -> void
foreign import ccall "THTensorRandom.h &p_THTensorInt_geometric"
  p_geometric :: FunPtr (Ptr (CTHIntTensor) -> Ptr (CTHGenerator) -> CDouble -> IO (()))

-- | p_bernoulli : Pointer to function : self _generator p -> void
foreign import ccall "THTensorRandom.h &p_THTensorInt_bernoulli"
  p_bernoulli :: FunPtr (Ptr (CTHIntTensor) -> Ptr (CTHGenerator) -> CDouble -> IO (()))

-- | p_bernoulli_FloatTensor : Pointer to function : self _generator p -> void
foreign import ccall "THTensorRandom.h &p_THTensorInt_bernoulli_FloatTensor"
  p_bernoulli_FloatTensor :: FunPtr (Ptr (CTHIntTensor) -> Ptr (CTHGenerator) -> Ptr (CTHFloatTensor) -> IO (()))

-- | p_bernoulli_DoubleTensor : Pointer to function : self _generator p -> void
foreign import ccall "THTensorRandom.h &p_THTensorInt_bernoulli_DoubleTensor"
  p_bernoulli_DoubleTensor :: FunPtr (Ptr (CTHIntTensor) -> Ptr (CTHGenerator) -> Ptr (CTHDoubleTensor) -> IO (()))