{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Long.TensorRandom
  ( c_random
  , c_clampedRandom
  , c_cappedRandom
  , c_bernoulli
  , c_bernoulli_DoubleTensor
  , c_geometric
  , p_random
  , p_clampedRandom
  , p_cappedRandom
  , p_bernoulli
  , p_bernoulli_DoubleTensor
  , p_geometric
  ) where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_random :  state self -> void
foreign import ccall "THCTensorRandom.h THLongTensor_random"
  c_random :: Ptr (CTHState) -> Ptr (CTHLongTensor) -> IO (())

-- | c_clampedRandom :  state self min max -> void
foreign import ccall "THCTensorRandom.h THLongTensor_clampedRandom"
  c_clampedRandom :: Ptr (CTHState) -> Ptr (CTHLongTensor) -> CLLong -> CLLong -> IO (())

-- | c_cappedRandom :  state self max -> void
foreign import ccall "THCTensorRandom.h THLongTensor_cappedRandom"
  c_cappedRandom :: Ptr (CTHState) -> Ptr (CTHLongTensor) -> CLLong -> IO (())

-- | c_bernoulli :  state self p -> void
foreign import ccall "THCTensorRandom.h THLongTensor_bernoulli"
  c_bernoulli :: Ptr (CTHState) -> Ptr (CTHLongTensor) -> CDouble -> IO (())

-- | c_bernoulli_DoubleTensor :  state self p -> void
foreign import ccall "THCTensorRandom.h THLongTensor_bernoulli_DoubleTensor"
  c_bernoulli_DoubleTensor :: Ptr (CTHState) -> Ptr (CTHLongTensor) -> Ptr (CTHDoubleTensor) -> IO (())

-- | c_geometric :  state self p -> void
foreign import ccall "THCTensorRandom.h THLongTensor_geometric"
  c_geometric :: Ptr (CTHState) -> Ptr (CTHLongTensor) -> CDouble -> IO (())

-- | p_random : Pointer to function : state self -> void
foreign import ccall "THCTensorRandom.h &THLongTensor_random"
  p_random :: FunPtr (Ptr (CTHState) -> Ptr (CTHLongTensor) -> IO (()))

-- | p_clampedRandom : Pointer to function : state self min max -> void
foreign import ccall "THCTensorRandom.h &THLongTensor_clampedRandom"
  p_clampedRandom :: FunPtr (Ptr (CTHState) -> Ptr (CTHLongTensor) -> CLLong -> CLLong -> IO (()))

-- | p_cappedRandom : Pointer to function : state self max -> void
foreign import ccall "THCTensorRandom.h &THLongTensor_cappedRandom"
  p_cappedRandom :: FunPtr (Ptr (CTHState) -> Ptr (CTHLongTensor) -> CLLong -> IO (()))

-- | p_bernoulli : Pointer to function : state self p -> void
foreign import ccall "THCTensorRandom.h &THLongTensor_bernoulli"
  p_bernoulli :: FunPtr (Ptr (CTHState) -> Ptr (CTHLongTensor) -> CDouble -> IO (()))

-- | p_bernoulli_DoubleTensor : Pointer to function : state self p -> void
foreign import ccall "THCTensorRandom.h &THLongTensor_bernoulli_DoubleTensor"
  p_bernoulli_DoubleTensor :: FunPtr (Ptr (CTHState) -> Ptr (CTHLongTensor) -> Ptr (CTHDoubleTensor) -> IO (()))

-- | p_geometric : Pointer to function : state self p -> void
foreign import ccall "THCTensorRandom.h &THLongTensor_geometric"
  p_geometric :: FunPtr (Ptr (CTHState) -> Ptr (CTHLongTensor) -> CDouble -> IO (()))