{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Byte.TensorRandom where

import Foreign
import Foreign.C.Types
import Data.Word
import Data.Int
import Torch.Types.TH

-- | c_random :  self _generator -> void
foreign import ccall "THTensorRandom.h THByteTensor_random"
  c_random_ :: Ptr C'THByteTensor -> Ptr C'THGenerator -> IO ()

-- | alias of c_random_ with unused argument (for CTHState) to unify backpack signatures.
c_random :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THGenerator -> IO ()
c_random = const c_random_

-- | c_clampedRandom :  self _generator min max -> void
foreign import ccall "THTensorRandom.h THByteTensor_clampedRandom"
  c_clampedRandom_ :: Ptr C'THByteTensor -> Ptr C'THGenerator -> CLLong -> CLLong -> IO ()

-- | alias of c_clampedRandom_ with unused argument (for CTHState) to unify backpack signatures.
c_clampedRandom :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THGenerator -> CLLong -> CLLong -> IO ()
c_clampedRandom = const c_clampedRandom_

-- | c_cappedRandom :  self _generator max -> void
foreign import ccall "THTensorRandom.h THByteTensor_cappedRandom"
  c_cappedRandom_ :: Ptr C'THByteTensor -> Ptr C'THGenerator -> CLLong -> IO ()

-- | alias of c_cappedRandom_ with unused argument (for CTHState) to unify backpack signatures.
c_cappedRandom :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THGenerator -> CLLong -> IO ()
c_cappedRandom = const c_cappedRandom_

-- | c_geometric :  self _generator p -> void
foreign import ccall "THTensorRandom.h THByteTensor_geometric"
  c_geometric_ :: Ptr C'THByteTensor -> Ptr C'THGenerator -> CDouble -> IO ()

-- | alias of c_geometric_ with unused argument (for CTHState) to unify backpack signatures.
c_geometric :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THGenerator -> CDouble -> IO ()
c_geometric = const c_geometric_

-- | c_bernoulli :  self _generator p -> void
foreign import ccall "THTensorRandom.h THByteTensor_bernoulli"
  c_bernoulli_ :: Ptr C'THByteTensor -> Ptr C'THGenerator -> CDouble -> IO ()

-- | alias of c_bernoulli_ with unused argument (for CTHState) to unify backpack signatures.
c_bernoulli :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THGenerator -> CDouble -> IO ()
c_bernoulli = const c_bernoulli_

-- | c_bernoulli_FloatTensor :  self _generator p -> void
foreign import ccall "THTensorRandom.h THByteTensor_bernoulli_FloatTensor"
  c_bernoulli_FloatTensor_ :: Ptr C'THByteTensor -> Ptr C'THGenerator -> Ptr C'THFloatTensor -> IO ()

-- | alias of c_bernoulli_FloatTensor_ with unused argument (for CTHState) to unify backpack signatures.
c_bernoulli_FloatTensor :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THGenerator -> Ptr C'THFloatTensor -> IO ()
c_bernoulli_FloatTensor = const c_bernoulli_FloatTensor_

-- | c_bernoulli_DoubleTensor :  self _generator p -> void
foreign import ccall "THTensorRandom.h THByteTensor_bernoulli_DoubleTensor"
  c_bernoulli_DoubleTensor_ :: Ptr C'THByteTensor -> Ptr C'THGenerator -> Ptr C'THDoubleTensor -> IO ()

-- | alias of c_bernoulli_DoubleTensor_ with unused argument (for CTHState) to unify backpack signatures.
c_bernoulli_DoubleTensor :: Ptr C'THState -> Ptr C'THByteTensor -> Ptr C'THGenerator -> Ptr C'THDoubleTensor -> IO ()
c_bernoulli_DoubleTensor = const c_bernoulli_DoubleTensor_

-- | c_getRNGState :  _generator self -> void
foreign import ccall "THTensorRandom.h THByteTensor_getRNGState"
  c_getRNGState_ :: Ptr C'THGenerator -> Ptr C'THByteTensor -> IO ()

-- | alias of c_getRNGState_ with unused argument (for CTHState) to unify backpack signatures.
c_getRNGState :: Ptr C'THState -> Ptr C'THGenerator -> Ptr C'THByteTensor -> IO ()
c_getRNGState = const c_getRNGState_

-- | c_setRNGState :  _generator self -> void
foreign import ccall "THTensorRandom.h THByteTensor_setRNGState"
  c_setRNGState_ :: Ptr C'THGenerator -> Ptr C'THByteTensor -> IO ()

-- | alias of c_setRNGState_ with unused argument (for CTHState) to unify backpack signatures.
c_setRNGState :: Ptr C'THState -> Ptr C'THGenerator -> Ptr C'THByteTensor -> IO ()
c_setRNGState = const c_setRNGState_

-- | p_random : Pointer to function : self _generator -> void
foreign import ccall "THTensorRandom.h &THByteTensor_random"
  p_random :: FunPtr (Ptr C'THByteTensor -> Ptr C'THGenerator -> IO ())

-- | p_clampedRandom : Pointer to function : self _generator min max -> void
foreign import ccall "THTensorRandom.h &THByteTensor_clampedRandom"
  p_clampedRandom :: FunPtr (Ptr C'THByteTensor -> Ptr C'THGenerator -> CLLong -> CLLong -> IO ())

-- | p_cappedRandom : Pointer to function : self _generator max -> void
foreign import ccall "THTensorRandom.h &THByteTensor_cappedRandom"
  p_cappedRandom :: FunPtr (Ptr C'THByteTensor -> Ptr C'THGenerator -> CLLong -> IO ())

-- | p_geometric : Pointer to function : self _generator p -> void
foreign import ccall "THTensorRandom.h &THByteTensor_geometric"
  p_geometric :: FunPtr (Ptr C'THByteTensor -> Ptr C'THGenerator -> CDouble -> IO ())

-- | p_bernoulli : Pointer to function : self _generator p -> void
foreign import ccall "THTensorRandom.h &THByteTensor_bernoulli"
  p_bernoulli :: FunPtr (Ptr C'THByteTensor -> Ptr C'THGenerator -> CDouble -> IO ())

-- | p_bernoulli_FloatTensor : Pointer to function : self _generator p -> void
foreign import ccall "THTensorRandom.h &THByteTensor_bernoulli_FloatTensor"
  p_bernoulli_FloatTensor :: FunPtr (Ptr C'THByteTensor -> Ptr C'THGenerator -> Ptr C'THFloatTensor -> IO ())

-- | p_bernoulli_DoubleTensor : Pointer to function : self _generator p -> void
foreign import ccall "THTensorRandom.h &THByteTensor_bernoulli_DoubleTensor"
  p_bernoulli_DoubleTensor :: FunPtr (Ptr C'THByteTensor -> Ptr C'THGenerator -> Ptr C'THDoubleTensor -> IO ())

-- | p_getRNGState : Pointer to function : _generator self -> void
foreign import ccall "THTensorRandom.h &THByteTensor_getRNGState"
  p_getRNGState :: FunPtr (Ptr C'THGenerator -> Ptr C'THByteTensor -> IO ())

-- | p_setRNGState : Pointer to function : _generator self -> void
foreign import ccall "THTensorRandom.h &THByteTensor_setRNGState"
  p_setRNGState :: FunPtr (Ptr C'THGenerator -> Ptr C'THByteTensor -> IO ())