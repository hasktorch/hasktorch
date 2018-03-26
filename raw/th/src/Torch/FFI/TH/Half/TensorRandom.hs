{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Half.TensorRandom where

import Foreign
import Foreign.C.Types
import Data.Word
import Data.Int
import Torch.Types.TH

-- | c_random :  self _generator -> void
foreign import ccall "THTensorRandom.h THHalfTensor_random"
  c_random_ :: Ptr C'THHalfTensor -> Ptr C'THGenerator -> IO ()

-- | alias of c_random_ with unused argument (for CTHState) to unify backpack signatures.
c_random :: Ptr C'THState -> Ptr C'THHalfTensor -> Ptr C'THGenerator -> IO ()
c_random = const c_random_

-- | c_clampedRandom :  self _generator min max -> void
foreign import ccall "THTensorRandom.h THHalfTensor_clampedRandom"
  c_clampedRandom_ :: Ptr C'THHalfTensor -> Ptr C'THGenerator -> CLLong -> CLLong -> IO ()

-- | alias of c_clampedRandom_ with unused argument (for CTHState) to unify backpack signatures.
c_clampedRandom :: Ptr C'THState -> Ptr C'THHalfTensor -> Ptr C'THGenerator -> CLLong -> CLLong -> IO ()
c_clampedRandom = const c_clampedRandom_

-- | c_cappedRandom :  self _generator max -> void
foreign import ccall "THTensorRandom.h THHalfTensor_cappedRandom"
  c_cappedRandom_ :: Ptr C'THHalfTensor -> Ptr C'THGenerator -> CLLong -> IO ()

-- | alias of c_cappedRandom_ with unused argument (for CTHState) to unify backpack signatures.
c_cappedRandom :: Ptr C'THState -> Ptr C'THHalfTensor -> Ptr C'THGenerator -> CLLong -> IO ()
c_cappedRandom = const c_cappedRandom_

-- | c_geometric :  self _generator p -> void
foreign import ccall "THTensorRandom.h THHalfTensor_geometric"
  c_geometric_ :: Ptr C'THHalfTensor -> Ptr C'THGenerator -> CDouble -> IO ()

-- | alias of c_geometric_ with unused argument (for CTHState) to unify backpack signatures.
c_geometric :: Ptr C'THState -> Ptr C'THHalfTensor -> Ptr C'THGenerator -> CDouble -> IO ()
c_geometric = const c_geometric_

-- | c_bernoulli :  self _generator p -> void
foreign import ccall "THTensorRandom.h THHalfTensor_bernoulli"
  c_bernoulli_ :: Ptr C'THHalfTensor -> Ptr C'THGenerator -> CDouble -> IO ()

-- | alias of c_bernoulli_ with unused argument (for CTHState) to unify backpack signatures.
c_bernoulli :: Ptr C'THState -> Ptr C'THHalfTensor -> Ptr C'THGenerator -> CDouble -> IO ()
c_bernoulli = const c_bernoulli_

-- | c_bernoulli_FloatTensor :  self _generator p -> void
foreign import ccall "THTensorRandom.h THHalfTensor_bernoulli_FloatTensor"
  c_bernoulli_FloatTensor_ :: Ptr C'THHalfTensor -> Ptr C'THGenerator -> Ptr C'THFloatTensor -> IO ()

-- | alias of c_bernoulli_FloatTensor_ with unused argument (for CTHState) to unify backpack signatures.
c_bernoulli_FloatTensor :: Ptr C'THState -> Ptr C'THHalfTensor -> Ptr C'THGenerator -> Ptr C'THFloatTensor -> IO ()
c_bernoulli_FloatTensor = const c_bernoulli_FloatTensor_

-- | c_bernoulli_DoubleTensor :  self _generator p -> void
foreign import ccall "THTensorRandom.h THHalfTensor_bernoulli_DoubleTensor"
  c_bernoulli_DoubleTensor_ :: Ptr C'THHalfTensor -> Ptr C'THGenerator -> Ptr C'THDoubleTensor -> IO ()

-- | alias of c_bernoulli_DoubleTensor_ with unused argument (for CTHState) to unify backpack signatures.
c_bernoulli_DoubleTensor :: Ptr C'THState -> Ptr C'THHalfTensor -> Ptr C'THGenerator -> Ptr C'THDoubleTensor -> IO ()
c_bernoulli_DoubleTensor = const c_bernoulli_DoubleTensor_

-- | p_random : Pointer to function : self _generator -> void
foreign import ccall "THTensorRandom.h &THHalfTensor_random"
  p_random :: FunPtr (Ptr C'THHalfTensor -> Ptr C'THGenerator -> IO ())

-- | p_clampedRandom : Pointer to function : self _generator min max -> void
foreign import ccall "THTensorRandom.h &THHalfTensor_clampedRandom"
  p_clampedRandom :: FunPtr (Ptr C'THHalfTensor -> Ptr C'THGenerator -> CLLong -> CLLong -> IO ())

-- | p_cappedRandom : Pointer to function : self _generator max -> void
foreign import ccall "THTensorRandom.h &THHalfTensor_cappedRandom"
  p_cappedRandom :: FunPtr (Ptr C'THHalfTensor -> Ptr C'THGenerator -> CLLong -> IO ())

-- | p_geometric : Pointer to function : self _generator p -> void
foreign import ccall "THTensorRandom.h &THHalfTensor_geometric"
  p_geometric :: FunPtr (Ptr C'THHalfTensor -> Ptr C'THGenerator -> CDouble -> IO ())

-- | p_bernoulli : Pointer to function : self _generator p -> void
foreign import ccall "THTensorRandom.h &THHalfTensor_bernoulli"
  p_bernoulli :: FunPtr (Ptr C'THHalfTensor -> Ptr C'THGenerator -> CDouble -> IO ())

-- | p_bernoulli_FloatTensor : Pointer to function : self _generator p -> void
foreign import ccall "THTensorRandom.h &THHalfTensor_bernoulli_FloatTensor"
  p_bernoulli_FloatTensor :: FunPtr (Ptr C'THHalfTensor -> Ptr C'THGenerator -> Ptr C'THFloatTensor -> IO ())

-- | p_bernoulli_DoubleTensor : Pointer to function : self _generator p -> void
foreign import ccall "THTensorRandom.h &THHalfTensor_bernoulli_DoubleTensor"
  p_bernoulli_DoubleTensor :: FunPtr (Ptr C'THHalfTensor -> Ptr C'THGenerator -> Ptr C'THDoubleTensor -> IO ())