{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Half
  ( c_TH_float2halfbits
  , c_TH_halfbits2float
  , c_TH_float2half
  , c_TH_half2float
  , p_TH_float2halfbits
  , p_TH_halfbits2float
  , p_TH_float2half
  , p_TH_half2float
  ) where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_TH_float2halfbits :    -> void
foreign import ccall "THHalf.h TH_float2halfbits"
  c_TH_float2halfbits :: Ptr (CFloat) -> Ptr (CShort) -> IO (())

-- | c_TH_halfbits2float :    -> void
foreign import ccall "THHalf.h TH_halfbits2float"
  c_TH_halfbits2float :: Ptr (CShort) -> Ptr (CFloat) -> IO (())

-- | c_TH_float2half :   -> THHalf
foreign import ccall "THHalf.h TH_float2half"
  c_TH_float2half :: CFloat -> IO (CTHHalf)

-- | c_TH_half2float :   -> float
foreign import ccall "THHalf.h TH_half2float"
  c_TH_half2float :: CTHHalf -> IO (CFloat)

-- | p_TH_float2halfbits : Pointer to function :   -> void
foreign import ccall "THHalf.h &TH_float2halfbits"
  p_TH_float2halfbits :: FunPtr (Ptr (CFloat) -> Ptr (CShort) -> IO (()))

-- | p_TH_halfbits2float : Pointer to function :   -> void
foreign import ccall "THHalf.h &TH_halfbits2float"
  p_TH_halfbits2float :: FunPtr (Ptr (CShort) -> Ptr (CFloat) -> IO (()))

-- | p_TH_float2half : Pointer to function :  -> THHalf
foreign import ccall "THHalf.h &TH_float2half"
  p_TH_float2half :: FunPtr (CFloat -> IO (CTHHalf))

-- | p_TH_half2float : Pointer to function :  -> float
foreign import ccall "THHalf.h &TH_half2float"
  p_TH_half2float :: FunPtr (CTHHalf -> IO (CFloat))