{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.DiskFile where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_THDiskFile_new :  name mode isQuiet -> THFile *
foreign import ccall "THDiskFile.h THDiskFile_new"
  c_THDiskFile_new :: Ptr CChar -> Ptr CChar -> CInt -> IO (Ptr C'THFile)

-- | c_THPipeFile_new :  name mode isQuiet -> THFile *
foreign import ccall "THDiskFile.h THPipeFile_new"
  c_THPipeFile_new :: Ptr CChar -> Ptr CChar -> CInt -> IO (Ptr C'THFile)

-- | c_THDiskFile_name :  self -> char *
foreign import ccall "THDiskFile.h THDiskFile_name"
  c_THDiskFile_name :: Ptr C'THFile -> IO (Ptr CChar)

-- | c_THDiskFile_isLittleEndianCPU :   -> int
foreign import ccall "THDiskFile.h THDiskFile_isLittleEndianCPU"
  c_THDiskFile_isLittleEndianCPU :: IO CInt

-- | c_THDiskFile_isBigEndianCPU :   -> int
foreign import ccall "THDiskFile.h THDiskFile_isBigEndianCPU"
  c_THDiskFile_isBigEndianCPU :: IO CInt

-- | c_THDiskFile_nativeEndianEncoding :  self -> void
foreign import ccall "THDiskFile.h THDiskFile_nativeEndianEncoding"
  c_THDiskFile_nativeEndianEncoding :: Ptr C'THFile -> IO ()

-- | c_THDiskFile_littleEndianEncoding :  self -> void
foreign import ccall "THDiskFile.h THDiskFile_littleEndianEncoding"
  c_THDiskFile_littleEndianEncoding :: Ptr C'THFile -> IO ()

-- | c_THDiskFile_bigEndianEncoding :  self -> void
foreign import ccall "THDiskFile.h THDiskFile_bigEndianEncoding"
  c_THDiskFile_bigEndianEncoding :: Ptr C'THFile -> IO ()

-- | c_THDiskFile_longSize :  self size -> void
foreign import ccall "THDiskFile.h THDiskFile_longSize"
  c_THDiskFile_longSize :: Ptr C'THFile -> CInt -> IO ()

-- | c_THDiskFile_noBuffer :  self -> void
foreign import ccall "THDiskFile.h THDiskFile_noBuffer"
  c_THDiskFile_noBuffer :: Ptr C'THFile -> IO ()

-- | p_THDiskFile_new : Pointer to function : name mode isQuiet -> THFile *
foreign import ccall "THDiskFile.h &THDiskFile_new"
  p_THDiskFile_new :: FunPtr (Ptr CChar -> Ptr CChar -> CInt -> IO (Ptr C'THFile))

-- | p_THPipeFile_new : Pointer to function : name mode isQuiet -> THFile *
foreign import ccall "THDiskFile.h &THPipeFile_new"
  p_THPipeFile_new :: FunPtr (Ptr CChar -> Ptr CChar -> CInt -> IO (Ptr C'THFile))

-- | p_THDiskFile_name : Pointer to function : self -> char *
foreign import ccall "THDiskFile.h &THDiskFile_name"
  p_THDiskFile_name :: FunPtr (Ptr C'THFile -> IO (Ptr CChar))

-- | p_THDiskFile_isLittleEndianCPU : Pointer to function :  -> int
foreign import ccall "THDiskFile.h &THDiskFile_isLittleEndianCPU"
  p_THDiskFile_isLittleEndianCPU :: FunPtr (IO CInt)

-- | p_THDiskFile_isBigEndianCPU : Pointer to function :  -> int
foreign import ccall "THDiskFile.h &THDiskFile_isBigEndianCPU"
  p_THDiskFile_isBigEndianCPU :: FunPtr (IO CInt)

-- | p_THDiskFile_nativeEndianEncoding : Pointer to function : self -> void
foreign import ccall "THDiskFile.h &THDiskFile_nativeEndianEncoding"
  p_THDiskFile_nativeEndianEncoding :: FunPtr (Ptr C'THFile -> IO ())

-- | p_THDiskFile_littleEndianEncoding : Pointer to function : self -> void
foreign import ccall "THDiskFile.h &THDiskFile_littleEndianEncoding"
  p_THDiskFile_littleEndianEncoding :: FunPtr (Ptr C'THFile -> IO ())

-- | p_THDiskFile_bigEndianEncoding : Pointer to function : self -> void
foreign import ccall "THDiskFile.h &THDiskFile_bigEndianEncoding"
  p_THDiskFile_bigEndianEncoding :: FunPtr (Ptr C'THFile -> IO ())

-- | p_THDiskFile_longSize : Pointer to function : self size -> void
foreign import ccall "THDiskFile.h &THDiskFile_longSize"
  p_THDiskFile_longSize :: FunPtr (Ptr C'THFile -> CInt -> IO ())

-- | p_THDiskFile_noBuffer : Pointer to function : self -> void
foreign import ccall "THDiskFile.h &THDiskFile_noBuffer"
  p_THDiskFile_noBuffer :: FunPtr (Ptr C'THFile -> IO ())