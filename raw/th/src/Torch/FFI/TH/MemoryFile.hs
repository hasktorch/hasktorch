{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.MemoryFile where

import Foreign
import Foreign.C.Types
import Data.Word
import Data.Int
import Torch.Types.TH

-- | c_THMemoryFile_newWithStorage :  storage mode -> THFile *
foreign import ccall "THMemoryFile.h THMemoryFile_newWithStorage"
  c_THMemoryFile_newWithStorage :: Ptr C'THCharStorage -> Ptr CChar -> IO (Ptr C'THFile)

-- | c_THMemoryFile_new :  mode -> THFile *
foreign import ccall "THMemoryFile.h THMemoryFile_new"
  c_THMemoryFile_new :: Ptr CChar -> IO (Ptr C'THFile)

-- | c_THMemoryFile_storage :  self -> THCharStorage *
foreign import ccall "THMemoryFile.h THMemoryFile_storage"
  c_THMemoryFile_storage :: Ptr C'THFile -> IO (Ptr C'THCharStorage)

-- | c_THMemoryFile_longSize :  self size -> void
foreign import ccall "THMemoryFile.h THMemoryFile_longSize"
  c_THMemoryFile_longSize :: Ptr C'THFile -> CInt -> IO ()

-- | p_THMemoryFile_newWithStorage : Pointer to function : storage mode -> THFile *
foreign import ccall "THMemoryFile.h &THMemoryFile_newWithStorage"
  p_THMemoryFile_newWithStorage :: FunPtr (Ptr C'THCharStorage -> Ptr CChar -> IO (Ptr C'THFile))

-- | p_THMemoryFile_new : Pointer to function : mode -> THFile *
foreign import ccall "THMemoryFile.h &THMemoryFile_new"
  p_THMemoryFile_new :: FunPtr (Ptr CChar -> IO (Ptr C'THFile))

-- | p_THMemoryFile_storage : Pointer to function : self -> THCharStorage *
foreign import ccall "THMemoryFile.h &THMemoryFile_storage"
  p_THMemoryFile_storage :: FunPtr (Ptr C'THFile -> IO (Ptr C'THCharStorage))

-- | p_THMemoryFile_longSize : Pointer to function : self size -> void
foreign import ccall "THMemoryFile.h &THMemoryFile_longSize"
  p_THMemoryFile_longSize :: FunPtr (Ptr C'THFile -> CInt -> IO ())