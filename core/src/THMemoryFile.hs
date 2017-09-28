{-# LANGUAGE ForeignFunctionInterface #-}

module THMemoryFile (
    c_THMemoryFile_newWithStorage,
    c_THMemoryFile_new,
    c_THMemoryFile_storage,
    c_THMemoryFile_longSize) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THMemoryFile_newWithStorage : storage mode -> THFile *
foreign import ccall "THMemoryFile.h THMemoryFile_newWithStorage"
  c_THMemoryFile_newWithStorage :: Ptr CTHCharStorage -> Ptr CChar -> IO (Ptr CTHFile)

-- |c_THMemoryFile_new : mode -> THFile *
foreign import ccall "THMemoryFile.h THMemoryFile_new"
  c_THMemoryFile_new :: Ptr CChar -> IO (Ptr CTHFile)

-- |c_THMemoryFile_storage : self -> THCharStorage *
foreign import ccall "THMemoryFile.h THMemoryFile_storage"
  c_THMemoryFile_storage :: Ptr CTHFile -> IO (Ptr CTHCharStorage)

-- |c_THMemoryFile_longSize : self size -> void
foreign import ccall "THMemoryFile.h THMemoryFile_longSize"
  c_THMemoryFile_longSize :: Ptr CTHFile -> CInt -> IO ()