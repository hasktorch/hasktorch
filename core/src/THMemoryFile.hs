{-# LANGUAGE ForeignFunctionInterface #-}

module THMemoryFile (
    c_THMemoryFile_newWithStorage,
    c_THMemoryFile_new,
    c_THMemoryFile_storage,
    c_THMemoryFile_longSize,
    p_THMemoryFile_newWithStorage,
    p_THMemoryFile_new,
    p_THMemoryFile_storage,
    p_THMemoryFile_longSize) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THMemoryFile_newWithStorage : storage mode -> THFile *
foreign import ccall unsafe "THMemoryFile.h THMemoryFile_newWithStorage"
  c_THMemoryFile_newWithStorage :: Ptr CTHCharStorage -> Ptr CChar -> IO (Ptr CTHFile)

-- |c_THMemoryFile_new : mode -> THFile *
foreign import ccall unsafe "THMemoryFile.h THMemoryFile_new"
  c_THMemoryFile_new :: Ptr CChar -> IO (Ptr CTHFile)

-- |c_THMemoryFile_storage : self -> THCharStorage *
foreign import ccall unsafe "THMemoryFile.h THMemoryFile_storage"
  c_THMemoryFile_storage :: Ptr CTHFile -> IO (Ptr CTHCharStorage)

-- |c_THMemoryFile_longSize : self size -> void
foreign import ccall unsafe "THMemoryFile.h THMemoryFile_longSize"
  c_THMemoryFile_longSize :: Ptr CTHFile -> CInt -> IO ()

-- |p_THMemoryFile_newWithStorage : Pointer to storage mode -> THFile *
foreign import ccall unsafe "THMemoryFile.h &THMemoryFile_newWithStorage"
  p_THMemoryFile_newWithStorage :: FunPtr (Ptr CTHCharStorage -> Ptr CChar -> IO (Ptr CTHFile))

-- |p_THMemoryFile_new : Pointer to mode -> THFile *
foreign import ccall unsafe "THMemoryFile.h &THMemoryFile_new"
  p_THMemoryFile_new :: FunPtr (Ptr CChar -> IO (Ptr CTHFile))

-- |p_THMemoryFile_storage : Pointer to self -> THCharStorage *
foreign import ccall unsafe "THMemoryFile.h &THMemoryFile_storage"
  p_THMemoryFile_storage :: FunPtr (Ptr CTHFile -> IO (Ptr CTHCharStorage))

-- |p_THMemoryFile_longSize : Pointer to self size -> void
foreign import ccall unsafe "THMemoryFile.h &THMemoryFile_longSize"
  p_THMemoryFile_longSize :: FunPtr (Ptr CTHFile -> CInt -> IO ())