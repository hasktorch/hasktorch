{-# LANGUAGE ForeignFunctionInterface #-}

module THFile (
    c_THFile_THFile_isOpened,
    c_THFile_THFile_isQuiet,
    c_THFile_THFile_isReadable,
    c_THFile_THFile_isWritable,
    c_THFile_THFile_isBinary,
    c_THFile_THFile_isAutoSpacing,
    c_THFile_THFile_hasError,
    c_THFile_THFile_binary,
    c_THFile_THFile_ascii,
    c_THFile_THFile_autoSpacing,
    c_THFile_THFile_noAutoSpacing,
    c_THFile_THFile_quiet,
    c_THFile_THFile_pedantic,
    c_THFile_THFile_clearError,
    c_THFile_THFile_readByteScalar,
    c_THFile_THFile_readCharScalar,
    c_THFile_THFile_readShortScalar,
    c_THFile_THFile_readIntScalar,
    c_THFile_THFile_readLongScalar,
    c_THFile_THFile_readFloatScalar,
    c_THFile_THFile_readDoubleScalar,
    c_THFile_THFile_writeByteScalar,
    c_THFile_THFile_writeCharScalar,
    c_THFile_THFile_writeShortScalar,
    c_THFile_THFile_writeIntScalar,
    c_THFile_THFile_writeLongScalar,
    c_THFile_THFile_writeFloatScalar,
    c_THFile_THFile_writeDoubleScalar,
    c_THFile_THFile_readByte,
    c_THFile_THFile_readChar,
    c_THFile_THFile_readShort,
    c_THFile_THFile_readInt,
    c_THFile_THFile_readLong,
    c_THFile_THFile_readFloat,
    c_THFile_THFile_readDouble,
    c_THFile_THFile_writeByte,
    c_THFile_THFile_writeChar,
    c_THFile_THFile_writeShort,
    c_THFile_THFile_writeInt,
    c_THFile_THFile_writeLong,
    c_THFile_THFile_writeFloat,
    c_THFile_THFile_writeDouble,
    c_THFile_THFile_readByteRaw,
    c_THFile_THFile_readCharRaw,
    c_THFile_THFile_readShortRaw,
    c_THFile_THFile_readIntRaw,
    c_THFile_THFile_readLongRaw,
    c_THFile_THFile_readFloatRaw,
    c_THFile_THFile_readDoubleRaw,
    c_THFile_THFile_readStringRaw,
    c_THFile_THFile_writeByteRaw,
    c_THFile_THFile_writeCharRaw,
    c_THFile_THFile_writeShortRaw,
    c_THFile_THFile_writeIntRaw,
    c_THFile_THFile_writeLongRaw,
    c_THFile_THFile_writeFloatRaw,
    c_THFile_THFile_writeDoubleRaw,
    c_THFile_THFile_writeStringRaw,
    c_THFile_THFile_readHalfScalar,
    c_THFile_THFile_writeHalfScalar,
    c_THFile_THFile_readHalf,
    c_THFile_THFile_writeHalf,
    c_THFile_THFile_readHalfRaw,
    c_THFile_THFile_writeHalfRaw,
    c_THFile_THFile_synchronize,
    c_THFile_THFile_seek,
    c_THFile_THFile_seekEnd,
    c_THFile_THFile_position,
    c_THFile_THFile_close,
    c_THFile_THFile_free) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THFile_THFile_isOpened : self -> int
foreign import ccall "THFile.h THFile_THFile_isOpened"
  c_THFile_THFile_isOpened :: Ptr CTHFile -> CInt

-- |c_THFile_THFile_isQuiet : self -> int
foreign import ccall "THFile.h THFile_THFile_isQuiet"
  c_THFile_THFile_isQuiet :: Ptr CTHFile -> CInt

-- |c_THFile_THFile_isReadable : self -> int
foreign import ccall "THFile.h THFile_THFile_isReadable"
  c_THFile_THFile_isReadable :: Ptr CTHFile -> CInt

-- |c_THFile_THFile_isWritable : self -> int
foreign import ccall "THFile.h THFile_THFile_isWritable"
  c_THFile_THFile_isWritable :: Ptr CTHFile -> CInt

-- |c_THFile_THFile_isBinary : self -> int
foreign import ccall "THFile.h THFile_THFile_isBinary"
  c_THFile_THFile_isBinary :: Ptr CTHFile -> CInt

-- |c_THFile_THFile_isAutoSpacing : self -> int
foreign import ccall "THFile.h THFile_THFile_isAutoSpacing"
  c_THFile_THFile_isAutoSpacing :: Ptr CTHFile -> CInt

-- |c_THFile_THFile_hasError : self -> int
foreign import ccall "THFile.h THFile_THFile_hasError"
  c_THFile_THFile_hasError :: Ptr CTHFile -> CInt

-- |c_THFile_THFile_binary : self -> void
foreign import ccall "THFile.h THFile_THFile_binary"
  c_THFile_THFile_binary :: Ptr CTHFile -> IO ()

-- |c_THFile_THFile_ascii : self -> void
foreign import ccall "THFile.h THFile_THFile_ascii"
  c_THFile_THFile_ascii :: Ptr CTHFile -> IO ()

-- |c_THFile_THFile_autoSpacing : self -> void
foreign import ccall "THFile.h THFile_THFile_autoSpacing"
  c_THFile_THFile_autoSpacing :: Ptr CTHFile -> IO ()

-- |c_THFile_THFile_noAutoSpacing : self -> void
foreign import ccall "THFile.h THFile_THFile_noAutoSpacing"
  c_THFile_THFile_noAutoSpacing :: Ptr CTHFile -> IO ()

-- |c_THFile_THFile_quiet : self -> void
foreign import ccall "THFile.h THFile_THFile_quiet"
  c_THFile_THFile_quiet :: Ptr CTHFile -> IO ()

-- |c_THFile_THFile_pedantic : self -> void
foreign import ccall "THFile.h THFile_THFile_pedantic"
  c_THFile_THFile_pedantic :: Ptr CTHFile -> IO ()

-- |c_THFile_THFile_clearError : self -> void
foreign import ccall "THFile.h THFile_THFile_clearError"
  c_THFile_THFile_clearError :: Ptr CTHFile -> IO ()

-- |c_THFile_THFile_readByteScalar : self -> char
foreign import ccall "THFile.h THFile_THFile_readByteScalar"
  c_THFile_THFile_readByteScalar :: Ptr CTHFile -> CChar

-- |c_THFile_THFile_readCharScalar : self -> char
foreign import ccall "THFile.h THFile_THFile_readCharScalar"
  c_THFile_THFile_readCharScalar :: Ptr CTHFile -> CChar

-- |c_THFile_THFile_readShortScalar : self -> short
foreign import ccall "THFile.h THFile_THFile_readShortScalar"
  c_THFile_THFile_readShortScalar :: Ptr CTHFile -> CShort

-- |c_THFile_THFile_readIntScalar : self -> int
foreign import ccall "THFile.h THFile_THFile_readIntScalar"
  c_THFile_THFile_readIntScalar :: Ptr CTHFile -> CInt

-- |c_THFile_THFile_readLongScalar : self -> long
foreign import ccall "THFile.h THFile_THFile_readLongScalar"
  c_THFile_THFile_readLongScalar :: Ptr CTHFile -> CLong

-- |c_THFile_THFile_readFloatScalar : self -> float
foreign import ccall "THFile.h THFile_THFile_readFloatScalar"
  c_THFile_THFile_readFloatScalar :: Ptr CTHFile -> CFloat

-- |c_THFile_THFile_readDoubleScalar : self -> double
foreign import ccall "THFile.h THFile_THFile_readDoubleScalar"
  c_THFile_THFile_readDoubleScalar :: Ptr CTHFile -> CDouble

-- |c_THFile_THFile_writeByteScalar : self scalar -> void
foreign import ccall "THFile.h THFile_THFile_writeByteScalar"
  c_THFile_THFile_writeByteScalar :: Ptr CTHFile -> CChar -> IO ()

-- |c_THFile_THFile_writeCharScalar : self scalar -> void
foreign import ccall "THFile.h THFile_THFile_writeCharScalar"
  c_THFile_THFile_writeCharScalar :: Ptr CTHFile -> CChar -> IO ()

-- |c_THFile_THFile_writeShortScalar : self scalar -> void
foreign import ccall "THFile.h THFile_THFile_writeShortScalar"
  c_THFile_THFile_writeShortScalar :: Ptr CTHFile -> CShort -> IO ()

-- |c_THFile_THFile_writeIntScalar : self scalar -> void
foreign import ccall "THFile.h THFile_THFile_writeIntScalar"
  c_THFile_THFile_writeIntScalar :: Ptr CTHFile -> CInt -> IO ()

-- |c_THFile_THFile_writeLongScalar : self scalar -> void
foreign import ccall "THFile.h THFile_THFile_writeLongScalar"
  c_THFile_THFile_writeLongScalar :: Ptr CTHFile -> CLong -> IO ()

-- |c_THFile_THFile_writeFloatScalar : self scalar -> void
foreign import ccall "THFile.h THFile_THFile_writeFloatScalar"
  c_THFile_THFile_writeFloatScalar :: Ptr CTHFile -> CFloat -> IO ()

-- |c_THFile_THFile_writeDoubleScalar : self scalar -> void
foreign import ccall "THFile.h THFile_THFile_writeDoubleScalar"
  c_THFile_THFile_writeDoubleScalar :: Ptr CTHFile -> CDouble -> IO ()

-- |c_THFile_THFile_readByte : self storage -> size_t
foreign import ccall "THFile.h THFile_THFile_readByte"
  c_THFile_THFile_readByte :: Ptr CTHFile -> Ptr CTHByteStorage -> CSize

-- |c_THFile_THFile_readChar : self storage -> size_t
foreign import ccall "THFile.h THFile_THFile_readChar"
  c_THFile_THFile_readChar :: Ptr CTHFile -> Ptr CTHCharStorage -> CSize

-- |c_THFile_THFile_readShort : self storage -> size_t
foreign import ccall "THFile.h THFile_THFile_readShort"
  c_THFile_THFile_readShort :: Ptr CTHFile -> Ptr CTHShortStorage -> CSize

-- |c_THFile_THFile_readInt : self storage -> size_t
foreign import ccall "THFile.h THFile_THFile_readInt"
  c_THFile_THFile_readInt :: Ptr CTHFile -> Ptr CTHIntStorage -> CSize

-- |c_THFile_THFile_readLong : self storage -> size_t
foreign import ccall "THFile.h THFile_THFile_readLong"
  c_THFile_THFile_readLong :: Ptr CTHFile -> Ptr CTHLongStorage -> CSize

-- |c_THFile_THFile_readFloat : self storage -> size_t
foreign import ccall "THFile.h THFile_THFile_readFloat"
  c_THFile_THFile_readFloat :: Ptr CTHFile -> Ptr CTHFloatStorage -> CSize

-- |c_THFile_THFile_readDouble : self storage -> size_t
foreign import ccall "THFile.h THFile_THFile_readDouble"
  c_THFile_THFile_readDouble :: Ptr CTHFile -> Ptr CTHDoubleStorage -> CSize

-- |c_THFile_THFile_writeByte : self storage -> size_t
foreign import ccall "THFile.h THFile_THFile_writeByte"
  c_THFile_THFile_writeByte :: Ptr CTHFile -> Ptr CTHByteStorage -> CSize

-- |c_THFile_THFile_writeChar : self storage -> size_t
foreign import ccall "THFile.h THFile_THFile_writeChar"
  c_THFile_THFile_writeChar :: Ptr CTHFile -> Ptr CTHCharStorage -> CSize

-- |c_THFile_THFile_writeShort : self storage -> size_t
foreign import ccall "THFile.h THFile_THFile_writeShort"
  c_THFile_THFile_writeShort :: Ptr CTHFile -> Ptr CTHShortStorage -> CSize

-- |c_THFile_THFile_writeInt : self storage -> size_t
foreign import ccall "THFile.h THFile_THFile_writeInt"
  c_THFile_THFile_writeInt :: Ptr CTHFile -> Ptr CTHIntStorage -> CSize

-- |c_THFile_THFile_writeLong : self storage -> size_t
foreign import ccall "THFile.h THFile_THFile_writeLong"
  c_THFile_THFile_writeLong :: Ptr CTHFile -> Ptr CTHLongStorage -> CSize

-- |c_THFile_THFile_writeFloat : self storage -> size_t
foreign import ccall "THFile.h THFile_THFile_writeFloat"
  c_THFile_THFile_writeFloat :: Ptr CTHFile -> Ptr CTHFloatStorage -> CSize

-- |c_THFile_THFile_writeDouble : self storage -> size_t
foreign import ccall "THFile.h THFile_THFile_writeDouble"
  c_THFile_THFile_writeDouble :: Ptr CTHFile -> Ptr CTHDoubleStorage -> CSize

-- |c_THFile_THFile_readByteRaw : self data n -> size_t
foreign import ccall "THFile.h THFile_THFile_readByteRaw"
  c_THFile_THFile_readByteRaw :: Ptr CTHFile -> Ptr CChar -> CSize -> CSize

-- |c_THFile_THFile_readCharRaw : self data n -> size_t
foreign import ccall "THFile.h THFile_THFile_readCharRaw"
  c_THFile_THFile_readCharRaw :: Ptr CTHFile -> Ptr CChar -> CSize -> CSize

-- |c_THFile_THFile_readShortRaw : self data n -> size_t
foreign import ccall "THFile.h THFile_THFile_readShortRaw"
  c_THFile_THFile_readShortRaw :: Ptr CTHFile -> Ptr CShort -> CSize -> CSize

-- |c_THFile_THFile_readIntRaw : self data n -> size_t
foreign import ccall "THFile.h THFile_THFile_readIntRaw"
  c_THFile_THFile_readIntRaw :: Ptr CTHFile -> CIntPtr -> CSize -> CSize

-- |c_THFile_THFile_readLongRaw : self data n -> size_t
foreign import ccall "THFile.h THFile_THFile_readLongRaw"
  c_THFile_THFile_readLongRaw :: Ptr CTHFile -> Ptr CLong -> CSize -> CSize

-- |c_THFile_THFile_readFloatRaw : self data n -> size_t
foreign import ccall "THFile.h THFile_THFile_readFloatRaw"
  c_THFile_THFile_readFloatRaw :: Ptr CTHFile -> Ptr CFloat -> CSize -> CSize

-- |c_THFile_THFile_readDoubleRaw : self data n -> size_t
foreign import ccall "THFile.h THFile_THFile_readDoubleRaw"
  c_THFile_THFile_readDoubleRaw :: Ptr CTHFile -> Ptr CDouble -> CSize -> CSize

-- |c_THFile_THFile_readStringRaw : self format str_ -> size_t
foreign import ccall "THFile.h THFile_THFile_readStringRaw"
  c_THFile_THFile_readStringRaw :: Ptr CTHFile -> Ptr CChar -> Ptr (Ptr CChar) -> CSize

-- |c_THFile_THFile_writeByteRaw : self data n -> size_t
foreign import ccall "THFile.h THFile_THFile_writeByteRaw"
  c_THFile_THFile_writeByteRaw :: Ptr CTHFile -> Ptr CChar -> CSize -> CSize

-- |c_THFile_THFile_writeCharRaw : self data n -> size_t
foreign import ccall "THFile.h THFile_THFile_writeCharRaw"
  c_THFile_THFile_writeCharRaw :: Ptr CTHFile -> Ptr CChar -> CSize -> CSize

-- |c_THFile_THFile_writeShortRaw : self data n -> size_t
foreign import ccall "THFile.h THFile_THFile_writeShortRaw"
  c_THFile_THFile_writeShortRaw :: Ptr CTHFile -> Ptr CShort -> CSize -> CSize

-- |c_THFile_THFile_writeIntRaw : self data n -> size_t
foreign import ccall "THFile.h THFile_THFile_writeIntRaw"
  c_THFile_THFile_writeIntRaw :: Ptr CTHFile -> CIntPtr -> CSize -> CSize

-- |c_THFile_THFile_writeLongRaw : self data n -> size_t
foreign import ccall "THFile.h THFile_THFile_writeLongRaw"
  c_THFile_THFile_writeLongRaw :: Ptr CTHFile -> Ptr CLong -> CSize -> CSize

-- |c_THFile_THFile_writeFloatRaw : self data n -> size_t
foreign import ccall "THFile.h THFile_THFile_writeFloatRaw"
  c_THFile_THFile_writeFloatRaw :: Ptr CTHFile -> Ptr CFloat -> CSize -> CSize

-- |c_THFile_THFile_writeDoubleRaw : self data n -> size_t
foreign import ccall "THFile.h THFile_THFile_writeDoubleRaw"
  c_THFile_THFile_writeDoubleRaw :: Ptr CTHFile -> Ptr CDouble -> CSize -> CSize

-- |c_THFile_THFile_writeStringRaw : self str size -> size_t
foreign import ccall "THFile.h THFile_THFile_writeStringRaw"
  c_THFile_THFile_writeStringRaw :: Ptr CTHFile -> Ptr CChar -> CSize -> CSize

-- |c_THFile_THFile_readHalfScalar : self -> THHalf
foreign import ccall "THFile.h THFile_THFile_readHalfScalar"
  c_THFile_THFile_readHalfScalar :: Ptr CTHFile -> CTHHalf

-- |c_THFile_THFile_writeHalfScalar : self scalar -> void
foreign import ccall "THFile.h THFile_THFile_writeHalfScalar"
  c_THFile_THFile_writeHalfScalar :: Ptr CTHFile -> CTHHalf -> IO ()

-- |c_THFile_THFile_readHalf : self storage -> size_t
foreign import ccall "THFile.h THFile_THFile_readHalf"
  c_THFile_THFile_readHalf :: Ptr CTHFile -> Ptr CTHHalfStorage -> CSize

-- |c_THFile_THFile_writeHalf : self storage -> size_t
foreign import ccall "THFile.h THFile_THFile_writeHalf"
  c_THFile_THFile_writeHalf :: Ptr CTHFile -> Ptr CTHHalfStorage -> CSize

-- |c_THFile_THFile_readHalfRaw : self data size -> size_t
foreign import ccall "THFile.h THFile_THFile_readHalfRaw"
  c_THFile_THFile_readHalfRaw :: Ptr CTHFile -> Ptr CTHHalf -> CSize -> CSize

-- |c_THFile_THFile_writeHalfRaw : self data size -> size_t
foreign import ccall "THFile.h THFile_THFile_writeHalfRaw"
  c_THFile_THFile_writeHalfRaw :: Ptr CTHFile -> Ptr CTHHalf -> CSize -> CSize

-- |c_THFile_THFile_synchronize : self -> void
foreign import ccall "THFile.h THFile_THFile_synchronize"
  c_THFile_THFile_synchronize :: Ptr CTHFile -> IO ()

-- |c_THFile_THFile_seek : self position -> void
foreign import ccall "THFile.h THFile_THFile_seek"
  c_THFile_THFile_seek :: Ptr CTHFile -> CSize -> IO ()

-- |c_THFile_THFile_seekEnd : self -> void
foreign import ccall "THFile.h THFile_THFile_seekEnd"
  c_THFile_THFile_seekEnd :: Ptr CTHFile -> IO ()

-- |c_THFile_THFile_position : self -> size_t
foreign import ccall "THFile.h THFile_THFile_position"
  c_THFile_THFile_position :: Ptr CTHFile -> CSize

-- |c_THFile_THFile_close : self -> void
foreign import ccall "THFile.h THFile_THFile_close"
  c_THFile_THFile_close :: Ptr CTHFile -> IO ()

-- |c_THFile_THFile_free : self -> void
foreign import ccall "THFile.h THFile_THFile_free"
  c_THFile_THFile_free :: Ptr CTHFile -> IO ()