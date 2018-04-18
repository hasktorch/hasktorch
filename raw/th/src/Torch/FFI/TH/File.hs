{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.File where

import Foreign
import Foreign.C.Types
import Data.Word
import Data.Int
import Torch.Types.TH

-- | c_THFile_isOpened :  self -> int
foreign import ccall "THFile.h THFile_isOpened"
  c_THFile_isOpened :: Ptr C'THFile -> IO CInt

-- | c_THFile_isQuiet :  self -> int
foreign import ccall "THFile.h THFile_isQuiet"
  c_THFile_isQuiet :: Ptr C'THFile -> IO CInt

-- | c_THFile_isReadable :  self -> int
foreign import ccall "THFile.h THFile_isReadable"
  c_THFile_isReadable :: Ptr C'THFile -> IO CInt

-- | c_THFile_isWritable :  self -> int
foreign import ccall "THFile.h THFile_isWritable"
  c_THFile_isWritable :: Ptr C'THFile -> IO CInt

-- | c_THFile_isBinary :  self -> int
foreign import ccall "THFile.h THFile_isBinary"
  c_THFile_isBinary :: Ptr C'THFile -> IO CInt

-- | c_THFile_isAutoSpacing :  self -> int
foreign import ccall "THFile.h THFile_isAutoSpacing"
  c_THFile_isAutoSpacing :: Ptr C'THFile -> IO CInt

-- | c_THFile_hasError :  self -> int
foreign import ccall "THFile.h THFile_hasError"
  c_THFile_hasError :: Ptr C'THFile -> IO CInt

-- | c_THFile_binary :  self -> void
foreign import ccall "THFile.h THFile_binary"
  c_THFile_binary :: Ptr C'THFile -> IO ()

-- | c_THFile_ascii :  self -> void
foreign import ccall "THFile.h THFile_ascii"
  c_THFile_ascii :: Ptr C'THFile -> IO ()

-- | c_THFile_autoSpacing :  self -> void
foreign import ccall "THFile.h THFile_autoSpacing"
  c_THFile_autoSpacing :: Ptr C'THFile -> IO ()

-- | c_THFile_noAutoSpacing :  self -> void
foreign import ccall "THFile.h THFile_noAutoSpacing"
  c_THFile_noAutoSpacing :: Ptr C'THFile -> IO ()

-- | c_THFile_quiet :  self -> void
foreign import ccall "THFile.h THFile_quiet"
  c_THFile_quiet :: Ptr C'THFile -> IO ()

-- | c_THFile_pedantic :  self -> void
foreign import ccall "THFile.h THFile_pedantic"
  c_THFile_pedantic :: Ptr C'THFile -> IO ()

-- | c_THFile_clearError :  self -> void
foreign import ccall "THFile.h THFile_clearError"
  c_THFile_clearError :: Ptr C'THFile -> IO ()

-- | c_THFile_readByteScalar :  self -> uint8_t
foreign import ccall "THFile.h THFile_readByteScalar"
  c_THFile_readByteScalar :: Ptr C'THFile -> IO CUChar

-- | c_THFile_readCharScalar :  self -> int8_t
foreign import ccall "THFile.h THFile_readCharScalar"
  c_THFile_readCharScalar :: Ptr C'THFile -> IO CSChar

-- | c_THFile_readShortScalar :  self -> int16_t
foreign import ccall "THFile.h THFile_readShortScalar"
  c_THFile_readShortScalar :: Ptr C'THFile -> IO CShort

-- | c_THFile_readIntScalar :  self -> int32_t
foreign import ccall "THFile.h THFile_readIntScalar"
  c_THFile_readIntScalar :: Ptr C'THFile -> IO CInt

-- | c_THFile_readLongScalar :  self -> int64_t
foreign import ccall "THFile.h THFile_readLongScalar"
  c_THFile_readLongScalar :: Ptr C'THFile -> IO CLLong

-- | c_THFile_readFloatScalar :  self -> float
foreign import ccall "THFile.h THFile_readFloatScalar"
  c_THFile_readFloatScalar :: Ptr C'THFile -> IO CFloat

-- | c_THFile_readDoubleScalar :  self -> double
foreign import ccall "THFile.h THFile_readDoubleScalar"
  c_THFile_readDoubleScalar :: Ptr C'THFile -> IO CDouble

-- | c_THFile_writeByteScalar :  self scalar -> void
foreign import ccall "THFile.h THFile_writeByteScalar"
  c_THFile_writeByteScalar :: Ptr C'THFile -> CUChar -> IO ()

-- | c_THFile_writeCharScalar :  self scalar -> void
foreign import ccall "THFile.h THFile_writeCharScalar"
  c_THFile_writeCharScalar :: Ptr C'THFile -> CSChar -> IO ()

-- | c_THFile_writeShortScalar :  self scalar -> void
foreign import ccall "THFile.h THFile_writeShortScalar"
  c_THFile_writeShortScalar :: Ptr C'THFile -> CShort -> IO ()

-- | c_THFile_writeIntScalar :  self scalar -> void
foreign import ccall "THFile.h THFile_writeIntScalar"
  c_THFile_writeIntScalar :: Ptr C'THFile -> CInt -> IO ()

-- | c_THFile_writeLongScalar :  self scalar -> void
foreign import ccall "THFile.h THFile_writeLongScalar"
  c_THFile_writeLongScalar :: Ptr C'THFile -> CLLong -> IO ()

-- | c_THFile_writeFloatScalar :  self scalar -> void
foreign import ccall "THFile.h THFile_writeFloatScalar"
  c_THFile_writeFloatScalar :: Ptr C'THFile -> CFloat -> IO ()

-- | c_THFile_writeDoubleScalar :  self scalar -> void
foreign import ccall "THFile.h THFile_writeDoubleScalar"
  c_THFile_writeDoubleScalar :: Ptr C'THFile -> CDouble -> IO ()

-- | c_THFile_readByte :  self storage -> size_t
foreign import ccall "THFile.h THFile_readByte"
  c_THFile_readByte :: Ptr C'THFile -> Ptr C'THByteStorage -> IO CSize

-- | c_THFile_readChar :  self storage -> size_t
foreign import ccall "THFile.h THFile_readChar"
  c_THFile_readChar :: Ptr C'THFile -> Ptr C'THCharStorage -> IO CSize

-- | c_THFile_readShort :  self storage -> size_t
foreign import ccall "THFile.h THFile_readShort"
  c_THFile_readShort :: Ptr C'THFile -> Ptr C'THShortStorage -> IO CSize

-- | c_THFile_readInt :  self storage -> size_t
foreign import ccall "THFile.h THFile_readInt"
  c_THFile_readInt :: Ptr C'THFile -> Ptr C'THIntStorage -> IO CSize

-- | c_THFile_readLong :  self storage -> size_t
foreign import ccall "THFile.h THFile_readLong"
  c_THFile_readLong :: Ptr C'THFile -> Ptr C'THLongStorage -> IO CSize

-- | c_THFile_readFloat :  self storage -> size_t
foreign import ccall "THFile.h THFile_readFloat"
  c_THFile_readFloat :: Ptr C'THFile -> Ptr C'THFloatStorage -> IO CSize

-- | c_THFile_readDouble :  self storage -> size_t
foreign import ccall "THFile.h THFile_readDouble"
  c_THFile_readDouble :: Ptr C'THFile -> Ptr C'THDoubleStorage -> IO CSize

-- | c_THFile_writeByte :  self storage -> size_t
foreign import ccall "THFile.h THFile_writeByte"
  c_THFile_writeByte :: Ptr C'THFile -> Ptr C'THByteStorage -> IO CSize

-- | c_THFile_writeChar :  self storage -> size_t
foreign import ccall "THFile.h THFile_writeChar"
  c_THFile_writeChar :: Ptr C'THFile -> Ptr C'THCharStorage -> IO CSize

-- | c_THFile_writeShort :  self storage -> size_t
foreign import ccall "THFile.h THFile_writeShort"
  c_THFile_writeShort :: Ptr C'THFile -> Ptr C'THShortStorage -> IO CSize

-- | c_THFile_writeInt :  self storage -> size_t
foreign import ccall "THFile.h THFile_writeInt"
  c_THFile_writeInt :: Ptr C'THFile -> Ptr C'THIntStorage -> IO CSize

-- | c_THFile_writeLong :  self storage -> size_t
foreign import ccall "THFile.h THFile_writeLong"
  c_THFile_writeLong :: Ptr C'THFile -> Ptr C'THLongStorage -> IO CSize

-- | c_THFile_writeFloat :  self storage -> size_t
foreign import ccall "THFile.h THFile_writeFloat"
  c_THFile_writeFloat :: Ptr C'THFile -> Ptr C'THFloatStorage -> IO CSize

-- | c_THFile_writeDouble :  self storage -> size_t
foreign import ccall "THFile.h THFile_writeDouble"
  c_THFile_writeDouble :: Ptr C'THFile -> Ptr C'THDoubleStorage -> IO CSize

-- | c_THFile_readByteRaw :  self data n -> size_t
foreign import ccall "THFile.h THFile_readByteRaw"
  c_THFile_readByteRaw :: Ptr C'THFile -> Ptr CUChar -> CSize -> IO CSize

-- | c_THFile_readCharRaw :  self data n -> size_t
foreign import ccall "THFile.h THFile_readCharRaw"
  c_THFile_readCharRaw :: Ptr C'THFile -> Ptr CSChar -> CSize -> IO CSize

-- | c_THFile_readShortRaw :  self data n -> size_t
foreign import ccall "THFile.h THFile_readShortRaw"
  c_THFile_readShortRaw :: Ptr C'THFile -> Ptr CShort -> CSize -> IO CSize

-- | c_THFile_readIntRaw :  self data n -> size_t
foreign import ccall "THFile.h THFile_readIntRaw"
  c_THFile_readIntRaw :: Ptr C'THFile -> Ptr CInt -> CSize -> IO CSize

-- | c_THFile_readLongRaw :  self data n -> size_t
foreign import ccall "THFile.h THFile_readLongRaw"
  c_THFile_readLongRaw :: Ptr C'THFile -> Ptr CLLong -> CSize -> IO CSize

-- | c_THFile_readFloatRaw :  self data n -> size_t
foreign import ccall "THFile.h THFile_readFloatRaw"
  c_THFile_readFloatRaw :: Ptr C'THFile -> Ptr CFloat -> CSize -> IO CSize

-- | c_THFile_readDoubleRaw :  self data n -> size_t
foreign import ccall "THFile.h THFile_readDoubleRaw"
  c_THFile_readDoubleRaw :: Ptr C'THFile -> Ptr CDouble -> CSize -> IO CSize

-- | c_THFile_readStringRaw :  self format str_ -> size_t
foreign import ccall "THFile.h THFile_readStringRaw"
  c_THFile_readStringRaw :: Ptr C'THFile -> Ptr CChar -> Ptr (Ptr CChar) -> IO CSize

-- | c_THFile_writeByteRaw :  self data n -> size_t
foreign import ccall "THFile.h THFile_writeByteRaw"
  c_THFile_writeByteRaw :: Ptr C'THFile -> Ptr CUChar -> CSize -> IO CSize

-- | c_THFile_writeCharRaw :  self data n -> size_t
foreign import ccall "THFile.h THFile_writeCharRaw"
  c_THFile_writeCharRaw :: Ptr C'THFile -> Ptr CSChar -> CSize -> IO CSize

-- | c_THFile_writeShortRaw :  self data n -> size_t
foreign import ccall "THFile.h THFile_writeShortRaw"
  c_THFile_writeShortRaw :: Ptr C'THFile -> Ptr CShort -> CSize -> IO CSize

-- | c_THFile_writeIntRaw :  self data n -> size_t
foreign import ccall "THFile.h THFile_writeIntRaw"
  c_THFile_writeIntRaw :: Ptr C'THFile -> Ptr CInt -> CSize -> IO CSize

-- | c_THFile_writeLongRaw :  self data n -> size_t
foreign import ccall "THFile.h THFile_writeLongRaw"
  c_THFile_writeLongRaw :: Ptr C'THFile -> Ptr CLLong -> CSize -> IO CSize

-- | c_THFile_writeFloatRaw :  self data n -> size_t
foreign import ccall "THFile.h THFile_writeFloatRaw"
  c_THFile_writeFloatRaw :: Ptr C'THFile -> Ptr CFloat -> CSize -> IO CSize

-- | c_THFile_writeDoubleRaw :  self data n -> size_t
foreign import ccall "THFile.h THFile_writeDoubleRaw"
  c_THFile_writeDoubleRaw :: Ptr C'THFile -> Ptr CDouble -> CSize -> IO CSize

-- | c_THFile_writeStringRaw :  self str size -> size_t
foreign import ccall "THFile.h THFile_writeStringRaw"
  c_THFile_writeStringRaw :: Ptr C'THFile -> Ptr CChar -> CSize -> IO CSize

-- | c_THFile_readHalfScalar :  self -> THHalf
foreign import ccall "THFile.h THFile_readHalfScalar"
  c_THFile_readHalfScalar :: Ptr C'THFile -> IO C'THHalf

-- | c_THFile_writeHalfScalar :  self scalar -> void
foreign import ccall "THFile.h THFile_writeHalfScalar"
  c_THFile_writeHalfScalar :: Ptr C'THFile -> C'THHalf -> IO ()

-- | c_THFile_readHalf :  self storage -> size_t
foreign import ccall "THFile.h THFile_readHalf"
  c_THFile_readHalf :: Ptr C'THFile -> Ptr C'THHalfStorage -> IO CSize

-- | c_THFile_writeHalf :  self storage -> size_t
foreign import ccall "THFile.h THFile_writeHalf"
  c_THFile_writeHalf :: Ptr C'THFile -> Ptr C'THHalfStorage -> IO CSize

-- | c_THFile_readHalfRaw :  self data size -> size_t
foreign import ccall "THFile.h THFile_readHalfRaw"
  c_THFile_readHalfRaw :: Ptr C'THFile -> Ptr C'THHalf -> CSize -> IO CSize

-- | c_THFile_writeHalfRaw :  self data size -> size_t
foreign import ccall "THFile.h THFile_writeHalfRaw"
  c_THFile_writeHalfRaw :: Ptr C'THFile -> Ptr C'THHalf -> CSize -> IO CSize

-- | c_THFile_synchronize :  self -> void
foreign import ccall "THFile.h THFile_synchronize"
  c_THFile_synchronize :: Ptr C'THFile -> IO ()

-- | c_THFile_seek :  self position -> void
foreign import ccall "THFile.h THFile_seek"
  c_THFile_seek :: Ptr C'THFile -> CSize -> IO ()

-- | c_THFile_seekEnd :  self -> void
foreign import ccall "THFile.h THFile_seekEnd"
  c_THFile_seekEnd :: Ptr C'THFile -> IO ()

-- | c_THFile_position :  self -> size_t
foreign import ccall "THFile.h THFile_position"
  c_THFile_position :: Ptr C'THFile -> IO CSize

-- | c_THFile_close :  self -> void
foreign import ccall "THFile.h THFile_close"
  c_THFile_close :: Ptr C'THFile -> IO ()

-- | c_THFile_free :  self -> void
foreign import ccall "THFile.h THFile_free"
  c_THFile_free :: Ptr C'THFile -> IO ()

-- | p_THFile_isOpened : Pointer to function : self -> int
foreign import ccall "THFile.h &THFile_isOpened"
  p_THFile_isOpened :: FunPtr (Ptr C'THFile -> IO CInt)

-- | p_THFile_isQuiet : Pointer to function : self -> int
foreign import ccall "THFile.h &THFile_isQuiet"
  p_THFile_isQuiet :: FunPtr (Ptr C'THFile -> IO CInt)

-- | p_THFile_isReadable : Pointer to function : self -> int
foreign import ccall "THFile.h &THFile_isReadable"
  p_THFile_isReadable :: FunPtr (Ptr C'THFile -> IO CInt)

-- | p_THFile_isWritable : Pointer to function : self -> int
foreign import ccall "THFile.h &THFile_isWritable"
  p_THFile_isWritable :: FunPtr (Ptr C'THFile -> IO CInt)

-- | p_THFile_isBinary : Pointer to function : self -> int
foreign import ccall "THFile.h &THFile_isBinary"
  p_THFile_isBinary :: FunPtr (Ptr C'THFile -> IO CInt)

-- | p_THFile_isAutoSpacing : Pointer to function : self -> int
foreign import ccall "THFile.h &THFile_isAutoSpacing"
  p_THFile_isAutoSpacing :: FunPtr (Ptr C'THFile -> IO CInt)

-- | p_THFile_hasError : Pointer to function : self -> int
foreign import ccall "THFile.h &THFile_hasError"
  p_THFile_hasError :: FunPtr (Ptr C'THFile -> IO CInt)

-- | p_THFile_binary : Pointer to function : self -> void
foreign import ccall "THFile.h &THFile_binary"
  p_THFile_binary :: FunPtr (Ptr C'THFile -> IO ())

-- | p_THFile_ascii : Pointer to function : self -> void
foreign import ccall "THFile.h &THFile_ascii"
  p_THFile_ascii :: FunPtr (Ptr C'THFile -> IO ())

-- | p_THFile_autoSpacing : Pointer to function : self -> void
foreign import ccall "THFile.h &THFile_autoSpacing"
  p_THFile_autoSpacing :: FunPtr (Ptr C'THFile -> IO ())

-- | p_THFile_noAutoSpacing : Pointer to function : self -> void
foreign import ccall "THFile.h &THFile_noAutoSpacing"
  p_THFile_noAutoSpacing :: FunPtr (Ptr C'THFile -> IO ())

-- | p_THFile_quiet : Pointer to function : self -> void
foreign import ccall "THFile.h &THFile_quiet"
  p_THFile_quiet :: FunPtr (Ptr C'THFile -> IO ())

-- | p_THFile_pedantic : Pointer to function : self -> void
foreign import ccall "THFile.h &THFile_pedantic"
  p_THFile_pedantic :: FunPtr (Ptr C'THFile -> IO ())

-- | p_THFile_clearError : Pointer to function : self -> void
foreign import ccall "THFile.h &THFile_clearError"
  p_THFile_clearError :: FunPtr (Ptr C'THFile -> IO ())

-- | p_THFile_readByteScalar : Pointer to function : self -> uint8_t
foreign import ccall "THFile.h &THFile_readByteScalar"
  p_THFile_readByteScalar :: FunPtr (Ptr C'THFile -> IO CUChar)

-- | p_THFile_readCharScalar : Pointer to function : self -> int8_t
foreign import ccall "THFile.h &THFile_readCharScalar"
  p_THFile_readCharScalar :: FunPtr (Ptr C'THFile -> IO CSChar)

-- | p_THFile_readShortScalar : Pointer to function : self -> int16_t
foreign import ccall "THFile.h &THFile_readShortScalar"
  p_THFile_readShortScalar :: FunPtr (Ptr C'THFile -> IO CShort)

-- | p_THFile_readIntScalar : Pointer to function : self -> int32_t
foreign import ccall "THFile.h &THFile_readIntScalar"
  p_THFile_readIntScalar :: FunPtr (Ptr C'THFile -> IO CInt)

-- | p_THFile_readLongScalar : Pointer to function : self -> int64_t
foreign import ccall "THFile.h &THFile_readLongScalar"
  p_THFile_readLongScalar :: FunPtr (Ptr C'THFile -> IO CLLong)

-- | p_THFile_readFloatScalar : Pointer to function : self -> float
foreign import ccall "THFile.h &THFile_readFloatScalar"
  p_THFile_readFloatScalar :: FunPtr (Ptr C'THFile -> IO CFloat)

-- | p_THFile_readDoubleScalar : Pointer to function : self -> double
foreign import ccall "THFile.h &THFile_readDoubleScalar"
  p_THFile_readDoubleScalar :: FunPtr (Ptr C'THFile -> IO CDouble)

-- | p_THFile_writeByteScalar : Pointer to function : self scalar -> void
foreign import ccall "THFile.h &THFile_writeByteScalar"
  p_THFile_writeByteScalar :: FunPtr (Ptr C'THFile -> CUChar -> IO ())

-- | p_THFile_writeCharScalar : Pointer to function : self scalar -> void
foreign import ccall "THFile.h &THFile_writeCharScalar"
  p_THFile_writeCharScalar :: FunPtr (Ptr C'THFile -> CSChar -> IO ())

-- | p_THFile_writeShortScalar : Pointer to function : self scalar -> void
foreign import ccall "THFile.h &THFile_writeShortScalar"
  p_THFile_writeShortScalar :: FunPtr (Ptr C'THFile -> CShort -> IO ())

-- | p_THFile_writeIntScalar : Pointer to function : self scalar -> void
foreign import ccall "THFile.h &THFile_writeIntScalar"
  p_THFile_writeIntScalar :: FunPtr (Ptr C'THFile -> CInt -> IO ())

-- | p_THFile_writeLongScalar : Pointer to function : self scalar -> void
foreign import ccall "THFile.h &THFile_writeLongScalar"
  p_THFile_writeLongScalar :: FunPtr (Ptr C'THFile -> CLLong -> IO ())

-- | p_THFile_writeFloatScalar : Pointer to function : self scalar -> void
foreign import ccall "THFile.h &THFile_writeFloatScalar"
  p_THFile_writeFloatScalar :: FunPtr (Ptr C'THFile -> CFloat -> IO ())

-- | p_THFile_writeDoubleScalar : Pointer to function : self scalar -> void
foreign import ccall "THFile.h &THFile_writeDoubleScalar"
  p_THFile_writeDoubleScalar :: FunPtr (Ptr C'THFile -> CDouble -> IO ())

-- | p_THFile_readByte : Pointer to function : self storage -> size_t
foreign import ccall "THFile.h &THFile_readByte"
  p_THFile_readByte :: FunPtr (Ptr C'THFile -> Ptr C'THByteStorage -> IO CSize)

-- | p_THFile_readChar : Pointer to function : self storage -> size_t
foreign import ccall "THFile.h &THFile_readChar"
  p_THFile_readChar :: FunPtr (Ptr C'THFile -> Ptr C'THCharStorage -> IO CSize)

-- | p_THFile_readShort : Pointer to function : self storage -> size_t
foreign import ccall "THFile.h &THFile_readShort"
  p_THFile_readShort :: FunPtr (Ptr C'THFile -> Ptr C'THShortStorage -> IO CSize)

-- | p_THFile_readInt : Pointer to function : self storage -> size_t
foreign import ccall "THFile.h &THFile_readInt"
  p_THFile_readInt :: FunPtr (Ptr C'THFile -> Ptr C'THIntStorage -> IO CSize)

-- | p_THFile_readLong : Pointer to function : self storage -> size_t
foreign import ccall "THFile.h &THFile_readLong"
  p_THFile_readLong :: FunPtr (Ptr C'THFile -> Ptr C'THLongStorage -> IO CSize)

-- | p_THFile_readFloat : Pointer to function : self storage -> size_t
foreign import ccall "THFile.h &THFile_readFloat"
  p_THFile_readFloat :: FunPtr (Ptr C'THFile -> Ptr C'THFloatStorage -> IO CSize)

-- | p_THFile_readDouble : Pointer to function : self storage -> size_t
foreign import ccall "THFile.h &THFile_readDouble"
  p_THFile_readDouble :: FunPtr (Ptr C'THFile -> Ptr C'THDoubleStorage -> IO CSize)

-- | p_THFile_writeByte : Pointer to function : self storage -> size_t
foreign import ccall "THFile.h &THFile_writeByte"
  p_THFile_writeByte :: FunPtr (Ptr C'THFile -> Ptr C'THByteStorage -> IO CSize)

-- | p_THFile_writeChar : Pointer to function : self storage -> size_t
foreign import ccall "THFile.h &THFile_writeChar"
  p_THFile_writeChar :: FunPtr (Ptr C'THFile -> Ptr C'THCharStorage -> IO CSize)

-- | p_THFile_writeShort : Pointer to function : self storage -> size_t
foreign import ccall "THFile.h &THFile_writeShort"
  p_THFile_writeShort :: FunPtr (Ptr C'THFile -> Ptr C'THShortStorage -> IO CSize)

-- | p_THFile_writeInt : Pointer to function : self storage -> size_t
foreign import ccall "THFile.h &THFile_writeInt"
  p_THFile_writeInt :: FunPtr (Ptr C'THFile -> Ptr C'THIntStorage -> IO CSize)

-- | p_THFile_writeLong : Pointer to function : self storage -> size_t
foreign import ccall "THFile.h &THFile_writeLong"
  p_THFile_writeLong :: FunPtr (Ptr C'THFile -> Ptr C'THLongStorage -> IO CSize)

-- | p_THFile_writeFloat : Pointer to function : self storage -> size_t
foreign import ccall "THFile.h &THFile_writeFloat"
  p_THFile_writeFloat :: FunPtr (Ptr C'THFile -> Ptr C'THFloatStorage -> IO CSize)

-- | p_THFile_writeDouble : Pointer to function : self storage -> size_t
foreign import ccall "THFile.h &THFile_writeDouble"
  p_THFile_writeDouble :: FunPtr (Ptr C'THFile -> Ptr C'THDoubleStorage -> IO CSize)

-- | p_THFile_readByteRaw : Pointer to function : self data n -> size_t
foreign import ccall "THFile.h &THFile_readByteRaw"
  p_THFile_readByteRaw :: FunPtr (Ptr C'THFile -> Ptr CUChar -> CSize -> IO CSize)

-- | p_THFile_readCharRaw : Pointer to function : self data n -> size_t
foreign import ccall "THFile.h &THFile_readCharRaw"
  p_THFile_readCharRaw :: FunPtr (Ptr C'THFile -> Ptr CSChar -> CSize -> IO CSize)

-- | p_THFile_readShortRaw : Pointer to function : self data n -> size_t
foreign import ccall "THFile.h &THFile_readShortRaw"
  p_THFile_readShortRaw :: FunPtr (Ptr C'THFile -> Ptr CShort -> CSize -> IO CSize)

-- | p_THFile_readIntRaw : Pointer to function : self data n -> size_t
foreign import ccall "THFile.h &THFile_readIntRaw"
  p_THFile_readIntRaw :: FunPtr (Ptr C'THFile -> Ptr CInt -> CSize -> IO CSize)

-- | p_THFile_readLongRaw : Pointer to function : self data n -> size_t
foreign import ccall "THFile.h &THFile_readLongRaw"
  p_THFile_readLongRaw :: FunPtr (Ptr C'THFile -> Ptr CLLong -> CSize -> IO CSize)

-- | p_THFile_readFloatRaw : Pointer to function : self data n -> size_t
foreign import ccall "THFile.h &THFile_readFloatRaw"
  p_THFile_readFloatRaw :: FunPtr (Ptr C'THFile -> Ptr CFloat -> CSize -> IO CSize)

-- | p_THFile_readDoubleRaw : Pointer to function : self data n -> size_t
foreign import ccall "THFile.h &THFile_readDoubleRaw"
  p_THFile_readDoubleRaw :: FunPtr (Ptr C'THFile -> Ptr CDouble -> CSize -> IO CSize)

-- | p_THFile_readStringRaw : Pointer to function : self format str_ -> size_t
foreign import ccall "THFile.h &THFile_readStringRaw"
  p_THFile_readStringRaw :: FunPtr (Ptr C'THFile -> Ptr CChar -> Ptr (Ptr CChar) -> IO CSize)

-- | p_THFile_writeByteRaw : Pointer to function : self data n -> size_t
foreign import ccall "THFile.h &THFile_writeByteRaw"
  p_THFile_writeByteRaw :: FunPtr (Ptr C'THFile -> Ptr CUChar -> CSize -> IO CSize)

-- | p_THFile_writeCharRaw : Pointer to function : self data n -> size_t
foreign import ccall "THFile.h &THFile_writeCharRaw"
  p_THFile_writeCharRaw :: FunPtr (Ptr C'THFile -> Ptr CSChar -> CSize -> IO CSize)

-- | p_THFile_writeShortRaw : Pointer to function : self data n -> size_t
foreign import ccall "THFile.h &THFile_writeShortRaw"
  p_THFile_writeShortRaw :: FunPtr (Ptr C'THFile -> Ptr CShort -> CSize -> IO CSize)

-- | p_THFile_writeIntRaw : Pointer to function : self data n -> size_t
foreign import ccall "THFile.h &THFile_writeIntRaw"
  p_THFile_writeIntRaw :: FunPtr (Ptr C'THFile -> Ptr CInt -> CSize -> IO CSize)

-- | p_THFile_writeLongRaw : Pointer to function : self data n -> size_t
foreign import ccall "THFile.h &THFile_writeLongRaw"
  p_THFile_writeLongRaw :: FunPtr (Ptr C'THFile -> Ptr CLLong -> CSize -> IO CSize)

-- | p_THFile_writeFloatRaw : Pointer to function : self data n -> size_t
foreign import ccall "THFile.h &THFile_writeFloatRaw"
  p_THFile_writeFloatRaw :: FunPtr (Ptr C'THFile -> Ptr CFloat -> CSize -> IO CSize)

-- | p_THFile_writeDoubleRaw : Pointer to function : self data n -> size_t
foreign import ccall "THFile.h &THFile_writeDoubleRaw"
  p_THFile_writeDoubleRaw :: FunPtr (Ptr C'THFile -> Ptr CDouble -> CSize -> IO CSize)

-- | p_THFile_writeStringRaw : Pointer to function : self str size -> size_t
foreign import ccall "THFile.h &THFile_writeStringRaw"
  p_THFile_writeStringRaw :: FunPtr (Ptr C'THFile -> Ptr CChar -> CSize -> IO CSize)

-- | p_THFile_readHalfScalar : Pointer to function : self -> THHalf
foreign import ccall "THFile.h &THFile_readHalfScalar"
  p_THFile_readHalfScalar :: FunPtr (Ptr C'THFile -> IO C'THHalf)

-- | p_THFile_writeHalfScalar : Pointer to function : self scalar -> void
foreign import ccall "THFile.h &THFile_writeHalfScalar"
  p_THFile_writeHalfScalar :: FunPtr (Ptr C'THFile -> C'THHalf -> IO ())

-- | p_THFile_readHalf : Pointer to function : self storage -> size_t
foreign import ccall "THFile.h &THFile_readHalf"
  p_THFile_readHalf :: FunPtr (Ptr C'THFile -> Ptr C'THHalfStorage -> IO CSize)

-- | p_THFile_writeHalf : Pointer to function : self storage -> size_t
foreign import ccall "THFile.h &THFile_writeHalf"
  p_THFile_writeHalf :: FunPtr (Ptr C'THFile -> Ptr C'THHalfStorage -> IO CSize)

-- | p_THFile_readHalfRaw : Pointer to function : self data size -> size_t
foreign import ccall "THFile.h &THFile_readHalfRaw"
  p_THFile_readHalfRaw :: FunPtr (Ptr C'THFile -> Ptr C'THHalf -> CSize -> IO CSize)

-- | p_THFile_writeHalfRaw : Pointer to function : self data size -> size_t
foreign import ccall "THFile.h &THFile_writeHalfRaw"
  p_THFile_writeHalfRaw :: FunPtr (Ptr C'THFile -> Ptr C'THHalf -> CSize -> IO CSize)

-- | p_THFile_synchronize : Pointer to function : self -> void
foreign import ccall "THFile.h &THFile_synchronize"
  p_THFile_synchronize :: FunPtr (Ptr C'THFile -> IO ())

-- | p_THFile_seek : Pointer to function : self position -> void
foreign import ccall "THFile.h &THFile_seek"
  p_THFile_seek :: FunPtr (Ptr C'THFile -> CSize -> IO ())

-- | p_THFile_seekEnd : Pointer to function : self -> void
foreign import ccall "THFile.h &THFile_seekEnd"
  p_THFile_seekEnd :: FunPtr (Ptr C'THFile -> IO ())

-- | p_THFile_position : Pointer to function : self -> size_t
foreign import ccall "THFile.h &THFile_position"
  p_THFile_position :: FunPtr (Ptr C'THFile -> IO CSize)

-- | p_THFile_close : Pointer to function : self -> void
foreign import ccall "THFile.h &THFile_close"
  p_THFile_close :: FunPtr (Ptr C'THFile -> IO ())

-- | p_THFile_free : Pointer to function : self -> void
foreign import ccall "THFile.h &THFile_free"
  p_THFile_free :: FunPtr (Ptr C'THFile -> IO ())