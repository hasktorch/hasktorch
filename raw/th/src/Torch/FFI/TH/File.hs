{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.File
  ( c_THFile_isOpened
  , c_THFile_isQuiet
  , c_THFile_isReadable
  , c_THFile_isWritable
  , c_THFile_isBinary
  , c_THFile_isAutoSpacing
  , c_THFile_hasError
  , c_THFile_binary
  , c_THFile_ascii
  , c_THFile_autoSpacing
  , c_THFile_noAutoSpacing
  , c_THFile_quiet
  , c_THFile_pedantic
  , c_THFile_clearError
  , c_THFile_readByteScalar
  , c_THFile_readCharScalar
  , c_THFile_readShortScalar
  , c_THFile_readIntScalar
  , c_THFile_readLongScalar
  , c_THFile_readFloatScalar
  , c_THFile_readDoubleScalar
  , c_THFile_writeByteScalar
  , c_THFile_writeCharScalar
  , c_THFile_writeShortScalar
  , c_THFile_writeIntScalar
  , c_THFile_writeLongScalar
  , c_THFile_writeFloatScalar
  , c_THFile_writeDoubleScalar
  , c_THFile_readByte
  , c_THFile_readChar
  , c_THFile_readShort
  , c_THFile_readInt
  , c_THFile_readLong
  , c_THFile_readFloat
  , c_THFile_readDouble
  , c_THFile_writeByte
  , c_THFile_writeChar
  , c_THFile_writeShort
  , c_THFile_writeInt
  , c_THFile_writeLong
  , c_THFile_writeFloat
  , c_THFile_writeDouble
  , c_THFile_readByteRaw
  , c_THFile_readCharRaw
  , c_THFile_readShortRaw
  , c_THFile_readIntRaw
  , c_THFile_readLongRaw
  , c_THFile_readFloatRaw
  , c_THFile_readDoubleRaw
  , c_THFile_readStringRaw
  , c_THFile_writeByteRaw
  , c_THFile_writeCharRaw
  , c_THFile_writeShortRaw
  , c_THFile_writeIntRaw
  , c_THFile_writeLongRaw
  , c_THFile_writeFloatRaw
  , c_THFile_writeDoubleRaw
  , c_THFile_writeStringRaw
  , c_THFile_readHalfScalar
  , c_THFile_writeHalfScalar
  , c_THFile_readHalf
  , c_THFile_writeHalf
  , c_THFile_readHalfRaw
  , c_THFile_writeHalfRaw
  , c_THFile_synchronize
  , c_THFile_seek
  , c_THFile_seekEnd
  , c_THFile_position
  , c_THFile_close
  , c_THFile_free
  , p_THFile_isOpened
  , p_THFile_isQuiet
  , p_THFile_isReadable
  , p_THFile_isWritable
  , p_THFile_isBinary
  , p_THFile_isAutoSpacing
  , p_THFile_hasError
  , p_THFile_binary
  , p_THFile_ascii
  , p_THFile_autoSpacing
  , p_THFile_noAutoSpacing
  , p_THFile_quiet
  , p_THFile_pedantic
  , p_THFile_clearError
  , p_THFile_readByteScalar
  , p_THFile_readCharScalar
  , p_THFile_readShortScalar
  , p_THFile_readIntScalar
  , p_THFile_readLongScalar
  , p_THFile_readFloatScalar
  , p_THFile_readDoubleScalar
  , p_THFile_writeByteScalar
  , p_THFile_writeCharScalar
  , p_THFile_writeShortScalar
  , p_THFile_writeIntScalar
  , p_THFile_writeLongScalar
  , p_THFile_writeFloatScalar
  , p_THFile_writeDoubleScalar
  , p_THFile_readByte
  , p_THFile_readChar
  , p_THFile_readShort
  , p_THFile_readInt
  , p_THFile_readLong
  , p_THFile_readFloat
  , p_THFile_readDouble
  , p_THFile_writeByte
  , p_THFile_writeChar
  , p_THFile_writeShort
  , p_THFile_writeInt
  , p_THFile_writeLong
  , p_THFile_writeFloat
  , p_THFile_writeDouble
  , p_THFile_readByteRaw
  , p_THFile_readCharRaw
  , p_THFile_readShortRaw
  , p_THFile_readIntRaw
  , p_THFile_readLongRaw
  , p_THFile_readFloatRaw
  , p_THFile_readDoubleRaw
  , p_THFile_readStringRaw
  , p_THFile_writeByteRaw
  , p_THFile_writeCharRaw
  , p_THFile_writeShortRaw
  , p_THFile_writeIntRaw
  , p_THFile_writeLongRaw
  , p_THFile_writeFloatRaw
  , p_THFile_writeDoubleRaw
  , p_THFile_writeStringRaw
  , p_THFile_readHalfScalar
  , p_THFile_writeHalfScalar
  , p_THFile_readHalf
  , p_THFile_writeHalf
  , p_THFile_readHalfRaw
  , p_THFile_writeHalfRaw
  , p_THFile_synchronize
  , p_THFile_seek
  , p_THFile_seekEnd
  , p_THFile_position
  , p_THFile_close
  , p_THFile_free
  ) where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_THFile_isOpened :  self -> int
foreign import ccall "THFile.h c_THFile_isOpened"
  c_THFile_isOpened :: Ptr (CTHFile) -> IO (CInt)

-- | c_THFile_isQuiet :  self -> int
foreign import ccall "THFile.h c_THFile_isQuiet"
  c_THFile_isQuiet :: Ptr (CTHFile) -> IO (CInt)

-- | c_THFile_isReadable :  self -> int
foreign import ccall "THFile.h c_THFile_isReadable"
  c_THFile_isReadable :: Ptr (CTHFile) -> IO (CInt)

-- | c_THFile_isWritable :  self -> int
foreign import ccall "THFile.h c_THFile_isWritable"
  c_THFile_isWritable :: Ptr (CTHFile) -> IO (CInt)

-- | c_THFile_isBinary :  self -> int
foreign import ccall "THFile.h c_THFile_isBinary"
  c_THFile_isBinary :: Ptr (CTHFile) -> IO (CInt)

-- | c_THFile_isAutoSpacing :  self -> int
foreign import ccall "THFile.h c_THFile_isAutoSpacing"
  c_THFile_isAutoSpacing :: Ptr (CTHFile) -> IO (CInt)

-- | c_THFile_hasError :  self -> int
foreign import ccall "THFile.h c_THFile_hasError"
  c_THFile_hasError :: Ptr (CTHFile) -> IO (CInt)

-- | c_THFile_binary :  self -> void
foreign import ccall "THFile.h c_THFile_binary"
  c_THFile_binary :: Ptr (CTHFile) -> IO (())

-- | c_THFile_ascii :  self -> void
foreign import ccall "THFile.h c_THFile_ascii"
  c_THFile_ascii :: Ptr (CTHFile) -> IO (())

-- | c_THFile_autoSpacing :  self -> void
foreign import ccall "THFile.h c_THFile_autoSpacing"
  c_THFile_autoSpacing :: Ptr (CTHFile) -> IO (())

-- | c_THFile_noAutoSpacing :  self -> void
foreign import ccall "THFile.h c_THFile_noAutoSpacing"
  c_THFile_noAutoSpacing :: Ptr (CTHFile) -> IO (())

-- | c_THFile_quiet :  self -> void
foreign import ccall "THFile.h c_THFile_quiet"
  c_THFile_quiet :: Ptr (CTHFile) -> IO (())

-- | c_THFile_pedantic :  self -> void
foreign import ccall "THFile.h c_THFile_pedantic"
  c_THFile_pedantic :: Ptr (CTHFile) -> IO (())

-- | c_THFile_clearError :  self -> void
foreign import ccall "THFile.h c_THFile_clearError"
  c_THFile_clearError :: Ptr (CTHFile) -> IO (())

-- | c_THFile_readByteScalar :  self -> int8_t
foreign import ccall "THFile.h c_THFile_readByteScalar"
  c_THFile_readByteScalar :: Ptr (CTHFile) -> IO (CSChar)

-- | c_THFile_readCharScalar :  self -> int8_t
foreign import ccall "THFile.h c_THFile_readCharScalar"
  c_THFile_readCharScalar :: Ptr (CTHFile) -> IO (CSChar)

-- | c_THFile_readShortScalar :  self -> int16_t
foreign import ccall "THFile.h c_THFile_readShortScalar"
  c_THFile_readShortScalar :: Ptr (CTHFile) -> IO (CShort)

-- | c_THFile_readIntScalar :  self -> int32_t
foreign import ccall "THFile.h c_THFile_readIntScalar"
  c_THFile_readIntScalar :: Ptr (CTHFile) -> IO (Int)

-- | c_THFile_readLongScalar :  self -> int64_t
foreign import ccall "THFile.h c_THFile_readLongScalar"
  c_THFile_readLongScalar :: Ptr (CTHFile) -> IO (CLLong)

-- | c_THFile_readFloatScalar :  self -> float
foreign import ccall "THFile.h c_THFile_readFloatScalar"
  c_THFile_readFloatScalar :: Ptr (CTHFile) -> IO (CFloat)

-- | c_THFile_readDoubleScalar :  self -> double
foreign import ccall "THFile.h c_THFile_readDoubleScalar"
  c_THFile_readDoubleScalar :: Ptr (CTHFile) -> IO (CDouble)

-- | c_THFile_writeByteScalar :  self scalar -> void
foreign import ccall "THFile.h c_THFile_writeByteScalar"
  c_THFile_writeByteScalar :: Ptr (CTHFile) -> CSChar -> IO (())

-- | c_THFile_writeCharScalar :  self scalar -> void
foreign import ccall "THFile.h c_THFile_writeCharScalar"
  c_THFile_writeCharScalar :: Ptr (CTHFile) -> CSChar -> IO (())

-- | c_THFile_writeShortScalar :  self scalar -> void
foreign import ccall "THFile.h c_THFile_writeShortScalar"
  c_THFile_writeShortScalar :: Ptr (CTHFile) -> CShort -> IO (())

-- | c_THFile_writeIntScalar :  self scalar -> void
foreign import ccall "THFile.h c_THFile_writeIntScalar"
  c_THFile_writeIntScalar :: Ptr (CTHFile) -> Int -> IO (())

-- | c_THFile_writeLongScalar :  self scalar -> void
foreign import ccall "THFile.h c_THFile_writeLongScalar"
  c_THFile_writeLongScalar :: Ptr (CTHFile) -> CLLong -> IO (())

-- | c_THFile_writeFloatScalar :  self scalar -> void
foreign import ccall "THFile.h c_THFile_writeFloatScalar"
  c_THFile_writeFloatScalar :: Ptr (CTHFile) -> CFloat -> IO (())

-- | c_THFile_writeDoubleScalar :  self scalar -> void
foreign import ccall "THFile.h c_THFile_writeDoubleScalar"
  c_THFile_writeDoubleScalar :: Ptr (CTHFile) -> CDouble -> IO (())

-- | c_THFile_readByte :  self storage -> size_t
foreign import ccall "THFile.h c_THFile_readByte"
  c_THFile_readByte :: Ptr (CTHFile) -> Ptr (CTHByteStorage) -> IO (CSize)

-- | c_THFile_readChar :  self storage -> size_t
foreign import ccall "THFile.h c_THFile_readChar"
  c_THFile_readChar :: Ptr (CTHFile) -> Ptr (CTHCharStorage) -> IO (CSize)

-- | c_THFile_readShort :  self storage -> size_t
foreign import ccall "THFile.h c_THFile_readShort"
  c_THFile_readShort :: Ptr (CTHFile) -> Ptr (CTHShortStorage) -> IO (CSize)

-- | c_THFile_readInt :  self storage -> size_t
foreign import ccall "THFile.h c_THFile_readInt"
  c_THFile_readInt :: Ptr (CTHFile) -> Ptr (CTHIntStorage) -> IO (CSize)

-- | c_THFile_readLong :  self storage -> size_t
foreign import ccall "THFile.h c_THFile_readLong"
  c_THFile_readLong :: Ptr (CTHFile) -> Ptr (CTHLongStorage) -> IO (CSize)

-- | c_THFile_readFloat :  self storage -> size_t
foreign import ccall "THFile.h c_THFile_readFloat"
  c_THFile_readFloat :: Ptr (CTHFile) -> Ptr (CTHFloatStorage) -> IO (CSize)

-- | c_THFile_readDouble :  self storage -> size_t
foreign import ccall "THFile.h c_THFile_readDouble"
  c_THFile_readDouble :: Ptr (CTHFile) -> Ptr (CTHDoubleStorage) -> IO (CSize)

-- | c_THFile_writeByte :  self storage -> size_t
foreign import ccall "THFile.h c_THFile_writeByte"
  c_THFile_writeByte :: Ptr (CTHFile) -> Ptr (CTHByteStorage) -> IO (CSize)

-- | c_THFile_writeChar :  self storage -> size_t
foreign import ccall "THFile.h c_THFile_writeChar"
  c_THFile_writeChar :: Ptr (CTHFile) -> Ptr (CTHCharStorage) -> IO (CSize)

-- | c_THFile_writeShort :  self storage -> size_t
foreign import ccall "THFile.h c_THFile_writeShort"
  c_THFile_writeShort :: Ptr (CTHFile) -> Ptr (CTHShortStorage) -> IO (CSize)

-- | c_THFile_writeInt :  self storage -> size_t
foreign import ccall "THFile.h c_THFile_writeInt"
  c_THFile_writeInt :: Ptr (CTHFile) -> Ptr (CTHIntStorage) -> IO (CSize)

-- | c_THFile_writeLong :  self storage -> size_t
foreign import ccall "THFile.h c_THFile_writeLong"
  c_THFile_writeLong :: Ptr (CTHFile) -> Ptr (CTHLongStorage) -> IO (CSize)

-- | c_THFile_writeFloat :  self storage -> size_t
foreign import ccall "THFile.h c_THFile_writeFloat"
  c_THFile_writeFloat :: Ptr (CTHFile) -> Ptr (CTHFloatStorage) -> IO (CSize)

-- | c_THFile_writeDouble :  self storage -> size_t
foreign import ccall "THFile.h c_THFile_writeDouble"
  c_THFile_writeDouble :: Ptr (CTHFile) -> Ptr (CTHDoubleStorage) -> IO (CSize)

-- | c_THFile_readByteRaw :  self data n -> size_t
foreign import ccall "THFile.h c_THFile_readByteRaw"
  c_THFile_readByteRaw :: Ptr (CTHFile) -> Ptr (CSChar) -> CSize -> IO (CSize)

-- | c_THFile_readCharRaw :  self data n -> size_t
foreign import ccall "THFile.h c_THFile_readCharRaw"
  c_THFile_readCharRaw :: Ptr (CTHFile) -> Ptr (CSChar) -> CSize -> IO (CSize)

-- | c_THFile_readShortRaw :  self data n -> size_t
foreign import ccall "THFile.h c_THFile_readShortRaw"
  c_THFile_readShortRaw :: Ptr (CTHFile) -> Ptr (CShort) -> CSize -> IO (CSize)

-- | c_THFile_readIntRaw :  self data n -> size_t
foreign import ccall "THFile.h c_THFile_readIntRaw"
  c_THFile_readIntRaw :: Ptr (CTHFile) -> Ptr (Int) -> CSize -> IO (CSize)

-- | c_THFile_readLongRaw :  self data n -> size_t
foreign import ccall "THFile.h c_THFile_readLongRaw"
  c_THFile_readLongRaw :: Ptr (CTHFile) -> Ptr (CLLong) -> CSize -> IO (CSize)

-- | c_THFile_readFloatRaw :  self data n -> size_t
foreign import ccall "THFile.h c_THFile_readFloatRaw"
  c_THFile_readFloatRaw :: Ptr (CTHFile) -> Ptr (CFloat) -> CSize -> IO (CSize)

-- | c_THFile_readDoubleRaw :  self data n -> size_t
foreign import ccall "THFile.h c_THFile_readDoubleRaw"
  c_THFile_readDoubleRaw :: Ptr (CTHFile) -> Ptr (CDouble) -> CSize -> IO (CSize)

-- | c_THFile_readStringRaw :  self format str_ -> size_t
foreign import ccall "THFile.h c_THFile_readStringRaw"
  c_THFile_readStringRaw :: Ptr (CTHFile) -> Ptr (CChar) -> Ptr (Ptr (CChar)) -> IO (CSize)

-- | c_THFile_writeByteRaw :  self data n -> size_t
foreign import ccall "THFile.h c_THFile_writeByteRaw"
  c_THFile_writeByteRaw :: Ptr (CTHFile) -> Ptr (CSChar) -> CSize -> IO (CSize)

-- | c_THFile_writeCharRaw :  self data n -> size_t
foreign import ccall "THFile.h c_THFile_writeCharRaw"
  c_THFile_writeCharRaw :: Ptr (CTHFile) -> Ptr (CSChar) -> CSize -> IO (CSize)

-- | c_THFile_writeShortRaw :  self data n -> size_t
foreign import ccall "THFile.h c_THFile_writeShortRaw"
  c_THFile_writeShortRaw :: Ptr (CTHFile) -> Ptr (CShort) -> CSize -> IO (CSize)

-- | c_THFile_writeIntRaw :  self data n -> size_t
foreign import ccall "THFile.h c_THFile_writeIntRaw"
  c_THFile_writeIntRaw :: Ptr (CTHFile) -> Ptr (Int) -> CSize -> IO (CSize)

-- | c_THFile_writeLongRaw :  self data n -> size_t
foreign import ccall "THFile.h c_THFile_writeLongRaw"
  c_THFile_writeLongRaw :: Ptr (CTHFile) -> Ptr (CLLong) -> CSize -> IO (CSize)

-- | c_THFile_writeFloatRaw :  self data n -> size_t
foreign import ccall "THFile.h c_THFile_writeFloatRaw"
  c_THFile_writeFloatRaw :: Ptr (CTHFile) -> Ptr (CFloat) -> CSize -> IO (CSize)

-- | c_THFile_writeDoubleRaw :  self data n -> size_t
foreign import ccall "THFile.h c_THFile_writeDoubleRaw"
  c_THFile_writeDoubleRaw :: Ptr (CTHFile) -> Ptr (CDouble) -> CSize -> IO (CSize)

-- | c_THFile_writeStringRaw :  self str size -> size_t
foreign import ccall "THFile.h c_THFile_writeStringRaw"
  c_THFile_writeStringRaw :: Ptr (CTHFile) -> Ptr (CChar) -> CSize -> IO (CSize)

-- | c_THFile_readHalfScalar :  self -> THHalf
foreign import ccall "THFile.h c_THFile_readHalfScalar"
  c_THFile_readHalfScalar :: Ptr (CTHFile) -> IO (CTHHalf)

-- | c_THFile_writeHalfScalar :  self scalar -> void
foreign import ccall "THFile.h c_THFile_writeHalfScalar"
  c_THFile_writeHalfScalar :: Ptr (CTHFile) -> CTHHalf -> IO (())

-- | c_THFile_readHalf :  self storage -> size_t
foreign import ccall "THFile.h c_THFile_readHalf"
  c_THFile_readHalf :: Ptr (CTHFile) -> Ptr (CTHHalfStorage) -> IO (CSize)

-- | c_THFile_writeHalf :  self storage -> size_t
foreign import ccall "THFile.h c_THFile_writeHalf"
  c_THFile_writeHalf :: Ptr (CTHFile) -> Ptr (CTHHalfStorage) -> IO (CSize)

-- | c_THFile_readHalfRaw :  self data size -> size_t
foreign import ccall "THFile.h c_THFile_readHalfRaw"
  c_THFile_readHalfRaw :: Ptr (CTHFile) -> Ptr CTHHalf -> CSize -> IO (CSize)

-- | c_THFile_writeHalfRaw :  self data size -> size_t
foreign import ccall "THFile.h c_THFile_writeHalfRaw"
  c_THFile_writeHalfRaw :: Ptr (CTHFile) -> Ptr CTHHalf -> CSize -> IO (CSize)

-- | c_THFile_synchronize :  self -> void
foreign import ccall "THFile.h c_THFile_synchronize"
  c_THFile_synchronize :: Ptr (CTHFile) -> IO (())

-- | c_THFile_seek :  self position -> void
foreign import ccall "THFile.h c_THFile_seek"
  c_THFile_seek :: Ptr (CTHFile) -> CSize -> IO (())

-- | c_THFile_seekEnd :  self -> void
foreign import ccall "THFile.h c_THFile_seekEnd"
  c_THFile_seekEnd :: Ptr (CTHFile) -> IO (())

-- | c_THFile_position :  self -> size_t
foreign import ccall "THFile.h c_THFile_position"
  c_THFile_position :: Ptr (CTHFile) -> IO (CSize)

-- | c_THFile_close :  self -> void
foreign import ccall "THFile.h c_THFile_close"
  c_THFile_close :: Ptr (CTHFile) -> IO (())

-- | c_THFile_free :  self -> void
foreign import ccall "THFile.h c_THFile_free"
  c_THFile_free :: Ptr (CTHFile) -> IO (())

-- | p_THFile_isOpened : Pointer to function : self -> int
foreign import ccall "THFile.h &p_THFile_isOpened"
  p_THFile_isOpened :: FunPtr (Ptr (CTHFile) -> IO (CInt))

-- | p_THFile_isQuiet : Pointer to function : self -> int
foreign import ccall "THFile.h &p_THFile_isQuiet"
  p_THFile_isQuiet :: FunPtr (Ptr (CTHFile) -> IO (CInt))

-- | p_THFile_isReadable : Pointer to function : self -> int
foreign import ccall "THFile.h &p_THFile_isReadable"
  p_THFile_isReadable :: FunPtr (Ptr (CTHFile) -> IO (CInt))

-- | p_THFile_isWritable : Pointer to function : self -> int
foreign import ccall "THFile.h &p_THFile_isWritable"
  p_THFile_isWritable :: FunPtr (Ptr (CTHFile) -> IO (CInt))

-- | p_THFile_isBinary : Pointer to function : self -> int
foreign import ccall "THFile.h &p_THFile_isBinary"
  p_THFile_isBinary :: FunPtr (Ptr (CTHFile) -> IO (CInt))

-- | p_THFile_isAutoSpacing : Pointer to function : self -> int
foreign import ccall "THFile.h &p_THFile_isAutoSpacing"
  p_THFile_isAutoSpacing :: FunPtr (Ptr (CTHFile) -> IO (CInt))

-- | p_THFile_hasError : Pointer to function : self -> int
foreign import ccall "THFile.h &p_THFile_hasError"
  p_THFile_hasError :: FunPtr (Ptr (CTHFile) -> IO (CInt))

-- | p_THFile_binary : Pointer to function : self -> void
foreign import ccall "THFile.h &p_THFile_binary"
  p_THFile_binary :: FunPtr (Ptr (CTHFile) -> IO (()))

-- | p_THFile_ascii : Pointer to function : self -> void
foreign import ccall "THFile.h &p_THFile_ascii"
  p_THFile_ascii :: FunPtr (Ptr (CTHFile) -> IO (()))

-- | p_THFile_autoSpacing : Pointer to function : self -> void
foreign import ccall "THFile.h &p_THFile_autoSpacing"
  p_THFile_autoSpacing :: FunPtr (Ptr (CTHFile) -> IO (()))

-- | p_THFile_noAutoSpacing : Pointer to function : self -> void
foreign import ccall "THFile.h &p_THFile_noAutoSpacing"
  p_THFile_noAutoSpacing :: FunPtr (Ptr (CTHFile) -> IO (()))

-- | p_THFile_quiet : Pointer to function : self -> void
foreign import ccall "THFile.h &p_THFile_quiet"
  p_THFile_quiet :: FunPtr (Ptr (CTHFile) -> IO (()))

-- | p_THFile_pedantic : Pointer to function : self -> void
foreign import ccall "THFile.h &p_THFile_pedantic"
  p_THFile_pedantic :: FunPtr (Ptr (CTHFile) -> IO (()))

-- | p_THFile_clearError : Pointer to function : self -> void
foreign import ccall "THFile.h &p_THFile_clearError"
  p_THFile_clearError :: FunPtr (Ptr (CTHFile) -> IO (()))

-- | p_THFile_readByteScalar : Pointer to function : self -> int8_t
foreign import ccall "THFile.h &p_THFile_readByteScalar"
  p_THFile_readByteScalar :: FunPtr (Ptr (CTHFile) -> IO (CSChar))

-- | p_THFile_readCharScalar : Pointer to function : self -> int8_t
foreign import ccall "THFile.h &p_THFile_readCharScalar"
  p_THFile_readCharScalar :: FunPtr (Ptr (CTHFile) -> IO (CSChar))

-- | p_THFile_readShortScalar : Pointer to function : self -> int16_t
foreign import ccall "THFile.h &p_THFile_readShortScalar"
  p_THFile_readShortScalar :: FunPtr (Ptr (CTHFile) -> IO (CShort))

-- | p_THFile_readIntScalar : Pointer to function : self -> int32_t
foreign import ccall "THFile.h &p_THFile_readIntScalar"
  p_THFile_readIntScalar :: FunPtr (Ptr (CTHFile) -> IO (Int))

-- | p_THFile_readLongScalar : Pointer to function : self -> int64_t
foreign import ccall "THFile.h &p_THFile_readLongScalar"
  p_THFile_readLongScalar :: FunPtr (Ptr (CTHFile) -> IO (CLLong))

-- | p_THFile_readFloatScalar : Pointer to function : self -> float
foreign import ccall "THFile.h &p_THFile_readFloatScalar"
  p_THFile_readFloatScalar :: FunPtr (Ptr (CTHFile) -> IO (CFloat))

-- | p_THFile_readDoubleScalar : Pointer to function : self -> double
foreign import ccall "THFile.h &p_THFile_readDoubleScalar"
  p_THFile_readDoubleScalar :: FunPtr (Ptr (CTHFile) -> IO (CDouble))

-- | p_THFile_writeByteScalar : Pointer to function : self scalar -> void
foreign import ccall "THFile.h &p_THFile_writeByteScalar"
  p_THFile_writeByteScalar :: FunPtr (Ptr (CTHFile) -> CSChar -> IO (()))

-- | p_THFile_writeCharScalar : Pointer to function : self scalar -> void
foreign import ccall "THFile.h &p_THFile_writeCharScalar"
  p_THFile_writeCharScalar :: FunPtr (Ptr (CTHFile) -> CSChar -> IO (()))

-- | p_THFile_writeShortScalar : Pointer to function : self scalar -> void
foreign import ccall "THFile.h &p_THFile_writeShortScalar"
  p_THFile_writeShortScalar :: FunPtr (Ptr (CTHFile) -> CShort -> IO (()))

-- | p_THFile_writeIntScalar : Pointer to function : self scalar -> void
foreign import ccall "THFile.h &p_THFile_writeIntScalar"
  p_THFile_writeIntScalar :: FunPtr (Ptr (CTHFile) -> Int -> IO (()))

-- | p_THFile_writeLongScalar : Pointer to function : self scalar -> void
foreign import ccall "THFile.h &p_THFile_writeLongScalar"
  p_THFile_writeLongScalar :: FunPtr (Ptr (CTHFile) -> CLLong -> IO (()))

-- | p_THFile_writeFloatScalar : Pointer to function : self scalar -> void
foreign import ccall "THFile.h &p_THFile_writeFloatScalar"
  p_THFile_writeFloatScalar :: FunPtr (Ptr (CTHFile) -> CFloat -> IO (()))

-- | p_THFile_writeDoubleScalar : Pointer to function : self scalar -> void
foreign import ccall "THFile.h &p_THFile_writeDoubleScalar"
  p_THFile_writeDoubleScalar :: FunPtr (Ptr (CTHFile) -> CDouble -> IO (()))

-- | p_THFile_readByte : Pointer to function : self storage -> size_t
foreign import ccall "THFile.h &p_THFile_readByte"
  p_THFile_readByte :: FunPtr (Ptr (CTHFile) -> Ptr (CTHByteStorage) -> IO (CSize))

-- | p_THFile_readChar : Pointer to function : self storage -> size_t
foreign import ccall "THFile.h &p_THFile_readChar"
  p_THFile_readChar :: FunPtr (Ptr (CTHFile) -> Ptr (CTHCharStorage) -> IO (CSize))

-- | p_THFile_readShort : Pointer to function : self storage -> size_t
foreign import ccall "THFile.h &p_THFile_readShort"
  p_THFile_readShort :: FunPtr (Ptr (CTHFile) -> Ptr (CTHShortStorage) -> IO (CSize))

-- | p_THFile_readInt : Pointer to function : self storage -> size_t
foreign import ccall "THFile.h &p_THFile_readInt"
  p_THFile_readInt :: FunPtr (Ptr (CTHFile) -> Ptr (CTHIntStorage) -> IO (CSize))

-- | p_THFile_readLong : Pointer to function : self storage -> size_t
foreign import ccall "THFile.h &p_THFile_readLong"
  p_THFile_readLong :: FunPtr (Ptr (CTHFile) -> Ptr (CTHLongStorage) -> IO (CSize))

-- | p_THFile_readFloat : Pointer to function : self storage -> size_t
foreign import ccall "THFile.h &p_THFile_readFloat"
  p_THFile_readFloat :: FunPtr (Ptr (CTHFile) -> Ptr (CTHFloatStorage) -> IO (CSize))

-- | p_THFile_readDouble : Pointer to function : self storage -> size_t
foreign import ccall "THFile.h &p_THFile_readDouble"
  p_THFile_readDouble :: FunPtr (Ptr (CTHFile) -> Ptr (CTHDoubleStorage) -> IO (CSize))

-- | p_THFile_writeByte : Pointer to function : self storage -> size_t
foreign import ccall "THFile.h &p_THFile_writeByte"
  p_THFile_writeByte :: FunPtr (Ptr (CTHFile) -> Ptr (CTHByteStorage) -> IO (CSize))

-- | p_THFile_writeChar : Pointer to function : self storage -> size_t
foreign import ccall "THFile.h &p_THFile_writeChar"
  p_THFile_writeChar :: FunPtr (Ptr (CTHFile) -> Ptr (CTHCharStorage) -> IO (CSize))

-- | p_THFile_writeShort : Pointer to function : self storage -> size_t
foreign import ccall "THFile.h &p_THFile_writeShort"
  p_THFile_writeShort :: FunPtr (Ptr (CTHFile) -> Ptr (CTHShortStorage) -> IO (CSize))

-- | p_THFile_writeInt : Pointer to function : self storage -> size_t
foreign import ccall "THFile.h &p_THFile_writeInt"
  p_THFile_writeInt :: FunPtr (Ptr (CTHFile) -> Ptr (CTHIntStorage) -> IO (CSize))

-- | p_THFile_writeLong : Pointer to function : self storage -> size_t
foreign import ccall "THFile.h &p_THFile_writeLong"
  p_THFile_writeLong :: FunPtr (Ptr (CTHFile) -> Ptr (CTHLongStorage) -> IO (CSize))

-- | p_THFile_writeFloat : Pointer to function : self storage -> size_t
foreign import ccall "THFile.h &p_THFile_writeFloat"
  p_THFile_writeFloat :: FunPtr (Ptr (CTHFile) -> Ptr (CTHFloatStorage) -> IO (CSize))

-- | p_THFile_writeDouble : Pointer to function : self storage -> size_t
foreign import ccall "THFile.h &p_THFile_writeDouble"
  p_THFile_writeDouble :: FunPtr (Ptr (CTHFile) -> Ptr (CTHDoubleStorage) -> IO (CSize))

-- | p_THFile_readByteRaw : Pointer to function : self data n -> size_t
foreign import ccall "THFile.h &p_THFile_readByteRaw"
  p_THFile_readByteRaw :: FunPtr (Ptr (CTHFile) -> Ptr (CSChar) -> CSize -> IO (CSize))

-- | p_THFile_readCharRaw : Pointer to function : self data n -> size_t
foreign import ccall "THFile.h &p_THFile_readCharRaw"
  p_THFile_readCharRaw :: FunPtr (Ptr (CTHFile) -> Ptr (CSChar) -> CSize -> IO (CSize))

-- | p_THFile_readShortRaw : Pointer to function : self data n -> size_t
foreign import ccall "THFile.h &p_THFile_readShortRaw"
  p_THFile_readShortRaw :: FunPtr (Ptr (CTHFile) -> Ptr (CShort) -> CSize -> IO (CSize))

-- | p_THFile_readIntRaw : Pointer to function : self data n -> size_t
foreign import ccall "THFile.h &p_THFile_readIntRaw"
  p_THFile_readIntRaw :: FunPtr (Ptr (CTHFile) -> Ptr (Int) -> CSize -> IO (CSize))

-- | p_THFile_readLongRaw : Pointer to function : self data n -> size_t
foreign import ccall "THFile.h &p_THFile_readLongRaw"
  p_THFile_readLongRaw :: FunPtr (Ptr (CTHFile) -> Ptr (CLLong) -> CSize -> IO (CSize))

-- | p_THFile_readFloatRaw : Pointer to function : self data n -> size_t
foreign import ccall "THFile.h &p_THFile_readFloatRaw"
  p_THFile_readFloatRaw :: FunPtr (Ptr (CTHFile) -> Ptr (CFloat) -> CSize -> IO (CSize))

-- | p_THFile_readDoubleRaw : Pointer to function : self data n -> size_t
foreign import ccall "THFile.h &p_THFile_readDoubleRaw"
  p_THFile_readDoubleRaw :: FunPtr (Ptr (CTHFile) -> Ptr (CDouble) -> CSize -> IO (CSize))

-- | p_THFile_readStringRaw : Pointer to function : self format str_ -> size_t
foreign import ccall "THFile.h &p_THFile_readStringRaw"
  p_THFile_readStringRaw :: FunPtr (Ptr (CTHFile) -> Ptr (CChar) -> Ptr (Ptr (CChar)) -> IO (CSize))

-- | p_THFile_writeByteRaw : Pointer to function : self data n -> size_t
foreign import ccall "THFile.h &p_THFile_writeByteRaw"
  p_THFile_writeByteRaw :: FunPtr (Ptr (CTHFile) -> Ptr (CSChar) -> CSize -> IO (CSize))

-- | p_THFile_writeCharRaw : Pointer to function : self data n -> size_t
foreign import ccall "THFile.h &p_THFile_writeCharRaw"
  p_THFile_writeCharRaw :: FunPtr (Ptr (CTHFile) -> Ptr (CSChar) -> CSize -> IO (CSize))

-- | p_THFile_writeShortRaw : Pointer to function : self data n -> size_t
foreign import ccall "THFile.h &p_THFile_writeShortRaw"
  p_THFile_writeShortRaw :: FunPtr (Ptr (CTHFile) -> Ptr (CShort) -> CSize -> IO (CSize))

-- | p_THFile_writeIntRaw : Pointer to function : self data n -> size_t
foreign import ccall "THFile.h &p_THFile_writeIntRaw"
  p_THFile_writeIntRaw :: FunPtr (Ptr (CTHFile) -> Ptr (Int) -> CSize -> IO (CSize))

-- | p_THFile_writeLongRaw : Pointer to function : self data n -> size_t
foreign import ccall "THFile.h &p_THFile_writeLongRaw"
  p_THFile_writeLongRaw :: FunPtr (Ptr (CTHFile) -> Ptr (CLLong) -> CSize -> IO (CSize))

-- | p_THFile_writeFloatRaw : Pointer to function : self data n -> size_t
foreign import ccall "THFile.h &p_THFile_writeFloatRaw"
  p_THFile_writeFloatRaw :: FunPtr (Ptr (CTHFile) -> Ptr (CFloat) -> CSize -> IO (CSize))

-- | p_THFile_writeDoubleRaw : Pointer to function : self data n -> size_t
foreign import ccall "THFile.h &p_THFile_writeDoubleRaw"
  p_THFile_writeDoubleRaw :: FunPtr (Ptr (CTHFile) -> Ptr (CDouble) -> CSize -> IO (CSize))

-- | p_THFile_writeStringRaw : Pointer to function : self str size -> size_t
foreign import ccall "THFile.h &p_THFile_writeStringRaw"
  p_THFile_writeStringRaw :: FunPtr (Ptr (CTHFile) -> Ptr (CChar) -> CSize -> IO (CSize))

-- | p_THFile_readHalfScalar : Pointer to function : self -> THHalf
foreign import ccall "THFile.h &p_THFile_readHalfScalar"
  p_THFile_readHalfScalar :: FunPtr (Ptr (CTHFile) -> IO (CTHHalf))

-- | p_THFile_writeHalfScalar : Pointer to function : self scalar -> void
foreign import ccall "THFile.h &p_THFile_writeHalfScalar"
  p_THFile_writeHalfScalar :: FunPtr (Ptr (CTHFile) -> CTHHalf -> IO (()))

-- | p_THFile_readHalf : Pointer to function : self storage -> size_t
foreign import ccall "THFile.h &p_THFile_readHalf"
  p_THFile_readHalf :: FunPtr (Ptr (CTHFile) -> Ptr (CTHHalfStorage) -> IO (CSize))

-- | p_THFile_writeHalf : Pointer to function : self storage -> size_t
foreign import ccall "THFile.h &p_THFile_writeHalf"
  p_THFile_writeHalf :: FunPtr (Ptr (CTHFile) -> Ptr (CTHHalfStorage) -> IO (CSize))

-- | p_THFile_readHalfRaw : Pointer to function : self data size -> size_t
foreign import ccall "THFile.h &p_THFile_readHalfRaw"
  p_THFile_readHalfRaw :: FunPtr (Ptr (CTHFile) -> Ptr CTHHalf -> CSize -> IO (CSize))

-- | p_THFile_writeHalfRaw : Pointer to function : self data size -> size_t
foreign import ccall "THFile.h &p_THFile_writeHalfRaw"
  p_THFile_writeHalfRaw :: FunPtr (Ptr (CTHFile) -> Ptr CTHHalf -> CSize -> IO (CSize))

-- | p_THFile_synchronize : Pointer to function : self -> void
foreign import ccall "THFile.h &p_THFile_synchronize"
  p_THFile_synchronize :: FunPtr (Ptr (CTHFile) -> IO (()))

-- | p_THFile_seek : Pointer to function : self position -> void
foreign import ccall "THFile.h &p_THFile_seek"
  p_THFile_seek :: FunPtr (Ptr (CTHFile) -> CSize -> IO (()))

-- | p_THFile_seekEnd : Pointer to function : self -> void
foreign import ccall "THFile.h &p_THFile_seekEnd"
  p_THFile_seekEnd :: FunPtr (Ptr (CTHFile) -> IO (()))

-- | p_THFile_position : Pointer to function : self -> size_t
foreign import ccall "THFile.h &p_THFile_position"
  p_THFile_position :: FunPtr (Ptr (CTHFile) -> IO (CSize))

-- | p_THFile_close : Pointer to function : self -> void
foreign import ccall "THFile.h &p_THFile_close"
  p_THFile_close :: FunPtr (Ptr (CTHFile) -> IO (()))

-- | p_THFile_free : Pointer to function : self -> void
foreign import ccall "THFile.h &p_THFile_free"
  p_THFile_free :: FunPtr (Ptr (CTHFile) -> IO (()))