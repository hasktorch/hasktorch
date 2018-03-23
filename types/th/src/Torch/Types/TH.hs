module Torch.Types.TH
  ( CTHDescBuff
  , CTHAllocatorPtr
  , CTHAllocator
  , CTHGenerator
  , CTHFile, C'THFile
  , CTHHalf, C'THHalf
  , THLongBlas
  , THShortBlas
  , THIntBlas
  , THByteBlas
  , THHalfBlas
  , THFloatBlas
  , THDoubleBlas
  , THLongLapack
  , THShortLapack
  , THIntLapack
  , THByteLapack
  , THHalfLapack
  , THFloatLapack
  , THDoubleLapack
  , CTHLongVector
  , CTHShortVector
  , CTHIntVector
  , CTHByteVector
  , CTHHalfVector
  , CTHFloatVector
  , CTHDoubleVector
  , CTHNNState, C'THNNState
  , CTHIndexTensor, C'THIndexTensor
  , CTHIntegerTensor, C'THIntegerTensor
  , CTHByteTensor
  , CTHByteStorage
  , CTHByteStorageCopy
  , CTHCharTensor
  , CTHCharStorage
  , CTHCharStorageCopy
  , CTHDoubleTensor
  , CTHDoubleStorage
  , CTHDoubleStorageCopy
  , CTHFloatTensor
  , CTHFloatStorage
  , CTHFloatStorageCopy
  , C'THHalfTensor
  , C'THHalfStorage
  , C'THHalfStorageCopy
  , CTHHalfTensor
  , CTHHalfStorage
  , CTHHalfStorageCopy
  , CTHIntTensor
  , CTHIntStorage
  , CTHIntStorageCopy
  , CTHLongTensor
  , CTHLongStorage
  , CTHLongStorageCopy
  , CTHShortTensor
  , CTHShortStorage
  , CTHShortStorageCopy

  , module Torch.Types.TH.Structs
  ) where

import Foreign
import Foreign.C.String ()
import Foreign.C.Types
import Foreign.Ptr ()
import Foreign.Storable ()

import Torch.Types.TH.Structs

type CTHDescBuff = Ptr C'THDescBuff
type CTHAllocatorPtr = Ptr C'THAllocator
type CTHAllocator = C'THAllocator
type CTHGenerator = C'THGenerator
type CTHFile = ()
type C'THFile = ()
type CTHHalf = Ptr ()
type C'THHalf = Ptr ()

type THLongBlas = CLong
type THShortBlas = CShort
type THIntBlas = CInt
type THByteBlas = CChar
type THHalfBlas = CShort
type THFloatBlas = CFloat
type THDoubleBlas = CDouble

type THLongLapack = CLong
type THShortLapack = CShort
type THIntLapack = CInt
type THByteLapack = CChar
type THHalfLapack = CShort
type THFloatLapack = CFloat
type THDoubleLapack = CDouble

type CTHLongVector = CLong
type CTHShortVector = CShort
type CTHIntVector = CInt
type CTHByteVector = CChar
type CTHHalfVector = CShort
type CTHFloatVector = CFloat
type CTHDoubleVector = CDouble

-- in THNN.h
-- typedef void THNNState
type CTHNNState = Ptr ()
type C'THNNState = CTHNNState

-- typedef int64_t THIndex_t;
-- typedef int32_t THInteger_t;
type CTHIndexTensor = CLong
type CTHIntegerTensor = CInt
type C'THIndexTensor = CLong
type C'THIntegerTensor = CInt
-- ----------------------------------------
-- Templated types
-- ----------------------------------------

{- Byte -}

type CTHByteTensor = C'THByteTensor
type CTHByteStorage = C'THByteStorage
type CTHByteStorageCopy = CTHByteStorage


{- Char -}

type CTHCharTensor = C'THCharTensor
type CTHCharStorage = C'THCharStorage
type CTHCharStorageCopy = CTHCharStorage

{- Double -}

type CTHDoubleTensor = C'THDoubleTensor
type CTHDoubleStorage = C'THDoubleStorage
type CTHDoubleStorageCopy = CTHDoubleStorage

{- Float -}

type CTHFloatTensor = C'THFloatTensor
type CTHFloatStorage = C'THFloatStorage
type CTHFloatStorageCopy = CTHFloatStorage

{- Half -}

type C'THHalfTensor = ()
type C'THHalfStorage = ()
type C'THHalfStorageCopy = ()
type CTHHalfTensor = ()
type CTHHalfStorage = ()
type CTHHalfStorageCopy = ()

{- Int -}

type CTHIntTensor = C'THIntTensor
type CTHIntStorage = C'THIntStorage
type CTHIntStorageCopy = CTHIntStorage

{- Long -}

type CTHLongTensor = C'THLongTensor
type CTHLongStorage = C'THLongStorage
type CTHLongStorageCopy = CTHLongStorage

{- Short -}

type CTHShortTensor = C'THShortTensor
type CTHShortStorage = C'THShortStorage
type CTHShortStorageCopy = CTHShortStorage
