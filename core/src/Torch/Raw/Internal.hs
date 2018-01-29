{-# LANGUAGE TypeFamilies #-}
module Torch.Raw.Internal
  ( HaskReal
  , HaskAccReal
  , Storage

  , module X
  ) where

import Foreign as X (Ptr, FunPtr)
import Foreign.C.Types as X (CPtrdiff, CLLong, CLong, CDouble, CUShort, CShort, CLong, CChar, CInt, CIntPtr, CFloat)
import THTypes as X

-- | the "real" type of the bytetensor -- notation is taken from TH.
type family HaskReal t
type instance HaskReal CTHByteTensor = CChar
type instance HaskReal CTHDoubleTensor = CDouble
type instance HaskReal CTHFloatTensor = CFloat
type instance HaskReal CTHIntTensor = CInt
type instance HaskReal CTHLongTensor = CLong
type instance HaskReal CTHShortTensor = CShort

type instance HaskReal CTHByteVector = CChar
type instance HaskReal CTHDoubleVector = CDouble
type instance HaskReal CTHFloatVector = CFloat
type instance HaskReal CTHIntVector = CInt
type instance HaskReal CTHLongVector = CLong
type instance HaskReal CTHShortVector = CShort

type instance HaskReal CTHByteStorage = CChar
type instance HaskReal CTHDoubleStorage = CDouble
type instance HaskReal CTHFloatStorage = CFloat
type instance HaskReal CTHIntStorage = CInt
type instance HaskReal CTHLongStorage = CLong
type instance HaskReal CTHShortStorage = CShort
type instance HaskReal CTHHalfStorage = CUShort

type instance HaskReal CTHByteStorageCopy = CChar
type instance HaskReal CTHDoubleStorageCopy = CDouble
type instance HaskReal CTHFloatStorageCopy = CFloat
type instance HaskReal CTHIntStorageCopy = CInt
type instance HaskReal CTHLongStorageCopy = CLong
type instance HaskReal CTHShortStorageCopy = CShort
type instance HaskReal CTHHalfStorageCopy = CUShort

-- | the "accreal" type of the bytetensor -- notation is taken from TH.
type family HaskAccReal t
type instance HaskAccReal CTHByteTensor = CLong
type instance HaskAccReal CTHDoubleTensor = CDouble
type instance HaskAccReal CTHFloatTensor = CDouble
type instance HaskAccReal CTHIntTensor = CLong
type instance HaskAccReal CTHLongTensor = CLong
type instance HaskAccReal CTHShortTensor = CLong

type instance HaskAccReal CTHByteVector = CLong
type instance HaskAccReal CTHDoubleVector = CDouble
type instance HaskAccReal CTHFloatVector = CDouble
type instance HaskAccReal CTHIntVector = CLong
type instance HaskAccReal CTHLongVector = CLong
type instance HaskAccReal CTHShortVector = CLong


-- | The corresponding storage backend for a CTH tensor
type family Storage t
type instance Storage CTHByteTensor = CTHByteStorage
type instance Storage CTHDoubleTensor = CTHDoubleStorage
type instance Storage CTHFloatTensor = CTHFloatStorage
type instance Storage CTHIntTensor = CTHIntStorage
type instance Storage CTHLongTensor = CTHLongStorage
type instance Storage CTHShortTensor = CTHShortStorage

type instance Storage CTHByteVector = CTHByteStorage
type instance Storage CTHDoubleVector = CTHDoubleStorage
type instance Storage CTHFloatVector = CTHFloatStorage
type instance Storage CTHIntVector = CTHIntStorage
type instance Storage CTHLongVector = CTHLongStorage
type instance Storage CTHShortVector = CTHShortStorage

