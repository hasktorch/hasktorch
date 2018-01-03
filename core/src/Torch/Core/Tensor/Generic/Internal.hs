{-# LANGUAGE TypeFamilies #-}
module Torch.Core.Tensor.Generic.Internal
  ( module X
  , HaskReal
  , HaskAccReal
  , Storage
  ) where

import Foreign as X (Ptr, FunPtr)
import Foreign.C.Types as X (CPtrdiff, CLLong, CLong, CDouble, CShort, CLong, CChar, CInt, CFloat)
import THTypes as X

-- | the "real" type of the bytetensor -- notation is taken from TH.
type family HaskReal t
type instance HaskReal CTHByteTensor = CChar
type instance HaskReal CTHDoubleTensor = CDouble
type instance HaskReal CTHFloatTensor = CFloat
type instance HaskReal CTHIntTensor = CInt
type instance HaskReal CTHLongTensor = CLong
type instance HaskReal CTHShortTensor = CShort

-- | the "accreal" type of the bytetensor -- notation is taken from TH.
type family HaskAccReal t
type instance HaskAccReal CTHByteTensor = CLong
type instance HaskAccReal CTHDoubleTensor = CDouble
type instance HaskAccReal CTHFloatTensor = CDouble
type instance HaskAccReal CTHIntTensor = CLong
type instance HaskAccReal CTHLongTensor = CLong
type instance HaskAccReal CTHShortTensor = CLong

-- | The corresponding storage backend for a CTH tensor
type family Storage t
type instance Storage CTHByteTensor = CTHByteStorage
type instance Storage CTHDoubleTensor = CTHDoubleStorage
type instance Storage CTHFloatTensor = CTHFloatStorage
type instance Storage CTHIntTensor = CTHIntStorage
type instance Storage CTHLongTensor = CTHLongStorage
type instance Storage CTHShortTensor = CTHShortStorage


