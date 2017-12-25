{-# LANGUAGE TypeFamilies #-}
module Torch.Core.Tensor.Generic.Internal
  ( module X
  , HaskType
  , Storage
  ) where

import Foreign as X (Ptr)
import Foreign.C.Types as X (CLLong, CLong, CDouble, CShort, CLong, CChar, CInt, CFloat)
import THTypes as X

type family HaskType t
type instance HaskType CTHByteTensor = CChar
type instance HaskType CTHDoubleTensor = CDouble
type instance HaskType CTHFloatTensor = CFloat
type instance HaskType CTHIntTensor = CInt
type instance HaskType CTHLongTensor = CLong
type instance HaskType CTHShortTensor = CShort


type family Storage t
type instance Storage CTHByteTensor = CTHByteStorage
type instance Storage CTHDoubleTensor = CTHDoubleStorage
type instance Storage CTHFloatTensor = CTHFloatStorage
type instance Storage CTHIntTensor = CTHIntStorage
type instance Storage CTHLongTensor = CTHLongStorage
type instance Storage CTHShortTensor = CTHShortStorage


