module Torch.Core.Tensor.Dynamic
  ( Tensor
  , tensor
  , DynTensor
  , asTensor
  , Class.IsTensor(..)
  , Class.TensorCopy(..)
  , Class.TensorConv(..)
  , Class.TensorMath(..)
  , Class.TensorRandom(..)
  ) where

import Data.Coerce (coerce)
import Foreign (Ptr, withForeignPtr, newForeignPtr, Storable(peek))
import Foreign.C.Types
import GHC.ForeignPtr (ForeignPtr)
import GHC.Int
import THTypes

import qualified Foreign.Marshal.Array as FM
import qualified Tensor as Sig
import qualified Storage as StorageSig (c_size)
import qualified Torch.Class.C.Tensor as Class
import qualified Torch.Class.C.IsTensor as Class
import qualified Torch.Class.C.Tensor.Copy as Class
import qualified Torch.Class.C.Tensor.Conv as Class
import qualified Torch.Class.C.Tensor.Math as Class
import qualified Torch.Class.C.Tensor.Random as Class

import Torch.Core.Types
import Torch.Core.Storage (asStorageM)
import Torch.Core.Tensor.Dynamic.IsTensor as X
import Torch.Core.Tensor.Dynamic.Copy as X
import Torch.Core.Tensor.Dynamic.Conv as X
import Torch.Core.Tensor.Dynamic.Math as X
import Torch.Core.Tensor.Dynamic.Random as X


