{-# OPTIONS_GHC -fno-cse -fno-full-laziness #-}
{-# LANGUAGE InstanceSigs, RankNTypes, PolyKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeInType #-}
module Torch.Core.Tensor.Dynamic.Long
  ( TensorLong(..)
  , tl_get
  , tl_new
  , fillRaw
  , fillRaw0
  , wrapLong
  ) where

import Control.Monad (void)
import Foreign.C.Types
import Foreign (Ptr, ForeignPtr, withForeignPtr, newForeignPtr, finalizeForeignPtr)
import GHC.TypeLits (Nat)
import GHC.Exts (fromList, toList, IsList, Item)

import Torch.Core.Tensor.Dim (Dim(..), SomeDims(..), someDimsM)
import Torch.Raw.Internal (CTHDoubleTensor, CTHLongTensor)
import qualified THDoubleTensor as T
import qualified THLongTensor as T
import qualified Torch.Raw.Tensor.Generic as GenRaw
import qualified Torch.Core.Tensor.Dynamic.Generic as Gen
import qualified Torch.Core.Tensor.Dim as Dim

import GHC.Ptr (FunPtr)
import Numeric (showGFloat)
import System.IO.Unsafe (unsafePerformIO)

import Torch.Core.Internal (w2cll)
import Torch.Core.Tensor.Types (TensorLong(..), TensorLongRaw, THForeignRef(getForeign))

import THTypes
import THLongTensor
import THLongTensorMath
import THLongLapack

wrapLong :: TensorLongRaw -> IO TensorLong
wrapLong = fmap TensorLong . newForeignPtr GenRaw.p_free

tl_get :: SomeDims -> TensorLong -> IO CLong
tl_get loc t = withForeignPtr (getForeign t) (pure . flip GenRaw.genericGet' loc)

-- | Returns a function that accepts a tensor and fills it with specified value
-- and returns the IO context with the mutated tensor
fillRaw :: Integral a => a -> TensorLongRaw -> IO ()
fillRaw value = flip GenRaw.c_fill (fromIntegral value)

-- | Fill a raw Long tensor with 0.0
fillRaw0 :: TensorLongRaw -> IO (TensorLongRaw)
fillRaw0 t = fillRaw 0 t >> pure t

-- | Create a new (Long) tensor of specified dimensions and fill it with 0
tl_new :: SomeDims -> TensorLong
tl_new dims = unsafePerformIO $ do
  newPtr <- GenRaw.genericNew' dims
  fPtr <- newForeignPtr GenRaw.p_free newPtr
  withForeignPtr fPtr fillRaw0
  pure $ TensorLong fPtr
{-# NOINLINE tl_new #-}

