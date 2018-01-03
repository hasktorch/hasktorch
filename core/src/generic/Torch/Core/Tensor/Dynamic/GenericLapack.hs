{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
module Torch.Core.Tensor.Dynamic.GenericLapack where

import Control.Monad.IO.Class (liftIO)
import Control.Monad.Managed (managed, runManaged)
import Control.Monad (void)
import Foreign.C.Types
import Foreign (Ptr, ForeignPtr, withForeignPtr, newForeignPtr, finalizeForeignPtr)
import GHC.TypeLits (Nat)
import GHC.Exts (fromList, toList, IsList, Item)
import System.IO.Unsafe (unsafePerformIO)

import Torch.Core.Tensor.Dim (Dim(..), SomeDims(..), someDimsM)
import Torch.Core.Tensor.Generic.Internal (CTHDoubleTensor, CTHFloatTensor, HaskReal)
import Torch.Core.Tensor.Dynamic.Generic.Internal (LapackOps')
import qualified THDoubleTensor as T
import qualified THLongTensor as T
import qualified Torch.Core.Tensor.Generic as GenRaw
import qualified Torch.Core.Tensor.Dynamic.Generic as Gen
import qualified Torch.Core.Tensor.Dim as Dim

import qualified Torch.Core.Tensor.Dynamic.Generic as Gen
import Torch.Core.Tensor.Dynamic.Generic.Internal

td_gesv
  :: forall t . (Num (HaskReal' t), LapackOps' t, GenericMath' t)
  => t -> t -> (t, t)
td_gesv a b = unsafePerformIO $ do
  td_gesv_ resA resB a b
  pure (resA, resB)
 where
  dimA :: SomeDims
  dimA = Gen.genericDynamicDims a

  resA, resB :: t
  resA = Gen.genericNew' dimA
  resB = Gen.genericNew' dimA
{-# NOINLINE td_gesv #-}

td_gesv_
  :: LapackOps' t => t -> t -> t -> t -> IO ()
td_gesv_ resA resB a b = runManaged $ do
  resBRaw <- managed (withForeignPtr (getForeign resB))
  resARaw <- managed (withForeignPtr (getForeign resA))
  bRaw <- managed (withForeignPtr (getForeign b))
  aRaw <- managed (withForeignPtr (getForeign a))
  liftIO (GenRaw.c_gesv resBRaw resARaw bRaw aRaw)

{-
td_gels :: TensorDouble -> TensorDouble -> (TensorDouble, TensorDouble)
td_gels a b = unsafePerformIO $ do
  let resB = td_new (tdDim a)
  let resA = td_new (tdDim a)
  td_gels_ resA resB a b
  pure (resA, resB)

td_gels_
  :: TensorDouble -> TensorDouble -> TensorDouble -> TensorDouble -> IO ()
td_gels_ resA resB a b = runManaged $ do
  resBRaw <- managed (withForeignPtr (tdTensor resB))
  resARaw <- managed (withForeignPtr (tdTensor resA))
  bRaw <- managed (withForeignPtr (tdTensor b))
  aRaw <- managed (withForeignPtr (tdTensor a))
  liftIO (c_THDoubleTensor_gels resBRaw resARaw bRaw aRaw)

td_getri :: TensorDouble -> TensorDouble
td_getri a = unsafePerformIO $ do
  let resA = td_new (tdDim a)
  td_getri_ resA a
  pure resA

td_getri_ :: TensorDouble -> TensorDouble -> IO ()
td_getri_ resA a = runManaged $ do
  resARaw <- managed (withForeignPtr (tdTensor resA))
  aRaw <- managed (withForeignPtr (tdTensor a))
  liftIO (c_THDoubleTensor_getri resARaw aRaw)

td_qr :: TensorDouble -> (TensorDouble, TensorDouble)
td_qr a = unsafePerformIO $ do
  let resQ = td_new (tdDim a)
  let resR = td_new (tdDim a)
  td_qr_ resQ resR a
  pure (resQ, resR)

td_qr_ :: TensorDouble -> TensorDouble -> TensorDouble -> IO ()
td_qr_ resQ resR a = runManaged $ do
  resQRaw <- managed (withForeignPtr (tdTensor resQ))
  resRRaw <- managed (withForeignPtr (tdTensor resR))
  aRaw <- managed (withForeignPtr (tdTensor a))
  liftIO (c_THDoubleTensor_qr resQRaw resRRaw aRaw)

td_gesvd = undefined

td_gesvd2 = undefined

-}
