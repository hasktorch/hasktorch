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
import Torch.Core.Tensor.Generic.Internal (CTHDoubleTensor, CTHFloatTensor)
import Torch.Core.Tensor.Dynamic.Generic.Internal (HaskReal', LapackOps', GenericOps', GenericMath', THTensor(..), THPtrType)
import qualified THDoubleTensor as T
import qualified THLongTensor as T
import qualified Torch.Core.Tensor.Generic as GenRaw
import qualified Torch.Core.Tensor.Dim as Dim
import qualified Torch.Core.Tensor.Dynamic.Generic as Gen

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

td_gesv_ :: LapackOps' t => t -> t -> t -> t -> IO ()
td_gesv_ = Gen.withManaged4 GenRaw.c_gesv

td_gels
  :: forall t . (Num (HaskReal' t), LapackOps' t, GenericMath' t)
  => t -> t -> (t, t)
td_gels a b = unsafePerformIO $ do
  td_gels_ resA resB a b
  pure (resA, resB)
 where
  dimA :: SomeDims
  dimA = Gen.genericDynamicDims a

  resA, resB :: t
  resA = Gen.genericNew' dimA
  resB = Gen.genericNew' dimA
{-# NOINLINE td_gels #-}

td_gels_ :: LapackOps' t => t -> t -> t -> t -> IO ()
td_gels_ = Gen.withManaged4 GenRaw.c_gels

td_getri :: forall t . (Num (HaskReal' t), LapackOps' t, GenericOps' t, GenericMath' t) => t -> t
td_getri a = unsafePerformIO $ do
  let resA = Gen.genericNew' (Gen.genericDynamicDims a) :: t
  td_getri_ resA a
  pure resA
{-# NOINLINE td_getri #-}


td_getri_ :: forall t . LapackOps' t => t -> t -> IO ()
td_getri_ = Gen.withManaged2 GenRaw.c_getri

td_qr
  :: forall t . (Num (HaskReal' t), LapackOps' t, GenericMath' t)
  => t -> (t, t)
td_qr a = unsafePerformIO $ do
  td_qr_ resQ resR a
  pure (resQ, resR)
 where
  dimA :: SomeDims
  dimA = Gen.genericDynamicDims a

  resR, resQ :: t
  resR = Gen.genericNew' dimA
  resQ = Gen.genericNew' dimA
{-# NOINLINE td_qr #-}

td_qr_ :: forall t . LapackOps' t => t -> t -> t -> IO ()
td_qr_ resQ resR a = Gen.withManaged3 GenRaw.c_qr resQ resR a

td_gesvd = undefined

td_gesvd2 = undefined
