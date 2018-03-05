{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
module Torch.Indef.Tensor.Dynamic.GenericLapack where

import Control.Monad.IO.Class (liftIO)
import Control.Monad.Managed (managed, runManaged)
import Control.Monad (void)
import Foreign.C.Types
import Foreign (Ptr, ForeignPtr, withForeignPtr, newForeignPtr, finalizeForeignPtr)
import GHC.TypeLits (Nat)
import GHC.Exts (fromList, toList, IsList, Item)
import System.IO.Unsafe (unsafePerformIO)

import Torch.Dimensions (Dim(..), SomeDims(..), someDimsM)
import Torch.Raw.Tensor.Generic (CTorch.FFI.TH.Double.Tensor, CTorch.FFI.TH.Float.Tensor, THTensor(..))
import Torch.Indef.Tensor.Dynamic.Generic (HaskReal', THTensorLapack', THTensor', THTensorMath')
import Torch.Indef.Tensor.Types (THForeignType, withManaged4, withManaged3, withManaged2)
import qualified Torch.FFI.TH.Double.Tensor as T
import qualified Torch.FFI.TH.Long.Tensor as T
import qualified Torch.Raw.Tensor.Generic as GenRaw
import qualified Torch.Dimensions as Dim
import qualified Torch.Indef.Tensor.Dynamic.Generic as Gen

td_gesv
  :: forall t . (Num (HaskReal' t), THTensorLapack' t, THTensorMath' t)
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

td_gesv_ :: THTensorLapack' t => t -> t -> t -> t -> IO ()
td_gesv_ = withManaged4 GenRaw.c_gesv

td_gels
  :: forall t . (Num (HaskReal' t), THTensorLapack' t, THTensorMath' t)
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

td_gels_ :: THTensorLapack' t => t -> t -> t -> t -> IO ()
td_gels_ = withManaged4 GenRaw.c_gels

td_getri :: forall t . (Num (HaskReal' t), THTensorLapack' t, THTensor' t, THTensorMath' t) => t -> t
td_getri a = unsafePerformIO $ do
  let resA = Gen.genericNew' (Gen.genericDynamicDims a) :: t
  td_getri_ resA a
  pure resA
{-# NOINLINE td_getri #-}


td_getri_ :: forall t . THTensorLapack' t => t -> t -> IO ()
td_getri_ = withManaged2 GenRaw.c_getri

td_qr
  :: forall t . (Num (HaskReal' t), THTensorLapack' t, THTensorMath' t)
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

td_qr_ :: forall t . THTensorLapack' t => t -> t -> t -> IO ()
td_qr_ resQ resR a = withManaged3 GenRaw.c_qr resQ resR a

td_gesvd = undefined

td_gesvd2 = undefined
