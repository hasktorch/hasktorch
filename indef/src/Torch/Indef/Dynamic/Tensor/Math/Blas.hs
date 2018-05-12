{-# OPTIONS_GHC -fno-cse #-}
module Torch.Indef.Dynamic.Tensor.Math.Blas
  ( _addmv
  , _addmm
  , _addr
  , _addbmm
  , _baddbmm

  , addmv
  , addmm
  , addr
  , addbmm
  , baddbmm

  , addmv_
  , addmm_
  , addr_
  , addbmm_
  , baddbmm_

  , dot
  , (<.>)
  ) where

import Foreign
import GHC.Int

import Data.Void
import System.IO.Unsafe

import Torch.Indef.Types
import Torch.Indef.Dynamic.Tensor
import qualified Torch.Sig.Tensor.Math.Blas as Sig

blasOp
  :: (Ptr CState -> Ptr CTensor -> CReal -> Ptr CTensor -> CReal -> Ptr CTensor -> Ptr CTensor -> IO ())
  -> Dynamic -> HsReal -> Dynamic -> HsReal -> Dynamic -> Dynamic -> IO ()
blasOp fn r a x b y z =
  with2DynamicState r x $ \s' r' x' ->
    with2DynamicState y z $ \_ y' z' ->
      fn s' r' (hs2cReal a) x' (hs2cReal b) y' z'


_addmv   = blasOp Sig.c_addmv
_addmm   = blasOp Sig.c_addmm
_addr    = blasOp Sig.c_addr
_addbmm  = blasOp Sig.c_addbmm
_baddbmm = blasOp Sig.c_baddbmm

dot :: Dynamic -> Dynamic -> IO HsAccReal
dot a b = with2DynamicState a b $ fmap c2hsAccReal ..: Sig.c_dot

-- class GPUTensorMathBlas t where
--   btrifact :: t -> IntTensor -> IntTensor -> Int -> t -> io ()
--   btrisolve :: t -> t -> t -> IntTensor -> io ()


(<.>) :: Dynamic -> Dynamic -> HsAccReal
(<.>) a b = unsafePerformIO $ dot a b
{-# NOINLINE (<.>) #-}

mkInplaceFunction, mkNewFunction
  :: (Dynamic -> HsReal -> Dynamic -> HsReal -> Dynamic -> Dynamic -> IO ())
  -> HsReal -> Dynamic -> HsReal -> Dynamic -> Dynamic -> IO Dynamic
mkInplaceFunction op a m b x y = op x a m b x y >> pure x
mkNewFunction     op a m b x y = withEmpty x $ \r -> op r a m b x y

addmv, addmv_ :: HsReal -> Dynamic -> HsReal -> Dynamic -> Dynamic -> IO Dynamic
addmv  = mkNewFunction     _addmv
addmv_ = mkInplaceFunction _addmv

addmm, addmm_ :: HsReal -> Dynamic -> HsReal -> Dynamic -> Dynamic -> IO Dynamic
addmm  = mkNewFunction     _addmm
addmm_ = mkInplaceFunction _addmm

addr, addr_ :: HsReal -> Dynamic -> HsReal -> Dynamic -> Dynamic -> IO Dynamic
addr  = mkNewFunction     _addr
addr_ = mkInplaceFunction _addr

addbmm, addbmm_ :: HsReal -> Dynamic -> HsReal -> Dynamic -> Dynamic -> IO Dynamic
addbmm  = mkNewFunction     _addbmm
addbmm_ = mkInplaceFunction _addbmm

baddbmm, baddbmm_ :: HsReal -> Dynamic -> HsReal -> Dynamic -> Dynamic -> IO Dynamic
baddbmm  = mkNewFunction     _baddbmm
baddbmm_ = mkInplaceFunction _baddbmm

