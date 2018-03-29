{-# LANGUAGE InstanceSigs #-}
module Torch.Core.Tensor.Dynamic.Random.Floating where

import Foreign
import Foreign.C.Types
import GHC.Int
import qualified TensorRandomFloating as Sig
import qualified Torch.Class.C.Tensor.Random as Class
import THTypes
import THRandomTypes
import Torch.Core.Types
import Torch.Core.Tensor.Dynamic.Random (withTensorAndRNG)

import qualified THFloatTensor  as F
import qualified THFloatTypes   as F
import qualified THDoubleTensor as D
import qualified THDoubleTypes  as D

-- withTensorAndRNG :: Tensor -> Generator -> (Ptr CTensor -> Ptr CTHGenerator -> IO ()) -> IO ()

instance Class.TensorRandomFloating DynTensor where
  uniform_ :: DynTensor -> Generator -> HsAccReal -> HsAccReal -> IO ()
  uniform_ t g a b = withTensorAndRNG t g $ \t' g' -> Sig.c_uniform t' g' (hs2cAccReal a) (hs2cAccReal b)

  normal_ :: DynTensor -> Generator -> HsAccReal -> HsAccReal -> IO ()
  normal_ t g a b = withTensorAndRNG t g $ \t' g' -> Sig.c_normal t' g' (hs2cAccReal a) (hs2cAccReal b)

  normal_means_ :: DynTensor -> Generator -> DynTensor -> HsAccReal -> IO ()
  normal_means_ t g m s = withTensorAndRNG t g $ \t' g' -> withForeignPtr (tensor m) $ \m' -> Sig.c_normal_means t' g' m' (hs2cAccReal s)

  normal_stddevs_ :: DynTensor -> Generator -> HsAccReal -> DynTensor -> IO ()
  normal_stddevs_ t g m s = withTensorAndRNG t g $ \t' g' -> withForeignPtr (tensor s) $ \s' -> Sig.c_normal_stddevs t' g' (hs2cAccReal m) s'

  normal_means_stddevs_  :: DynTensor -> Generator -> DynTensor -> DynTensor -> IO ()
  normal_means_stddevs_ t g m s =
    withTensorAndRNG t g $ \t' g' ->
      withForeignPtr (tensor m) $ \m' ->
        withForeignPtr (tensor s) $ \s' ->
          Sig.c_normal_means_stddevs t' g' m' s'

  exponential_ :: DynTensor -> Generator -> HsAccReal -> IO ()
  exponential_ t g a = withTensorAndRNG t g $ \t' g' -> Sig.c_exponential t' g' (hs2cAccReal a)

  standard_gamma_ :: DynTensor -> Generator -> DynTensor -> IO ()
  standard_gamma_ t g m = withTensorAndRNG t g $ \t' g' -> withForeignPtr (tensor m) $ \m' -> Sig.c_standard_gamma t' g' m'

  cauchy_ :: DynTensor -> Generator -> HsAccReal -> HsAccReal -> IO ()
  cauchy_ t g a b = withTensorAndRNG t g $ \t' g' -> Sig.c_cauchy t' g' (hs2cAccReal a) (hs2cAccReal b)

  logNormal_ :: DynTensor -> Generator -> HsAccReal -> HsAccReal -> IO ()
  logNormal_ t g a b = withTensorAndRNG t g $ \t' g' -> Sig.c_logNormal t' g' (hs2cAccReal a) (hs2cAccReal b)

--   multinomial :: Ptr CTHLongTensor -> Ptr CTHGenerator -> Tensor -> Int32 -> Int32 -> IO ()
--   multinomial lt g t i0 i1 = _withTensor t $ \ t' -> Sig.c_multinomial lt g t' (CInt i0) (CInt i1)
--
--   multinomialAliasSetup  :: Tensor -> Ptr CTHLongTensor -> Tensor -> IO ()
--   multinomialAliasSetup t0 g t1 = _with2Tensors t0 t1 $ \ t0' t1' -> Sig.c_multinomialAliasSetup t0' g t1'
--
--   multinomialAliasDraw   :: Ptr CTHLongTensor -> Ptr CTHGenerator -> Ptr CTHLongTensor -> Tensor -> IO ()
--   multinomialAliasDraw lt0 g lt1 t = _withTensor t $ \ t' -> Sig.c_multinomialAliasDraw lt0 g lt1 t'

