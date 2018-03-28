{-# LANGUAGE InstanceSigs #-}
module Torch.Indef.Tensor.Dynamic.Random.Floating where

import Foreign
import Foreign.C.Types
import GHC.Int
import qualified Torch.Sig.Tensor.RandomFloating as Sig
import qualified Torch.Class.Tensor.Random as Class
import Torch.Types.TH
import Torch.Types.TH.Random
import Torch.Indef.Types
import Torch.Indef.Tensor.Dynamic.Random (withTensorAndRNG)

import qualified Torch.FFI.TH.Float.Tensor  as F
import qualified Torch.Types.TH.Float   as F
import qualified Torch.FFI.TH.Double.Tensor as D
import qualified Torch.Types.TH.Double  as D

-- withTensorAndRNG :: Tensor -> Generator -> (Ptr CTensor -> Ptr CTHGenerator -> IO ()) -> IO ()

instance Class.TensorRandomFloating Dynamic where
  uniform_ :: Dynamic -> Generator -> HsAccReal -> HsAccReal -> IO ()
  uniform_ t g a b = withTensorAndRNG t g $ \t' g' -> Sig.c_uniform t' g' (hs2cAccReal a) (hs2cAccReal b)

  normal_ :: Dynamic -> Generator -> HsAccReal -> HsAccReal -> IO ()
  normal_ t g a b = withTensorAndRNG t g $ \t' g' -> Sig.c_normal t' g' (hs2cAccReal a) (hs2cAccReal b)

  normal_means_ :: Dynamic -> Generator -> Dynamic -> HsAccReal -> IO ()
  normal_means_ t g m s = withTensorAndRNG t g $ \t' g' -> withForeignPtr (tensor m) $ \m' -> Sig.c_normal_means t' g' m' (hs2cAccReal s)

  normal_stddevs_ :: Dynamic -> Generator -> HsAccReal -> Dynamic -> IO ()
  normal_stddevs_ t g m s = withTensorAndRNG t g $ \t' g' -> withForeignPtr (tensor s) $ \s' -> Sig.c_normal_stddevs t' g' (hs2cAccReal m) s'

  normal_means_stddevs_  :: Dynamic -> Generator -> Dynamic -> Dynamic -> IO ()
  normal_means_stddevs_ t g m s =
    withTensorAndRNG t g $ \t' g' ->
      withForeignPtr (tensor m) $ \m' ->
        withForeignPtr (tensor s) $ \s' ->
          Sig.c_normal_means_stddevs t' g' m' s'

  exponential_ :: Dynamic -> Generator -> HsAccReal -> IO ()
  exponential_ t g a = withTensorAndRNG t g $ \t' g' -> Sig.c_exponential t' g' (hs2cAccReal a)

  standard_gamma_ :: Dynamic -> Generator -> Dynamic -> IO ()
  standard_gamma_ t g m = withTensorAndRNG t g $ \t' g' -> withForeignPtr (tensor m) $ \m' -> Sig.c_standard_gamma t' g' m'

  cauchy_ :: Dynamic -> Generator -> HsAccReal -> HsAccReal -> IO ()
  cauchy_ t g a b = withTensorAndRNG t g $ \t' g' -> Sig.c_cauchy t' g' (hs2cAccReal a) (hs2cAccReal b)

  logNormal_ :: Dynamic -> Generator -> HsAccReal -> HsAccReal -> IO ()
  logNormal_ t g a b = withTensorAndRNG t g $ \t' g' -> Sig.c_logNormal t' g' (hs2cAccReal a) (hs2cAccReal b)

--   multinomial :: Ptr CTorch.FFI.TH.Long.Tensor -> Ptr CTHGenerator -> Tensor -> Int32 -> Int32 -> IO ()
--   multinomial lt g t i0 i1 = _withTensor t $ \ t' -> Sig.c_multinomial lt g t' (CInt i0) (CInt i1)
--
--   multinomialAliasSetup  :: Tensor -> Ptr CTorch.FFI.TH.Long.Tensor -> Tensor -> IO ()
--   multinomialAliasSetup t0 g t1 = _with2Tensors t0 t1 $ \ t0' t1' -> Sig.c_multinomialAliasSetup t0' g t1'
--
--   multinomialAliasDraw   :: Ptr CTorch.FFI.TH.Long.Tensor -> Ptr CTHGenerator -> Ptr CTHLongTensor -> Tensor -> IO ()
--   multinomialAliasDraw lt0 g lt1 t = _withTensor t $ \ t' -> Sig.c_multinomialAliasDraw lt0 g lt1 t'

