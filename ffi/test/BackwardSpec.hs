{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE DataKinds #-}

-- This test does the same test as deps/pytorch/aten/src/ATen/test/basic.cpp

module BackwardSpec (spec) where

import Test.Hspec
import Control.Exception.Safe
import qualified Language.C.Inline.Cpp.Exceptions as C
import Control.Monad (forM_,forM,join)
import Data.Int
import Foreign
import Aten.Const
import Aten.Type
import Aten.Class
import Aten.Managed.Type.TensorOptions
import Aten.Managed.Type.Tensor
import Aten.Managed.Type.TensorList
import Aten.Managed.Type.Extra
import Aten.Managed.Type.IntArray
import Aten.Managed.Type.Scalar
import Aten.Managed.Type.Tuple
import qualified Aten.Managed.Native as A
import Torch.Managed.Native

intArray :: [Int64] -> IO (ForeignPtr IntArray)
intArray dims = do
  ary <- newIntArray
  forM_ dims $ intArray_push_back_l ary
  return ary

tensorList :: [ForeignPtr Tensor] -> IO (ForeignPtr TensorList)
tensorList dims = do
  ary <- newTensorList
  forM_ dims $ tensorList_push_back_t ary
  return ary

ap1 fn a0  = join $ fn <$> a0
ap2 fn a0 a1  = join $ fn <$> a0 <*> a1
ap3 fn a0 a1 a2  = join $ fn <$> a0 <*> a1 <*> a2
ap4 fn a0 a1 a2 a3 = join $ fn <$> a0 <*> a1 <*> a2 <*> a3

at1 tensor i0 = tensor__at__l tensor i0
at2 tensor i0 i1 = ap2 tensor__at__l (at1 tensor i0) (pure i1)
at3 tensor i0 i1 i2 = ap2 tensor__at__l (at2 tensor i0 i1) (pure i2)

new' fn dsize dtype = ap2 fn (intArray dsize) (options kCPU dtype)
add' a b = join $ A.add_tts <$> pure a <*> pure b <*> newScalar_d 1
addM' a b = join $ A.add_tts <$> a <*> b <*> newScalar_d 1
add_s' a b = join $ A.add_tss <$> pure a <*> pure b <*> newScalar_d 1
addM_s' a b = join $ A.add_tss <$> a <*> b <*> newScalar_d 1


options :: DeviceType -> ScalarType -> IO (ForeignPtr TensorOptions)
options dtype stype = ap2 tensorOptions_requires_grad_b (ap2 tensorOptions_dtype_s (device_D dtype) (pure stype)) (pure 1)

spec :: Spec
spec = forM_ [
  (kFloat,"float"),
  (kDouble,"double")
  ] $ \(dtype,dtype_str) -> describe ("BasicSpec:" <> dtype_str) $ do
--        torch::Tensor a = torch::ones({2, 2}, torch::requires_grad());
--        torch::Tensor b = torch::randn({2, 2});
--        auto c = a + b;
--        c.backward();
--        std::cout << a << std::endl << b << std::endl << c << std::endl;
  it "Backward" $ do
    a <- new' ones_lo [2,2] dtype
    print "--a--"
    forM_ [0..1] $ \i ->
      forM_ [0..1] $ \j ->
        at2 a i j >>= tensor_item_double >>= print
    b <- new' randn_lo [2,2] dtype
    print "--b--"
    forM_ [0..1] $ \i ->
      forM_ [0..1] $ \j ->
        at2 b i j >>= tensor_item_double >>= print
    print "--c--"
    c <- add' a b
    forM_ [0..1] $ \i ->
      forM_ [0..1] $ \j ->
        at2 c i j >>= tensor_item_double >>= print
    tensor_print c
    tensor_backward c
    a' <- tensor_grad a
    print "--a'--"
    forM_ [0..1] $ \i ->
      forM_ [0..1] $ \j ->
        (at2 a' i j >>= tensor_item_double) `shouldReturn` 1


