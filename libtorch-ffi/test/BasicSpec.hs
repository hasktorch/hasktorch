{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE DataKinds #-}

-- This test does the same test as deps/pytorch/aten/src/ATen/test/basic.cpp

module BasicSpec (spec) where

import Test.Hspec
import Control.Exception.Safe
import Control.Monad (forM_,forM,join)
import Data.Int
import Foreign
import Torch.Internal.Const
import Torch.Internal.Type
import Torch.Internal.Class
import Torch.Internal.Managed.Type.TensorOptions
import Torch.Internal.Managed.Type.Tensor
import Torch.Internal.Managed.Type.TensorList
import Torch.Internal.Managed.Type.Extra
import Torch.Internal.Managed.Type.IntArray
import Torch.Internal.Managed.Type.Scalar
import Torch.Internal.Managed.Type.Tuple
import Torch.Internal.Managed.Type.Context
import Torch.Internal.Managed.Native

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

options :: DeviceType -> ScalarType -> IO (ForeignPtr TensorOptions)
options dtype stype = join $ tensorOptions_dtype_s <$> device_D dtype <*> pure stype

ap1 fn a0  = join $ fn <$> a0
ap2 fn a0 a1  = join $ fn <$> a0 <*> a1
ap3 fn a0 a1 a2  = join $ fn <$> a0 <*> a1 <*> a2
ap4 fn a0 a1 a2 a3 = join $ fn <$> a0 <*> a1 <*> a2 <*> a3

at1 tensor i0 = tensor__at__l tensor i0
at2 tensor i0 i1 = ap2 tensor__at__l (at1 tensor i0) (pure i1)
at3 tensor i0 i1 i2 = ap2 tensor__at__l (at2 tensor i0 i1) (pure i2)

new' fn dsize dtype = ap2 fn (intArray dsize) (options kCPU dtype)
add' a b = join $ add_tts <$> pure a <*> pure b <*> newScalar_d 1
addM' a b = join $ add_tts <$> a <*> b <*> newScalar_d 1
add_s' a b = join $ add_tss <$> pure a <*> pure b <*> newScalar_d 1
addM_s' a b = join $ add_tss <$> a <*> b <*> newScalar_d 1

spec :: Spec
spec = forM_ [
  (kFloat,"float"),
  (kDouble,"double")
  ] $ \(dtype,dtype_str) -> describe ("BasicSpec:" <> dtype_str) $ do
-- void TestResize(Type& type) {
--   auto a = at::empty({0}, type.options());
--   a.resize_({3, 4});
--   ASSERT_EQ_RESOLVED(a.numel(), 12);
--   a.resize_({5, 7});
--   ASSERT_EQ_RESOLVED(a.numel(), 35);
-- }
  it "TestReisze" $ do
    a <- new' empty_lo [1,1] dtype
    a1 <- join $ tensor_resize__l <$> pure a <*> intArray [3,4]
    tensor_numel a1 `shouldReturn` 12

    a2 <- join $ tensor_resize__l <$> pure a <*> intArray [5,7]
    tensor_numel a2 `shouldReturn` 35

-- void TestOnesAndDot(Type& type) {
--   Tensor b0 = ones({1, 1}, type);
--   ASSERT_EQ_RESOLVED((b0 + b0).sum().item<double>(), 2);

--   Tensor b1 = ones({1, 2}, type);
--   ASSERT_EQ_RESOLVED((b1 + b1).sum().item<double>(), 4);

--   Tensor b = ones({3, 4}, type);
--   ASSERT_EQ_RESOLVED((b + b).sum().item<double>(), 24);
--   ASSERT_EQ_RESOLVED(b.numel(), 12);
--   ASSERT_EQ_RESOLVED(b.view(-1).dot(b.view(-1)).item<double>(), 12);
-- }
  it "TestOnesAndDot" $ do
    b0 <- new' ones_lo [1,1] dtype
    b01 <- add' b0 b0
    b02 <- sum_t b01
    tensor_item_double b02 `shouldReturn` 2

    b0 <- new' ones_lo [1,2] dtype
    b01 <- add' b0 b0
    b02 <- sum_t b01
    tensor_item_double b02 `shouldReturn` 4

    b0 <- new' ones_lo [3,4] dtype
    b01 <- add' b0 b0
    b02 <- sum_t b01
    tensor_item_double b02 `shouldReturn` 24
    tensor_numel b0 `shouldReturn` 12
    b03 <- join $ tensor_view_l <$> pure b0 <*> intArray [-1]
    b04 <- tensor_dot_t b03 b03
    tensor_item_double b04 `shouldReturn` 12


-- void TestSort(Type& type) {
--   Tensor b = rand({3, 4}, type);

--   auto z = b.sort(1);
--   auto z_sorted = std::get<0>(z);

--   bool isLT = z_sorted[0][0].item<float>() < z_sorted[0][1].item<float>();
--   ASSERT_TRUE(isLT);
-- }
  it "TestSort" $ do
    b <- new' rand_lo [3,4] dtype
    z <- tensor_sort_lb b 1 0 :: IO (ForeignPtr (StdTuple '(Tensor,Tensor)))
    z_sorted <- get0 z
    z00 <- at2 z_sorted 0 0 >>= tensor_item_float
    z01 <- at2 z_sorted 0 1 >>= tensor_item_float
    z00 < z01 `shouldBe` True

-- void TestRandperm(Type& type) {
--   if (type.backend() != Backend::CUDA) {
--     Tensor b = randperm(15, type);
--     Tensor rv, ri;
--     std::tie(rv, ri) = sort(b, 0);
--     bool isLE = (rv[0].item<float>() <= rv[1].item<float>());
--     ASSERT_TRUE(isLE);
--   }
-- }

-- void SendContext() {
--   std::stringstream ss;
--   ss << "context: " << std::hex << (int64_t)&globalContext() << std::endl;
-- }

-- void TestAdd(Type& type) {
--   Tensor a = rand({3, 4}, type);
--   Tensor b = rand({3, 4}, type);
--   Tensor c = add(a, add(a, b));
--   // TODO:0-dim Tensor d(3.f);
--   Scalar d = 3.f;
--   ASSERT_TRUE(add(c, d).allclose(a + a + b + d));
-- }
  it "TestAdd" $ do
    a <- new' rand_lo [3,4] dtype
    b <- new' rand_lo [3,4] dtype
    c <- addM' (pure a) (add' a b)
    d <- newScalar_d 3
    e <- add_s' c d
    f <- addM_s' (addM' (add' a a) (pure b)) (pure d)
    allclose_ttddb e f (1e-05) (1e-08) 0 `shouldReturn` 1

  it "TestAdd2" $ do
    a <- new' ones_lo [3,4] dtype
    b <- new' ones_lo [3,4] dtype
    c <- add' a b
    (at2 c 0 0 >>= tensor_item_double) `shouldReturn` 2

-- void TestLoadsOfAdds(Type& type) {
--   auto begin = std::chrono::high_resolution_clock::now();
--   Tensor d = ones({3, 4}, type);
--   Tensor r = zeros({3, 4}, type);
--   for (auto i = 0; i < 100000; i++) {
--     add_out(r, r, d);
--   }
--   auto end = std::chrono::high_resolution_clock::now();
--   // TODO TEST PERF?
--   std::cout << std::dec << "   "
--             << std::chrono::duration_cast<std::chrono::milliseconds>(
--                    end - begin)
--                    .count()
--             << " ms" << std::endl;
--   ASSERT_EQ_RESOLVED(norm(100000 * d).item<double>(), norm(r).item<double>());
-- }
{-
  it "TestLoadsOfAdds" $ do
    d <- new' ones_lo [3,4] dtype
    r <- new' rand_lo [3,4] dtype
    one <- newScalar_d 1
    forM_ [0..99999] $ \_ -> do
      void $ add_out_ttts r r d one
    a <- join $ tensor_item_double <$> (join $ mul_tss <$> pure d <*> newScalar_i 100000)
-}

-- void TestLoadOfAddsWithCopy(Type& type) {
--   auto begin = std::chrono::high_resolution_clock::now();
--   Tensor d = ones({3, 4}, type);
--   Tensor r = zeros({3, 4}, type);
--   for (auto i = 0; i < 100000; i++) {
--     r = add(r, d);
--   }
--   auto end = std::chrono::high_resolution_clock::now();
--   // TODO TEST PERF?
--   std::cout << std::dec << "   "
--             << std::chrono::duration_cast<std::chrono::milliseconds>(
--                    end - begin)
--                    .count()
--             << " ms" << std::endl;
--   ASSERT_EQ_RESOLVED(norm(100000 * d).item<double>(), norm(r).item<double>());
-- }

-- void TestIsContiguous(Type& type) {
--   Tensor a = rand({3, 4}, type);
--   ASSERT_TRUE(a.is_contiguous());
--   a = a.transpose(0, 1);
--   ASSERT_FALSE(a.is_contiguous());
-- }
  it "TestIsContiguous" $ do
    a <- new' rand_lo [3,4] dtype
    tensor_is_contiguous a `shouldReturn` 1
    (join $ tensor_is_contiguous <$> tensor_transpose_ll a 0 1) `shouldReturn` 0

-- void TestPermute(Type& type) {
--   Tensor a = rand({3, 4, 5}, type);
--   Tensor b = a.permute({1, 2, 0});
--   ASSERT_TRUE(b.sizes().equals({4, 5, 3}));
--   ASSERT_TRUE(b.strides().equals({5, 1, 20}));
-- }

-- void TestMm(Type& type) {
--   Tensor a = rand({3, 4}, type);
--   Tensor b = rand({4}, type);
--   Tensor c = mv(a, b);
--   ASSERT_TRUE(c.equal(addmv(zeros({3}, type), a, b, 0, 1)));
-- }
  it "TTestMm" $ do
    a <- new' rand_lo [3,4] dtype
    b <- new' rand_lo [4] dtype
    c <- mv_tt a b
    z <- new' zeros_lo [3] dtype
    d <- join $ addmv_tttss <$> pure z <*> pure a <*> pure b <*> newScalar_d 0 <*> newScalar_d 1
    tensor_equal_t c d `shouldReturn` 1

-- void TestSqueeze(Type& type) {
--   Tensor a = rand({2, 1}, type);
--   Tensor b = squeeze(a);
--   ASSERT_EQ_RESOLVED(b.dim(), 1);
--   a = rand({1}, type);
--   b = squeeze(a);
--   // TODO 0-dim squeeze
--   ASSERT_TRUE(a[0].equal(b));
-- }
  it "TestSqueeze" $ do
    a <- new' rand_lo [2,1] dtype
    b <- squeeze_t a
    tensor_dim b `shouldReturn` 1
    a <- new' rand_lo [1] dtype
    b <- squeeze_t a
    (join $ tensor_equal_t <$> (tensor__at__l a 0) <*> pure b) `shouldReturn` 1

-- void TestCopy(Type& type) {
--   Tensor a = zeros({4, 3}, type);
--   Tensor e = rand({4, 3}, type);
--   a.copy_(e);
--   ASSERT_TRUE(a.equal(e));
-- }
  it "TTestCopy" $ do
    a <- new' zeros_lo [4,3] dtype
    e <- new' rand_lo [4,3] dtype
    _ <- tensor_copy__tb a e 0
    tensor_equal_t a e `shouldReturn` 1

-- void TestCopyBroadcasting(Type& type) {
--   Tensor a = zeros({4, 3}, type);
--   Tensor e = rand({3}, type);
--   a.copy_(e);
--   for (int i = 0; i < 4; ++i) {
--     ASSERT_TRUE(a[i].equal(e));
--   }
-- }
  it "TestCopyBroadcasting" $ do
    a <- new' zeros_lo [4,3] dtype
    e <- new' rand_lo [3] dtype
    _ <- tensor_copy__tb a e 0
    forM_ [0..3] $ \i -> do
      (join $ tensor_equal_t <$>  tensor__at__l a i <*> pure e) `shouldReturn` 1

-- void TestAbsValue(Type& type) {
--   Tensor r = at::abs(at::scalar_tensor(-3, type.options()));
--   ASSERT_EQ_RESOLVED(r.item<int32_t>(), 3);
-- }
  it "TestAbsValue" $ do
    r <- join $ abs_t <$> (join $ scalar_tensor_so <$> newScalar_i (-3) <*> options kCPU dtype)
    tensor_item_float r `shouldReturn` 3

-- void TestAddingAValueWithScalar(Type& type) {
--   Tensor a = rand({4, 3}, type);
--   ASSERT_TRUE((ones({4, 3}, type) + a).equal(add(a, 1)));
-- }
  it "TestAddingAValueWithScalar" $ do
    a <- new' rand_lo [4, 3] dtype
    b <- new' ones_lo [4, 3] dtype
    one <- newScalar_d 1
    c <- add' b a
    d <- add_s' a one
    tensor_equal_t c d  `shouldReturn` 1

-- void TestSelect(Type& type) {
--   Tensor a = rand({3, 7}, type);
--   auto a_13 = select(a, 1, 3);
--   auto a_13_02 = select(select(a, 1, 3), 0, 2);
--   ASSERT_TRUE(a[0][3].equal(a_13[0]));
--   ASSERT_TRUE(a[2][3].equal(a_13_02));
-- }
  it "TestSelect" $ do
    a <- new' rand_lo [3, 7] dtype
    a13 <- select_tll a 1 3
    a13_02 <- ap3 select_tll (select_tll a 1 3) (pure 0) (pure 2)
    ap2 tensor_equal_t (at2 a 0 3) (at1 a13 0) `shouldReturn` 1
    ap2 tensor_equal_t (at2 a 2 3) (pure a13_02) `shouldReturn` 1


-- void TestZeroDim(Type& type) {
--   Tensor a = at::scalar_tensor(4, type.options()); // rand(type, {1});

--   Tensor b = rand({3, 4}, type);
--   ASSERT_EQ_RESOLVED((a + a).dim(), 0);
--   ASSERT_EQ_RESOLVED((1 + a).dim(), 0);
--   ASSERT_EQ_RESOLVED((b + a).dim(), 2);
--   ASSERT_EQ_RESOLVED((a + b).dim(), 2);
--   auto c = rand({3, 4}, type);
--   ASSERT_EQ_RESOLVED(c[1][2].dim(), 0);

--   auto f = rand({3, 4}, type);
--   f[2] = zeros({4}, type);
--   f[1][0] = -1;
--   ASSERT_EQ_RESOLVED(f[2][0].item<double>(), 0);
-- }
  it "TestZeroDim" $ do
    a <- ap2 scalar_tensor_so (newScalar_i 4) (options kCPU dtype)
    b <- new' rand_lo [3,4] dtype
    one <- newScalar_d 1
    (add' a a     >>= tensor_dim) `shouldReturn` 0
    (add_s' a one >>= tensor_dim) `shouldReturn` 0
    (add' b a     >>= tensor_dim) `shouldReturn` 2
    (add' a b     >>= tensor_dim) `shouldReturn` 2
    c <- new' rand_lo [3,4] dtype
    (at2 c 1 2    >>= tensor_dim) `shouldReturn` 0
    f <- new' rand_lo [3,4] dtype
    ap3 tensor_assign1_t (pure f) (pure 2) (new' zeros_lo [4] dtype)
    tensor_assign2_l f 1 0 (-1)
    (at2 f 2 0 >>= tensor_item_double) `shouldReturn` 0


-- void TestTensorFromTH() {
--   int a = 4;
--   THFloatTensor* t = THFloatTensor_newWithSize2d(a, a);
--   THFloatTensor_fill(t, a);
--   ASSERT_NO_THROW(CPU(kFloat).unsafeTensorFromTH(t, false));
-- }

-- void TestToCFloat() {
--   Tensor a = zeros({3, 4});
--   Tensor b = ones({3, 7});
--   Tensor c = cat({a, b}, 1);
--   ASSERT_EQ_RESOLVED(c.size(1), 11);
--   Tensor e = rand({});
--   ASSERT_EQ_RESOLVED(*e.data<float>(), e.sum().item<float>());
-- }
  -- it "TestToCFloat" $ do
  --   a <- new' zeros_lo [3,4] dtype
  --   b <- new' ones_lo [3,7] dtype
  --   c <- ap2 cat_ll (tensorList [a,b]) (pure 1)
  --   tensor_size_l c 1 `shouldReturn` 11

-- void TestToString() {
--   Tensor b = ones({3, 7}) * .0000001f;
--   std::stringstream s;
--   s << b << "\n";
--   std::string expect = "1e-07 *";
--   ASSERT_EQ_RESOLVED(s.str().substr(0, expect.size()), expect);
-- }

-- void TestIndexingByScalar() {
--   Tensor tensor = arange(0, 10, kInt);
--   Tensor one = ones({}, kInt);
--   for (int64_t i = 0; i < tensor.numel(); ++i) {
--     ASSERT_TRUE(tensor[i].equal(one * i));
--   }
--   for (size_t i = 0; i < static_cast<uint64_t>(tensor.numel()); ++i) {
--     ASSERT_TRUE(tensor[i].equal(one * static_cast<int64_t>(i)));
--   }
--   for (int i = 0; i < tensor.numel(); ++i) {
--     ASSERT_TRUE(tensor[i].equal(one * i));
--   }
--   for (int16_t i = 0; i < tensor.numel(); ++i) {
--     ASSERT_TRUE(tensor[i].equal(one * i));
--   }
--   for (int8_t i = 0; i < tensor.numel(); ++i) {
--     ASSERT_TRUE(tensor[i].equal(one * i));
--   }
--   // Throw StartsWith("Can only index tensors with integral scalars")
--   ASSERT_ANY_THROW(tensor[Scalar(3.14)].equal(one));
-- }
  it "TestIndexingByScalar" $ do
    tensor <- ap3 arange_sso (newScalar_i 0) (newScalar_i 10) (options kCPU kInt)
    one <- new' ones_lo [] kInt
    num <- tensor_numel tensor
    forM_ [0..(num-1)] $ \i -> do
      ap2 tensor_equal_t (at1 tensor i) (ap2 mul_ts (pure one) (newScalar_i (fromIntegral i))) `shouldReturn` 1
    forM_ [0..(num-1)] $ \i -> do
      ap2 tensor_equal_t (at1 tensor i) (ap2 mul_ts (pure one) (newScalar_i (fromIntegral i))) `shouldReturn` 1
    at1 tensor 314 `shouldThrow` anyException


-- void TestIndexingByZerodimTensor() {
--   Tensor tensor = arange(0, 10, kInt);
--   Tensor one = ones({}, kInt);
--   for (int i = 0; i < tensor.numel(); ++i) {
--     ASSERT_TRUE(tensor[one * i].equal(one * i));
--   }
--   // Throw StartsWith(
--   //            "Can only index tensors with integral scalars")
--   ASSERT_ANY_THROW(tensor[ones({}) * 3.14].equal(one));
--   // Throw StartsWith("Can only index with tensors that are defined")
--   ASSERT_ANY_THROW(tensor[Tensor()].equal(one));
--   // Throw StartsWith("Can only index with tensors that are scalars (zero-dim)")
--   ASSERT_ANY_THROW(tensor[ones({2, 3, 4}, kInt)].equal(one));
-- }
-- void TestIndexingMixedDevice(Type& type) {
--   Tensor tensor = randn({20, 20}, type);
--   Tensor index = arange(10, kLong).cpu();
--   Tensor result = tensor.index({index});
--   ASSERT_TRUE(result[0].equal(tensor[0]));
-- }
-- void TestDispatch() {
--   Tensor tensor = randn({20, 20});
--   Tensor other = randn({20, 20});
--   auto result = tensor.m(relu).m(mse_loss, other, Reduction::Mean);
--   ASSERT_TRUE(result.allclose(mse_loss(relu(tensor), other)));
-- }

-- void TestNegativeDim(Type& type) {
--   ASSERT_ANY_THROW(empty({5, -5, 5}, type.options()));
--   ASSERT_ANY_THROW(empty({5, -5, -5}, type.options()));
--   Tensor tensor = empty({5, 5}, type.options());
--   ASSERT_ANY_THROW(tensor.reshape({-5, -5}));
-- }

-- TEST(BasicTest, BasicTestCPU) {
--   manual_seed(123);

--   test(CPU(kFloat));
-- }

-- TEST(BasicTest, BasicTestCUDA) {
--   manual_seed(123);

--   if (at::hasCUDA()) {
--     test(CUDA(kFloat));
--   }
-- }

