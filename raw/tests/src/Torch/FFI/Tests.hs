{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ConstraintKinds #-}
module Torch.FFI.Tests where

import Foreign
import Foreign.C.Types

import Test.Hspec


data TestFunctions state tensor real accreal = TestFunctions
  { _new :: state -> IO tensor
  , _newWithSize1d :: state -> CLLong -> IO tensor
  , _newWithSize2d :: state -> CLLong -> CLLong -> IO tensor
  , _newWithSize3d :: state -> CLLong -> CLLong -> CLLong -> IO tensor
  , _newWithSize4d :: state -> CLLong -> CLLong -> CLLong -> CLLong -> IO tensor
  , _nDimension :: state -> tensor -> IO CInt
  , _set1d :: state -> tensor -> CLLong -> real -> IO ()
  , _get1d :: state -> tensor -> CLLong -> IO real
  , _set2d :: state -> tensor -> CLLong -> CLLong -> real -> IO ()
  , _get2d :: state -> tensor -> CLLong -> CLLong -> IO real
  , _set3d :: state -> tensor -> CLLong -> CLLong -> CLLong -> real -> IO ()
  , _get3d :: state -> tensor -> CLLong -> CLLong -> CLLong -> IO real
  , _set4d :: state -> tensor -> CLLong -> CLLong -> CLLong -> CLLong -> real -> IO ()
  , _get4d :: state -> tensor -> CLLong -> CLLong -> CLLong -> CLLong -> IO real
  , _size :: state -> tensor -> CInt -> IO CLLong
  , _fill :: state -> tensor -> real -> IO ()
  , _free :: state -> tensor -> IO ()
  , _sumall :: state -> tensor -> IO accreal
  , _prodall :: state -> tensor -> IO accreal
  , _zero :: state -> tensor -> IO ()
  , _dot :: state -> tensor -> tensor -> IO accreal
  , _abs :: Maybe (state -> tensor -> tensor -> IO ())
  }

type RealConstr n = (Num n, Show n, Eq n)

signedSuite :: (RealConstr real, RealConstr accreal) => state -> TestFunctions state tensor real accreal -> Spec
signedSuite s fs = do
  it "initializes empty tensor with 0 dimension" $ do
    t <- new s
    nDimension s t >>= (`shouldBe` 0)
    free s t
  it "1D tensor has correct dimensions and sizes" $ do
    t <- newWithSize1d s 10
    nDimension s t >>= (`shouldBe` 1)
    size s t 0 >>= (`shouldBe` 10)
    free s t
  it "2D tensor has correct dimensions and sizes" $ do
    t <- newWithSize2d s 10 25
    nDimension s t >>= (`shouldBe` 2)
    size s t 0 >>= (`shouldBe` 10)
    size s t 1 >>= (`shouldBe` 25)
    free s t
  it "3D tensor has correct dimensions and sizes" $ do
    t <- newWithSize3d s 10 25 5
    nDimension s t >>= (`shouldBe` 3)
    size s t 0 >>= (`shouldBe` 10)
    size s t 1 >>= (`shouldBe` 25)
    size s t 2 >>= (`shouldBe` 5)
    free s t
  it "4D tensor has correct dimensions and sizes" $ do
    t <- newWithSize4d s 10 25 5 62
    nDimension s t >>= (`shouldBe` 4)
    size s t 0 >>= (`shouldBe` 10)
    size s t 1 >>= (`shouldBe` 25)
    size s t 2 >>= (`shouldBe` 5)
    size s t 3 >>= (`shouldBe` 62)
    free s t

  it "Can assign and retrieve correct 1D vector values" $ do
    t <- newWithSize1d s 10
    set1d s t 0 (20)
    set1d s t 1 (1)
    set1d s t 9 (3)
    get1d s t 0 >>= (`shouldBe` (20))
    get1d s t 1 >>= (`shouldBe` (1))
    get1d s t 9 >>= (`shouldBe` (3))
    free s t
  it "Can assign and retrieve correct 2D vector values" $ do
    t <- newWithSize2d s 10 15
    set2d s t 0 0 (20)
    set2d s t 1 5 (1)
    set2d s t 9 9 (3)
    get2d s t 0 0 >>= (`shouldBe` (20))
    get2d s t 1 5 >>= (`shouldBe` (1))
    get2d s t 9 9 >>= (`shouldBe` (3))
    free s t
  it "Can assign and retrieve correct 3D vector values" $ do
    t <- newWithSize3d s 10 15 10
    set3d s t 0 0 0 (20)
    set3d s t 1 5 3 (1)
    set3d s t 9 9 9 (3)
    get3d s t 0 0 0 >>= (`shouldBe` (20))
    get3d s t 1 5 3 >>= (`shouldBe` (1))
    get3d s t 9 9 9 >>= (`shouldBe` (3))
    free s t
  it "Can assign and retrieve correct 4D vector values" $ do
    t <- newWithSize4d s 10 15 10 20
    set4d s t 0 0 0 0 (20)
    set4d s t 1 5 3 2 (1)
    set4d s t 9 9 9 9 (3)
    get4d s t 0 0 0 0 >>= (`shouldBe` (20))
    get4d s t 1 5 3 2 >>= (`shouldBe` (1))
    get4d s t 9 9 9 9 >>= (`shouldBe` (3))
    free s t
  it "Can can initialize values with the fill method" $ do
    t1 <- newWithSize2d s 2 2
    fill s t1 3
    get2d s t1 0 0 >>= (`shouldBe` (3))
    free s t1
  it "Can compute correct dot product between 1D vectors" $ do
    t1 <- newWithSize1d s 3
    t2 <- newWithSize1d s 3
    fill s t1 3
    fill s t2 4
    let value = dot s t1 t2
    value >>= (`shouldBe` 36)
    free s t1
    free s t2
  it "Can compute correct dot product between 2D tensors" $ do
    t1 <- newWithSize2d s 2 2
    t2 <- newWithSize2d s 2 2
    fill s t1 3
    fill s t2 4
    let value = dot s t1 t2
    value >>= (`shouldBe` 48)
    free s t1
    free s t2
  it "Can compute correct dot product between 3D tensors" $ do
    t1 <- newWithSize3d s 2 2 4
    t2 <- newWithSize3d s 2 2 4
    fill s t1 3
    fill s t2 4
    let value = dot s t1 t2
    value >>= (`shouldBe` 192)
    free s t1
    free s t2
  it "Can compute correct dot product between 4D tensors" $ do
    t1 <- newWithSize4d s 2 2 2 1
    t2 <- newWithSize4d s 2 2 2 1
    fill s t1 3
    fill s t2 4
    let value = dot s t1 t2
    value >>= (`shouldBe` 96)
    free s t1
    free s t2
  it "Can zero out values" $ do
    t1 <- newWithSize4d s 2 2 4 3
    fill s t1 3
    -- let value = dot s t1 t1
    -- sequencing does not work if there is more than one shouldBe test in
    -- an "it" monad
    -- value >>= (`shouldBe` (432.0))
    zero s t1
    dot s t1 t1 >>= (`shouldBe` 0)
    free s t1
  it "Can compute sum of all values" $ do
    t1 <- newWithSize3d s 2 2 4
    fill s t1 2
    sumall s t1 >>= (`shouldBe` 32)
    free s t1
  it "Can compute product of all values" $ do
    t1 <- newWithSize2d s 2 2
    fill s t1 2
    prodall s t1 >>= (`shouldBe` 16)
    free s t1
  case mabs of
    Nothing  -> pure ()
    Just abs ->
      it "Can take abs of tensor values" $ do
        t1 <- newWithSize2d s 2 2
        fill s t1 (-2)
        -- sequencing does not work if there is more than one shouldBe test in
        -- an "it" monad
        -- sumall s t1 >>= (`shouldBe` (-6.0))
        abs s t1 t1
        sumall s t1 >>= (`shouldBe` 8)
        free s t1
 where
  new = _new fs
  newWithSize1d = _newWithSize1d fs
  newWithSize2d = _newWithSize2d fs
  newWithSize3d = _newWithSize3d fs
  newWithSize4d = _newWithSize4d fs
  nDimension = _nDimension fs
  set1d = _set1d fs
  get1d = _get1d fs
  set2d = _set2d fs
  get2d = _get2d fs
  set3d = _set3d fs
  get3d = _get3d fs
  set4d = _set4d fs
  get4d = _get4d fs
  size = _size fs
  fill = _fill fs
  free = _free fs
  sumall = _sumall fs
  mabs = _abs fs
  prodall = _prodall fs
  dot = _dot fs
  zero = _zero fs


