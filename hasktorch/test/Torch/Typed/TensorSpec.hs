{-# LANGUAGE DataKinds #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE GADTs #-}
{-# OPTIONS_GHC -freduction-depth=0 #-}

module Torch.Typed.TensorSpec
  ( Torch.Typed.TensorSpec.spec
  )
where

import           Data.Kind
import           Data.Proxy
import           GHC.TypeLits

import           Test.Hspec (Spec, shouldBe, describe, it)
import           Test.QuickCheck ()

import Torch.Typed
import Torch.Typed.AuxSpec
import qualified Torch as Torch
import Torch.Internal.Class (uncast, Castable(cast))

data BinarySpec = AddSpec | SubSpec | MulSpec

instance
  ( TensorOptions shape   dtype   device
  , TensorOptions shape'  dtype'  device
  , TensorOptions shape'' dtype'' device
  , shape'' ~ Broadcast shape shape'
  , dtype'' ~ DTypePromotion dtype dtype'
  , BasicArithmeticDTypeIsValid device dtype
  , BasicArithmeticDTypeIsValid device dtype'
  , BasicArithmeticDTypeIsValid device dtype''
  ) => Apply' BinarySpec ((Proxy device, ((Proxy dtype, Proxy dtype'), (Proxy shape, Proxy shape'))), IO ()) (IO ())
 where
  apply' AddSpec (_, agg) = agg >> do
    let a = ones @shape  @dtype  @device
    let b = ones @shape' @dtype' @device
    let c = add a b
    checkDynamicTensorAttributes c
  apply' SubSpec (_, agg) = agg >> do
    let a = ones @shape  @dtype  @device
    let b = ones @shape' @dtype' @device
    let c = sub a b
    checkDynamicTensorAttributes c
  apply' MulSpec (_, agg) = agg >> do
    let a = ones @shape  @dtype  @device
    let b = ones @shape' @dtype' @device
    let c = mul a b
    checkDynamicTensorAttributes c

data MatMulSpec = MatMulSpec

instance
  ( TensorOptions shape   dtype device
  , TensorOptions shape'  dtype device
  , TensorOptions shape'' dtype device
  , shape'' ~ MatMul shape shape'
  , MatMulDTypeIsValid device dtype
  ) => Apply' MatMulSpec ((Proxy device, (Proxy dtype, (Proxy shape, Proxy shape'))), IO ()) (IO ()) where
  apply' MatMulSpec (_, agg) = agg >> do
    let a = ones @shape  @dtype @device
    let b = ones @shape' @dtype @device
    let c = matmul a b
    checkDynamicTensorAttributes c

data BinaryCmpSpec = GTSpec | LTSpec | GESpec | LESpec | EQSpec | NESpec

instance
  ( TensorOptions shape   dtype  device
  , TensorOptions shape'  dtype' device
  , TensorOptions shape'' 'Bool device
  , shape'' ~ Broadcast shape shape'
  , ComparisonDTypeIsValid device dtype
  , ComparisonDTypeIsValid device dtype'
  ) => Apply' BinaryCmpSpec ((Proxy device, ((Proxy dtype, Proxy dtype'), (Proxy shape, Proxy shape'))), IO ()) (IO ()) where
  apply' GTSpec (_, agg) = agg >> do
    let a = ones @shape  @dtype  @device
    let b = ones @shape' @dtype' @device
    let c = gt a b
    checkDynamicTensorAttributes c
  apply' LTSpec (_, agg) = agg >> do
    let a = ones @shape  @dtype  @device
    let b = ones @shape' @dtype' @device
    let c = lt a b
    checkDynamicTensorAttributes c
  apply' GESpec (_, agg) = agg >> do
    let a = ones @shape  @dtype  @device
    let b = ones @shape' @dtype' @device
    let c = ge a b
    checkDynamicTensorAttributes c
  apply' LESpec (_, agg) = agg >> do
    let a = ones @shape  @dtype  @device
    let b = ones @shape' @dtype' @device
    let c = le a b
    checkDynamicTensorAttributes c
  apply' EQSpec (_, agg) = agg >> do
    let a = ones @shape  @dtype  @device
    let b = ones @shape' @dtype' @device
    let c = eq a b
    checkDynamicTensorAttributes c
  apply' NESpec (_, agg) = agg >> do
    let a = ones @shape  @dtype  @device
    let b = ones @shape' @dtype' @device
    let c = ne a b
    checkDynamicTensorAttributes c

data ReshapeSpec = ReshapeSpec

instance
  ( TensorOptions fromShape dtype device
  , TensorOptions toShape   dtype device
  , KnownShape fromShape
  , KnownShape toShape
  , Numel fromShape ~ Numel toShape
  ) => Apply' ReshapeSpec ((Proxy device, (Proxy dtype, (Proxy fromShape, Proxy toShape))), IO ()) (IO ()) where
  apply' ReshapeSpec (_, agg) = agg >> do
    let t = ones @fromShape @dtype @device
    let t' = reshape @toShape t
    checkDynamicTensorAttributes t'
    let t'' = reshape @fromShape t'
    checkDynamicTensorAttributes t''

data ToTypeSpec = ToTypeSpec

instance
  ( TensorOptions shape dtype  device
  , TensorOptions shape dtype' device
  , KnownDType dtype'
  ) => Apply' ToTypeSpec ((Proxy device, ((Proxy dtype, Proxy dtype'), Proxy shape)), IO ()) (IO ()) where
  apply' ToTypeSpec (_, agg) = agg >> do
    let t = ones @shape @dtype @device
        t' = toDType @dtype' @dtype t
    checkDynamicTensorAttributes t'

data ToDeviceSpec = ToDeviceSpec

instance
  ( TensorOptions shape dtype device
  , KnownDevice device
  ) => Apply' ToDeviceSpec ((Proxy device, (Proxy dtype, Proxy shape)), IO ()) (IO ()) where
  apply' ToDeviceSpec (_, agg) = agg >> foldMap 
    (\device' -> case someDevice device' of
      (SomeDevice (Proxy :: Proxy device')) -> do
        let t = ones @shape @dtype @device
        checkDynamicTensorAttributes t
        let t' = toDevice @device' @device t
        Torch.device (toDynamic t') `shouldBe` deviceVal @device'
        let t'' = toDevice @device @device' t'
        Torch.device (toDynamic t'') `shouldBe` deviceVal @device
    )
    availableDevices

spec = foldMap spec' availableDevices

spec' :: Device -> Spec
spec' device =
  describe ("for " <> show device) $ do
    let standardShapes       = Proxy @'[2, 3] :. HNil -- (Proxy :: Proxy ('[] :: [Nat])) :. Proxy @'[0]  :. Proxy @'[0, 1] :. Proxy @'[1, 0] :. Proxy @'[2, 3] :. HNil
        broadcastableShapes0 = Proxy @'[3, 1, 4, 1] :. HNil
        broadcastableShapes1 = Proxy @'[2, 1, 1] :. HNil
        standardDTypes2      = hproduct standardDTypes            standardDTypes
        almostAllDTypes2     = hproduct (withHalf standardDTypes) (withHalf standardDTypes)
        identicalShapes      = hzip standardShapes       standardShapes
        broadcastableShapes  = hzip broadcastableShapes0 broadcastableShapes1

    describe "basic arithmetic" $ do
      let dispatch binarySpec = do
            it "works on tensors of identical shapes" $
              case device of
                Device { deviceType = CPU,  deviceIndex = 0 } ->
                  hfoldrM @IO binarySpec () (hattach cpu   (hproduct standardDTypes2  identicalShapes))
                Device { deviceType = CUDA, deviceIndex = 0 } ->
                  hfoldrM @IO binarySpec () (hattach cuda0 (hproduct almostAllDTypes2 identicalShapes))
            it "works on broadcastable tensors of different shapes" $
              case device of
                Device { deviceType = CPU,  deviceIndex = 0 } ->
                  hfoldrM @IO binarySpec () (hattach cpu   (hproduct standardDTypes2  broadcastableShapes))
                Device { deviceType = CUDA, deviceIndex = 0 } ->
                  hfoldrM @IO binarySpec () (hattach cuda0 (hproduct almostAllDTypes2 broadcastableShapes))
      describe "addition" $ dispatch AddSpec
      describe "subtraction" $ dispatch SubSpec
      describe "multiplication" $ dispatch MulSpec

    describe "matrix multiplication" $ do
      it "returns the dot product if both tensors are 1-dimensional" $ do
        let shapes = hzip (Proxy @'[3] :. HNil) (Proxy @'[3] :. HNil)
        case device of
          Device { deviceType = CPU,  deviceIndex = 0 } ->
            hfoldrM @IO MatMulSpec () (hattach cpu   (hproduct standardDTypes shapes))
          Device { deviceType = CUDA, deviceIndex = 0 } ->
            hfoldrM @IO MatMulSpec () (hattach cuda0 (hproduct allFloatingPointDTypes shapes))
      it "returns the matrix-matrix product if both arguments are 2-dimensional" $ do
        let shapes = hzip (Proxy @'[3, 2] :. HNil) (Proxy @'[2, 4] :. HNil)
        case device of
          Device { deviceType = CPU,  deviceIndex = 0 } ->
            hfoldrM @IO MatMulSpec () (hattach cpu   (hproduct standardDTypes shapes))
          Device { deviceType = CUDA, deviceIndex = 0 } ->
            hfoldrM @IO MatMulSpec () (hattach cuda0 (hproduct allFloatingPointDTypes shapes))
      it "returns the matrix-matrix product if the first argument is 1-dimensional and the second argument is 2-dimensional by temporarily adding a 1 to the dimension of the first argument" $ do
        let shapes = hzip (Proxy @'[3] :. HNil) (Proxy @'[3, 4] :. HNil)
        case device of
          Device { deviceType = CPU,  deviceIndex = 0 } ->
            hfoldrM @IO MatMulSpec () (hattach cpu   (hproduct standardDTypes shapes))
          Device { deviceType = CUDA, deviceIndex = 0 } ->
            hfoldrM @IO MatMulSpec () (hattach cuda0 (hproduct allFloatingPointDTypes shapes))
      it "returns the matrix-vector product if the first argument is 2-dimensional and the second argument is 1-dimensional" $ do
        let shapes = hzip (Proxy @'[3, 4] :. HNil) (Proxy @'[4] :. HNil)
        case device of
          Device { deviceType = CPU,  deviceIndex = 0 } ->
            hfoldrM @IO MatMulSpec () (hattach cpu   (hproduct standardDTypes shapes))
          Device { deviceType = CUDA, deviceIndex = 0 } ->
            hfoldrM @IO MatMulSpec () (hattach cuda0 (hproduct allFloatingPointDTypes shapes))
      it "returns a batched matrix-matrix product if both arguments are at least 2-dimensional and the batch (i.e. non-matrix) dimensions are broadcastable" $ do
        let shapes = hzip (Proxy @'[2, 1, 4, 3] :. HNil) (Proxy @'[3, 3, 2] :. HNil)
        case device of
          Device { deviceType = CPU,  deviceIndex = 0 } ->
            hfoldrM @IO MatMulSpec () (hattach cpu   (hproduct standardDTypes shapes))
          Device { deviceType = CUDA, deviceIndex = 0 } ->
            hfoldrM @IO MatMulSpec () (hattach cuda0 (hproduct allFloatingPointDTypes shapes))
      it "returns a batched matrix-matrix product if the first argument is 1-dimensional and the second argument has more than 2 dimensions" $ do
        let shapes = hzip (Proxy @'[3] :. HNil) (Proxy @'[2, 3, 4] :. HNil)
        case device of
          Device { deviceType = CPU,  deviceIndex = 0 } ->
            hfoldrM @IO MatMulSpec () (hattach cpu   (hproduct standardDTypes shapes))
          Device { deviceType = CUDA, deviceIndex = 0 } ->
            hfoldrM @IO MatMulSpec () (hattach cuda0 (hproduct allFloatingPointDTypes shapes))
      it "returns a batched matrix-vector product if the first argument has more than 2 dimensions and the second argument is 1-dimensional" $ do
        let shapes = hzip (Proxy @'[2, 3, 4] :. HNil) (Proxy @'[4] :. HNil)
        case device of
          Device { deviceType = CPU,  deviceIndex = 0 } ->
            hfoldrM @IO MatMulSpec () (hattach cpu   (hproduct standardDTypes shapes))
          Device { deviceType = CUDA, deviceIndex = 0 } ->
            hfoldrM @IO MatMulSpec () (hattach cuda0 (hproduct allFloatingPointDTypes shapes))

    describe "binary comparison" $ do
      let dispatch binaryCmpSpec = do
            it "works on tensors of identical shapes" $
              case device of
                Device { deviceType = CPU,  deviceIndex = 0 } ->
                  hfoldrM @IO binaryCmpSpec () (hattach cpu   (hproduct standardDTypes2  identicalShapes))
                Device { deviceType = CUDA, deviceIndex = 0 } ->
                  hfoldrM @IO binaryCmpSpec () (hattach cuda0 (hproduct almostAllDTypes2 identicalShapes))
            it "works on broadcastable tensors of different shapes" $
              case device of
                Device { deviceType = CPU,  deviceIndex = 0 } ->
                  hfoldrM @IO binaryCmpSpec () (hattach cpu   (hproduct standardDTypes2  broadcastableShapes))
                Device { deviceType = CUDA, deviceIndex = 0 } ->
                  hfoldrM @IO binaryCmpSpec () (hattach cuda0 (hproduct almostAllDTypes2 broadcastableShapes))
      describe "greater than" $ dispatch GTSpec
      describe "lower than" $ dispatch LTSpec
      describe "greater or equal than" $ dispatch GESpec
      describe "lower or equal than" $ dispatch LESpec
      describe "equal to" $ dispatch EQSpec
      describe "not equal to" $ dispatch NESpec

    describe "tensor conversion" $ do
      it "reshape" $ do
        let fromShapes = Proxy @'[0]    :. Proxy @'[0, 0] :. Proxy @'[0, 1] :. Proxy @'[1, 0] :. (Proxy :: Proxy ('[] :: [Nat])) :. Proxy @'[1]       :. Proxy @'[1, 1] :. Proxy @'[1, 1, 1]               :. Proxy @'[1, 2]    :. Proxy @'[2, 1] :. Proxy @'[1, 4, 2] :. Proxy @'[1, 1, 8] :. Proxy @'[8]       :. Proxy @'[2, 2, 2] :. HNil
            toShapes   = Proxy @'[1, 0] :. Proxy @'[0, 1] :. Proxy @'[0]    :. Proxy @'[0, 0] :. Proxy @'[1, 1]                  :. Proxy @'[1, 1, 1] :. Proxy @'[1]    :. (Proxy :: Proxy ('[] :: [Nat])) :. Proxy @'[1, 2, 1] :. Proxy @'[2]    :. Proxy @'[8]       :. Proxy @'[1, 1, 8] :. Proxy @'[2, 2, 2] :. Proxy @'[1, 1, 8] :. HNil
            shapes     = hzip fromShapes toShapes
        case device of
          Device { deviceType = CPU,  deviceIndex = 0 } ->
            hfoldrM @IO ReshapeSpec () (hattach cpu   (hproduct allDTypes shapes))
          Device { deviceType = CUDA, deviceIndex = 0 } ->
            hfoldrM @IO ReshapeSpec () (hattach cuda0 (hproduct allDTypes shapes))

      it "toDevice" $ case device of
        Device { deviceType = CPU,  deviceIndex = 0 } ->
          hfoldrM @IO ToDeviceSpec () (hattach cpu   (hproduct allDTypes standardShapes))
        Device { deviceType = CUDA, deviceIndex = 0 } ->
          hfoldrM @IO ToDeviceSpec () (hattach cuda0 (hproduct allDTypes standardShapes))
      it "toType" $ case device of
        Device { deviceType = CPU,  deviceIndex = 0 } ->
          hfoldrM @IO ToTypeSpec () (hattach cpu   (hproduct (hproduct allDTypes allDTypes) standardShapes))
        Device { deviceType = CUDA, deviceIndex = 0 } ->
          hfoldrM @IO ToTypeSpec () (hattach cuda0 (hproduct (hproduct allDTypes allDTypes) standardShapes))

testTensorListFold
  :: forall device dtype shape . Tensor device dtype shape -> IO [Torch.ATenTensor]
testTensorListFold t = hfoldrM TensorListFold ([] :: [Torch.ATenTensor]) (t :. HNil)

testTensorListUnfold
  :: forall device dtype shape device' dtype' shape'
   . [Torch.ATenTensor]
  -> IO (HList '[Tensor device dtype shape, Tensor device' dtype' shape'])
testTensorListUnfold = hunfoldrM TensorListUnfold

testCast
  :: forall device dtype shape
   . HList '[Tensor device dtype shape]
  -> IO [Torch.ATenTensor]
testCast xs = cast xs return

testUncast
  :: forall device dtype shape
   . [Torch.ATenTensor]
  -> IO (HList '[Tensor device dtype shape])
testUncast xs = uncast xs return

testReplicate
  :: forall device dtype shape
   . Tensor device dtype shape
  -> HList (HReplicateR 3 (Tensor device dtype shape))
testReplicate = hreplicate @3
