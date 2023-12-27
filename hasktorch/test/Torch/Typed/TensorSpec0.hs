{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# OPTIONS_GHC -freduction-depth=0 #-}

module Torch.Typed.TensorSpec0
  ( Torch.Typed.TensorSpec0.spec,
  )
where

import Data.Kind
import Data.Proxy
import GHC.TypeLits
import Test.Hspec (Spec, describe, it, shouldBe)
import Test.QuickCheck ()
import qualified Torch as Torch
import Torch.Internal.Class (Castable (cast), uncast)
import Torch.Typed
import Torch.Typed.AuxiliarySpec

data BinarySpec = AddSpec | SubSpec | MulSpec

instance
  ( TensorOptions shape dtype device,
    TensorOptions shape' dtype' device,
    TensorOptions shape'' dtype'' device,
    shape'' ~ Broadcast shape shape',
    dtype'' ~ DTypePromotion dtype dtype',
    BasicArithmeticDTypeIsValid device dtype,
    BasicArithmeticDTypeIsValid device dtype',
    BasicArithmeticDTypeIsValid device dtype''
  ) =>
  Apply' BinarySpec ((Proxy device, ((Proxy dtype, Proxy dtype'), (Proxy shape, Proxy shape'))), IO ()) (IO ())
  where
  apply' AddSpec (_, agg) =
    agg >> do
      let a = ones @shape @dtype @device
      let b = ones @shape' @dtype' @device
      let c = add a b
      checkDynamicTensorAttributes c
  apply' SubSpec (_, agg) =
    agg >> do
      let a = ones @shape @dtype @device
      let b = ones @shape' @dtype' @device
      let c = sub a b
      checkDynamicTensorAttributes c
  apply' MulSpec (_, agg) =
    agg >> do
      let a = ones @shape @dtype @device
      let b = ones @shape' @dtype' @device
      let c = mul a b
      checkDynamicTensorAttributes c

data MatMulSpec = MatMulSpec

instance
  ( TensorOptions shape dtype device,
    TensorOptions shape' dtype device,
    TensorOptions shape'' dtype device,
    shape'' ~ MatMul shape shape',
    MatMulDTypeIsValid device dtype
  ) =>
  Apply' MatMulSpec ((Proxy device, (Proxy dtype, (Proxy shape, Proxy shape'))), IO ()) (IO ())
  where
  apply' MatMulSpec (_, agg) =
    agg >> do
      let a = ones @shape @dtype @device
      let b = ones @shape' @dtype @device
      let c = matmul a b
      checkDynamicTensorAttributes c

spec = foldMap spec' availableDevices

spec' :: Device -> Spec
spec' device =
  describe ("for " <> show device) $ do
    let standardShapes = Proxy @'[2, 3] :. HNil -- (Proxy :: Proxy ('[] :: [Nat])) :. Proxy @'[0]  :. Proxy @'[0, 1] :. Proxy @'[1, 0] :. Proxy @'[2, 3] :. HNil
        broadcastableShapes0 = Proxy @'[3, 1, 4, 1] :. HNil
        broadcastableShapes1 = Proxy @'[2, 1, 1] :. HNil
        standardDTypes2 = hproduct standardDTypes standardDTypes
        almostAllDTypes2 = hproduct (withHalf standardDTypes) (withHalf standardDTypes)
        identicalShapes = hzip standardShapes standardShapes
        broadcastableShapes = hzip broadcastableShapes0 broadcastableShapes1

    describe "basic arithmetic" $ do
      let dispatch binarySpec = do
            it "works on tensors of identical shapes" $
              case device of
                Device {deviceType = CPU, deviceIndex = 0} ->
                  hfoldrM @IO binarySpec () (hattach cpu (hproduct standardDTypes2 identicalShapes))
                Device {deviceType = CUDA, deviceIndex = 0} ->
                  hfoldrM @IO binarySpec () (hattach cuda0 (hproduct almostAllDTypes2 identicalShapes))
            it "works on broadcastable tensors of different shapes" $
              case device of
                Device {deviceType = CPU, deviceIndex = 0} ->
                  hfoldrM @IO binarySpec () (hattach cpu (hproduct standardDTypes2 broadcastableShapes))
                Device {deviceType = CUDA, deviceIndex = 0} ->
                  hfoldrM @IO binarySpec () (hattach cuda0 (hproduct almostAllDTypes2 broadcastableShapes))
      describe "addition" $ dispatch AddSpec
      describe "subtraction" $ dispatch SubSpec
      describe "multiplication" $ dispatch MulSpec

    describe "matrix multiplication" $ do
      it "returns the dot product if both tensors are 1-dimensional" $ do
        let shapes = hzip (Proxy @'[3] :. HNil) (Proxy @'[3] :. HNil)
        case device of
          Device {deviceType = CPU, deviceIndex = 0} ->
            hfoldrM @IO MatMulSpec () (hattach cpu (hproduct standardDTypes shapes))
          Device {deviceType = CUDA, deviceIndex = 0} ->
            hfoldrM @IO MatMulSpec () (hattach cuda0 (hproduct allFloatingPointDTypes shapes))
      it "returns the matrix-matrix product if both arguments are 2-dimensional" $ do
        let shapes = hzip (Proxy @'[3, 2] :. HNil) (Proxy @'[2, 4] :. HNil)
        case device of
          Device {deviceType = CPU, deviceIndex = 0} ->
            hfoldrM @IO MatMulSpec () (hattach cpu (hproduct standardDTypes shapes))
          Device {deviceType = CUDA, deviceIndex = 0} ->
            hfoldrM @IO MatMulSpec () (hattach cuda0 (hproduct allFloatingPointDTypes shapes))
      it "returns the matrix-matrix product if the first argument is 1-dimensional and the second argument is 2-dimensional by temporarily adding a 1 to the dimension of the first argument" $ do
        let shapes = hzip (Proxy @'[3] :. HNil) (Proxy @'[3, 4] :. HNil)
        case device of
          Device {deviceType = CPU, deviceIndex = 0} ->
            hfoldrM @IO MatMulSpec () (hattach cpu (hproduct standardDTypes shapes))
          Device {deviceType = CUDA, deviceIndex = 0} ->
            hfoldrM @IO MatMulSpec () (hattach cuda0 (hproduct allFloatingPointDTypes shapes))
      it "returns the matrix-vector product if the first argument is 2-dimensional and the second argument is 1-dimensional" $ do
        let shapes = hzip (Proxy @'[3, 4] :. HNil) (Proxy @'[4] :. HNil)
        case device of
          Device {deviceType = CPU, deviceIndex = 0} ->
            hfoldrM @IO MatMulSpec () (hattach cpu (hproduct standardDTypes shapes))
          Device {deviceType = CUDA, deviceIndex = 0} ->
            hfoldrM @IO MatMulSpec () (hattach cuda0 (hproduct allFloatingPointDTypes shapes))
      it "returns a batched matrix-matrix product if both arguments are at least 2-dimensional and the batch (i.e. non-matrix) dimensions are broadcastable" $ do
        let shapes = hzip (Proxy @'[2, 1, 4, 3] :. HNil) (Proxy @'[3, 3, 2] :. HNil)
        case device of
          Device {deviceType = CPU, deviceIndex = 0} ->
            hfoldrM @IO MatMulSpec () (hattach cpu (hproduct standardDTypes shapes))
          Device {deviceType = CUDA, deviceIndex = 0} ->
            hfoldrM @IO MatMulSpec () (hattach cuda0 (hproduct allFloatingPointDTypes shapes))
      it "returns a batched matrix-matrix product if the first argument is 1-dimensional and the second argument has more than 2 dimensions" $ do
        let shapes = hzip (Proxy @'[3] :. HNil) (Proxy @'[2, 3, 4] :. HNil)
        case device of
          Device {deviceType = CPU, deviceIndex = 0} ->
            hfoldrM @IO MatMulSpec () (hattach cpu (hproduct standardDTypes shapes))
          Device {deviceType = CUDA, deviceIndex = 0} ->
            hfoldrM @IO MatMulSpec () (hattach cuda0 (hproduct allFloatingPointDTypes shapes))
      it "returns a batched matrix-vector product if the first argument has more than 2 dimensions and the second argument is 1-dimensional" $ do
        let shapes = hzip (Proxy @'[2, 3, 4] :. HNil) (Proxy @'[4] :. HNil)
        case device of
          Device {deviceType = CPU, deviceIndex = 0} ->
            hfoldrM @IO MatMulSpec () (hattach cpu (hproduct standardDTypes shapes))
          Device {deviceType = CUDA, deviceIndex = 0} ->
            hfoldrM @IO MatMulSpec () (hattach cuda0 (hproduct allFloatingPointDTypes shapes))

testTensorListFold ::
  forall device dtype shape. Tensor device dtype shape -> IO [Torch.ATenTensor]
testTensorListFold t = hfoldrM TensorListFold ([] :: [Torch.ATenTensor]) (t :. HNil)

testTensorListUnfold ::
  forall device dtype shape device' dtype' shape'.
  [Torch.ATenTensor] ->
  IO (HList '[Tensor device dtype shape, Tensor device' dtype' shape'])
testTensorListUnfold = hunfoldrM TensorListUnfold

testCast ::
  forall device dtype shape.
  HList '[Tensor device dtype shape] ->
  IO [Torch.ATenTensor]
testCast xs = cast xs return

testUncast ::
  forall device dtype shape.
  [Torch.ATenTensor] ->
  IO (HList '[Tensor device dtype shape])
testUncast xs = uncast xs return

testReplicate ::
  forall device dtype shape.
  Tensor device dtype shape ->
  HList (HReplicateR 3 (Tensor device dtype shape))
testReplicate = hreplicate @3
