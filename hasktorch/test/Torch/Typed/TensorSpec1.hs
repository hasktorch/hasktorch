{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE FlexibleContexts #-}
{-# OPTIONS_GHC -freduction-depth=0 #-}

module Torch.Typed.TensorSpec1
  ( Torch.Typed.TensorSpec1.spec,
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

data BinaryCmpSpec = GTSpec | LTSpec | GESpec | LESpec | EQSpec | NESpec

instance
  ( TensorOptions shape dtype device,
    TensorOptions shape' dtype' device,
    TensorOptions shape'' 'Bool device,
    shape'' ~ Broadcast shape shape',
    ComparisonDTypeIsValid device dtype,
    ComparisonDTypeIsValid device dtype'
  ) =>
  Apply' BinaryCmpSpec ((Proxy device, ((Proxy dtype, Proxy dtype'), (Proxy shape, Proxy shape'))), IO ()) (IO ())
  where
  apply' GTSpec (_, agg) =
    agg >> do
      let a = ones @shape @dtype @device
      let b = ones @shape' @dtype' @device
      let c = gt a b
      checkDynamicTensorAttributes c
  apply' LTSpec (_, agg) =
    agg >> do
      let a = ones @shape @dtype @device
      let b = ones @shape' @dtype' @device
      let c = lt a b
      checkDynamicTensorAttributes c
  apply' GESpec (_, agg) =
    agg >> do
      let a = ones @shape @dtype @device
      let b = ones @shape' @dtype' @device
      let c = ge a b
      checkDynamicTensorAttributes c
  apply' LESpec (_, agg) =
    agg >> do
      let a = ones @shape @dtype @device
      let b = ones @shape' @dtype' @device
      let c = le a b
      checkDynamicTensorAttributes c
  apply' EQSpec (_, agg) =
    agg >> do
      let a = ones @shape @dtype @device
      let b = ones @shape' @dtype' @device
      let c = eq a b
      checkDynamicTensorAttributes c
  apply' NESpec (_, agg) =
    agg >> do
      let a = ones @shape @dtype @device
      let b = ones @shape' @dtype' @device
      let c = ne a b
      checkDynamicTensorAttributes c

data ReshapeSpec = ReshapeSpec

instance
  ( TensorOptions fromShape dtype device,
    TensorOptions toShape dtype device,
    KnownShape fromShape,
    KnownShape toShape,
    Numel fromShape ~ Numel toShape
  ) =>
  Apply' ReshapeSpec ((Proxy device, (Proxy dtype, (Proxy fromShape, Proxy toShape))), IO ()) (IO ())
  where
  apply' ReshapeSpec (_, agg) =
    agg >> do
      let t = ones @fromShape @dtype @device
      let t' = reshape @toShape t
      checkDynamicTensorAttributes t'
      let t'' = reshape @fromShape t'
      checkDynamicTensorAttributes t''

data ToTypeSpec = ToTypeSpec

instance
  ( TensorOptions shape dtype device,
    TensorOptions shape dtype' device,
    KnownDType dtype'
  ) =>
  Apply' ToTypeSpec ((Proxy device, ((Proxy dtype, Proxy dtype'), Proxy shape)), IO ()) (IO ())
  where
  apply' ToTypeSpec (_, agg) =
    agg >> do
      let t = ones @shape @dtype @device
          t' = toDType @dtype' @dtype t
      checkDynamicTensorAttributes t'

data ToDeviceSpec = ToDeviceSpec

instance
  ( TensorOptions shape dtype device,
    KnownDevice device
  ) =>
  Apply' ToDeviceSpec ((Proxy device, (Proxy dtype, Proxy shape)), IO ()) (IO ())
  where
  apply' ToDeviceSpec (_, agg) =
    agg
      >> foldMap
        ( \device' -> case someDevice device' of
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
    let standardShapes = Proxy @'[2, 3] :. HNil -- (Proxy :: Proxy ('[] :: [Nat])) :. Proxy @'[0]  :. Proxy @'[0, 1] :. Proxy @'[1, 0] :. Proxy @'[2, 3] :. HNil
        broadcastableShapes0 = Proxy @'[3, 1, 4, 1] :. HNil
        broadcastableShapes1 = Proxy @'[2, 1, 1] :. HNil
        standardDTypes2 = hproduct standardDTypes standardDTypes
        almostAllDTypes2 = hproduct (withHalf standardDTypes) (withHalf standardDTypes)
        identicalShapes = hzip standardShapes standardShapes
        broadcastableShapes = hzip broadcastableShapes0 broadcastableShapes1

    describe "binary comparison" $ do
      let dispatch binaryCmpSpec = do
            it "works on tensors of identical shapes" $
              case device of
                Device {deviceType = CPU, deviceIndex = 0} ->
                  hfoldrM @IO binaryCmpSpec () (hattach cpu (hproduct standardDTypes2 identicalShapes))
                Device {deviceType = CUDA, deviceIndex = 0} ->
                  hfoldrM @IO binaryCmpSpec () (hattach cuda0 (hproduct almostAllDTypes2 identicalShapes))
            it "works on broadcastable tensors of different shapes" $
              case device of
                Device {deviceType = CPU, deviceIndex = 0} ->
                  hfoldrM @IO binaryCmpSpec () (hattach cpu (hproduct standardDTypes2 broadcastableShapes))
                Device {deviceType = CUDA, deviceIndex = 0} ->
                  hfoldrM @IO binaryCmpSpec () (hattach cuda0 (hproduct almostAllDTypes2 broadcastableShapes))
      describe "greater than" $ dispatch GTSpec
      describe "lower than" $ dispatch LTSpec
      describe "greater or equal than" $ dispatch GESpec
      describe "lower or equal than" $ dispatch LESpec
      describe "equal to" $ dispatch EQSpec
      describe "not equal to" $ dispatch NESpec

    describe "tensor conversion" $ do
      it "reshape" $ do
        let fromShapes = Proxy @'[0] :. Proxy @'[0, 0] :. Proxy @'[0, 1] :. Proxy @'[1, 0] :. (Proxy :: Proxy ('[] :: [Nat])) :. Proxy @'[1] :. Proxy @'[1, 1] :. Proxy @'[1, 1, 1] :. Proxy @'[1, 2] :. Proxy @'[2, 1] :. Proxy @'[1, 4, 2] :. Proxy @'[1, 1, 8] :. Proxy @'[8] :. Proxy @'[2, 2, 2] :. HNil
            toShapes = Proxy @'[1, 0] :. Proxy @'[0, 1] :. Proxy @'[0] :. Proxy @'[0, 0] :. Proxy @'[1, 1] :. Proxy @'[1, 1, 1] :. Proxy @'[1] :. (Proxy :: Proxy ('[] :: [Nat])) :. Proxy @'[1, 2, 1] :. Proxy @'[2] :. Proxy @'[8] :. Proxy @'[1, 1, 8] :. Proxy @'[2, 2, 2] :. Proxy @'[1, 1, 8] :. HNil
            shapes = hzip fromShapes toShapes
        case device of
          Device {deviceType = CPU, deviceIndex = 0} ->
            hfoldrM @IO ReshapeSpec () (hattach cpu (hproduct allDTypes shapes))
          Device {deviceType = CUDA, deviceIndex = 0} ->
            hfoldrM @IO ReshapeSpec () (hattach cuda0 (hproduct allDTypes shapes))

      it "toDevice" $ case device of
        Device {deviceType = CPU, deviceIndex = 0} ->
          hfoldrM @IO ToDeviceSpec () (hattach cpu (hproduct allDTypes standardShapes))
        Device {deviceType = CUDA, deviceIndex = 0} ->
          hfoldrM @IO ToDeviceSpec () (hattach cuda0 (hproduct allDTypes standardShapes))
      it "toType" $ case device of
        Device {deviceType = CPU, deviceIndex = 0} ->
          hfoldrM @IO ToTypeSpec () (hattach cpu (hproduct (hproduct allDTypes allDTypes) standardShapes))
        Device {deviceType = CUDA, deviceIndex = 0} ->
          hfoldrM @IO ToTypeSpec () (hattach cuda0 (hproduct (hproduct allDTypes allDTypes) standardShapes))

    describe "untyped to typed tensor" $ do
      it "withTensor" $ do
        withTensor (Torch.zeros' [2, 3, 4]) $ \t -> print t
      it "withTensorShape with matmul" $ do
        --    ToDo: withTensor does not work with matmul.
        --        withTensor (Torch.zeros' [3,4]) $ \(t0 :: Tensor device0 dtype0 shape0)->
        --          withTensor (Torch.zeros' [4,3]) $ \(t1 :: Tensor device1 dtype1 shape1)-> do
        --            print (matmul t0 t1)
        withTensorShape @'(CPU, 0) @'Float (Torch.zeros' [3, 4]) $ \t0 -> do
          withTensorShape @'(CPU, 0) @'Float (Torch.zeros' [4, 3]) $ \t1 -> do
            print (matmul t0 t1)
      it "withNat" $ do
        withNat 2 $ \(_ :: Proxy n) -> print $ (zeros :: Tensor '(CPU, 0) 'Float [n, 2, 3])

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
