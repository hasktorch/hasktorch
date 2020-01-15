{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE UndecidableSuperClasses #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE NoStarIsType #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# OPTIONS_GHC -Wno-partial-type-signatures #-}
{-# OPTIONS_GHC -freduction-depth=0 #-}

module Torch.Typed.TensorSpec
  ( Torch.Typed.TensorSpec.spec
  )
where

import           Prelude                 hiding ( sin )
import           Control.Exception.Safe
import           Foreign.Storable
import           Torch.HList
import           Data.Kind
import           Data.Proxy
import           Data.Reflection
import           GHC.TypeLits

import           Test.Hspec
import           Test.QuickCheck

import           Torch.Internal.Class                     ( Castable(..) )
import qualified Torch.Device                  as D
import qualified Torch.DType                   as D
import qualified Torch.Functional               as D
import qualified Torch.Tensor                  as D
import qualified Torch.TensorFactories         as D
import qualified Torch.TensorOptions           as D
import           Torch.Typed.Aux
import           Torch.Typed.Factories
import           Torch.Typed.Functional
import           Torch.Typed.Tensor
import           Torch.Typed.AuxSpec

data BinarySpec = AddSpec | SubSpec | MulSpec

instance ( TensorOptions shape   dtype   device
         , TensorOptions shape'  dtype'  device
         , TensorOptions shape'' dtype'' device
         , shape'' ~ Broadcast shape shape'
         , dtype'' ~ DTypePromotion dtype dtype'
         , BasicArithmeticDTypeIsValid device dtype
         , BasicArithmeticDTypeIsValid device dtype'
         , BasicArithmeticDTypeIsValid device dtype''
         )
  => Apply
       BinarySpec
       (Proxy device, ((Proxy dtype, Proxy dtype'), (Proxy shape, Proxy shape')))
       (() -> IO ())
 where
  apply AddSpec _ _ = do
    let a = ones @shape  @dtype  @device
    let b = ones @shape' @dtype' @device
    let c = add a b
    checkDynamicTensorAttributes c
  apply SubSpec _ _ = do
    let a = ones @shape  @dtype  @device
    let b = ones @shape' @dtype' @device
    let c = sub a b
    checkDynamicTensorAttributes c
  apply MulSpec _ _ = do
    let a = ones @shape  @dtype  @device
    let b = ones @shape' @dtype' @device
    let c = mul a b
    checkDynamicTensorAttributes c

data MatMulSpec = MatMulSpec

instance ( TensorOptions shape   dtype device
         , TensorOptions shape'  dtype device
         , TensorOptions shape'' dtype device
         , shape'' ~ MatMul shape shape'
         , MatMulDTypeIsValid device dtype
         )
  => Apply
       MatMulSpec
       (Proxy device, (Proxy dtype, (Proxy shape, Proxy shape')))
       (() -> IO ())
 where
  apply MatMulSpec _ _ = do
    let a = ones @shape  @dtype @device
    let b = ones @shape' @dtype @device
    let c = matmul a b
    checkDynamicTensorAttributes c

data BinaryCmpSpec = GTSpec | LTSpec | GESpec | LESpec | EQSpec | NESpec

instance ( TensorOptions shape   dtype  device
         , TensorOptions shape'  dtype' device
         , TensorOptions shape'' D.Bool device
         , shape'' ~ Broadcast shape shape'
         , ComparisonDTypeIsValid device dtype
         , ComparisonDTypeIsValid device dtype'
         )
  => Apply
       BinaryCmpSpec
       (Proxy device, ((Proxy dtype, Proxy dtype'), (Proxy shape, Proxy shape')))
       (() -> IO ())
 where
  apply GTSpec _ _ = do
    let a = ones @shape  @dtype  @device
    let b = ones @shape' @dtype' @device
    let c = gt a b
    checkDynamicTensorAttributes c
  apply LTSpec _ _ = do
    let a = ones @shape  @dtype  @device
    let b = ones @shape' @dtype' @device
    let c = lt a b
    checkDynamicTensorAttributes c
  apply GESpec _ _ = do
    let a = ones @shape  @dtype  @device
    let b = ones @shape' @dtype' @device
    let c = ge a b
    checkDynamicTensorAttributes c
  apply LESpec _ _ = do
    let a = ones @shape  @dtype  @device
    let b = ones @shape' @dtype' @device
    let c = le a b
    checkDynamicTensorAttributes c
  apply EQSpec _ _ = do
    let a = ones @shape  @dtype  @device
    let b = ones @shape' @dtype' @device
    let c = eq a b
    checkDynamicTensorAttributes c
  apply NESpec _ _ = do
    let a = ones @shape  @dtype  @device
    let b = ones @shape' @dtype' @device
    let c = ne a b
    checkDynamicTensorAttributes c

data ReshapeSpec = ReshapeSpec

instance ( TensorOptions fromShape dtype device
         , TensorOptions toShape   dtype device
         , KnownShape fromShape
         , KnownShape toShape
         , Numel fromShape ~ Numel toShape
         )
  => Apply
       ReshapeSpec
       (Proxy device, (Proxy dtype, (Proxy fromShape, Proxy toShape)))
       (() -> IO ())
 where
  apply ReshapeSpec _ _ = do
    let t = ones @fromShape @dtype @device
    let t' = reshape @toShape t
    checkDynamicTensorAttributes t'
    let t'' = reshape @fromShape t'
    checkDynamicTensorAttributes t''

data ToTypeSpec = ToTypeSpec

instance ( TensorOptions shape dtype  device
         , TensorOptions shape dtype' device
         , KnownDType dtype'
         )
  => Apply
       ToTypeSpec
       (Proxy device, ((Proxy dtype, Proxy dtype'), Proxy shape))
       (() -> IO ())
 where
  apply ToTypeSpec _ _ = do
    let t = ones @shape @dtype @device
        t' = Torch.Typed.Tensor.toDType @dtype' t
    checkDynamicTensorAttributes t'

data ToDeviceSpec = ToDeviceSpec

instance ( TensorOptions shape dtype device
         , KnownDevice device
         )
  => Apply
       ToDeviceSpec
       (Proxy device, (Proxy dtype, Proxy shape))
       (() -> IO ())
 where
  apply ToDeviceSpec _ _ = 
    foldMap 
      (\device' -> case someDevice device' of
        (SomeDevice (Proxy :: Proxy device')) -> do
          let t = ones @shape @dtype @device
          checkDynamicTensorAttributes t
          let t' = toDevice @device' t
          D.device (toDynamic t') `shouldBe` deviceVal @device'
          let t'' = toDevice @device t'
          D.device (toDynamic t'') `shouldBe` deviceVal @device
      )
      availableDevices

spec = foldMap spec' availableDevices

spec' :: D.Device -> Spec
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
                D.Device { D.deviceType = D.CPU,  D.deviceIndex = 0 } ->
                  hfoldrM @IO binarySpec () (hattach cpu   (hproduct standardDTypes2  identicalShapes))
                D.Device { D.deviceType = D.CUDA, D.deviceIndex = 0 } ->
                  hfoldrM @IO binarySpec () (hattach cuda0 (hproduct almostAllDTypes2 identicalShapes))
            it "works on broadcastable tensors of different shapes" $
              case device of
                D.Device { D.deviceType = D.CPU,  D.deviceIndex = 0 } ->
                  hfoldrM @IO binarySpec () (hattach cpu   (hproduct standardDTypes2  broadcastableShapes))
                D.Device { D.deviceType = D.CUDA, D.deviceIndex = 0 } ->
                  hfoldrM @IO binarySpec () (hattach cuda0 (hproduct almostAllDTypes2 broadcastableShapes))
      describe "addition" $ dispatch AddSpec
      describe "subtraction" $ dispatch SubSpec
      describe "multiplication" $ dispatch MulSpec

    describe "matrix multiplication" $ do
      it "returns the dot product if both tensors are 1-dimensional" $ do
        let shapes = hzip (Proxy @'[3] :. HNil) (Proxy @'[3] :. HNil)
        case device of
          D.Device { D.deviceType = D.CPU,  D.deviceIndex = 0 } ->
            hfoldrM @IO MatMulSpec () (hattach cpu   (hproduct standardDTypes shapes))
          D.Device { D.deviceType = D.CUDA, D.deviceIndex = 0 } ->
            hfoldrM @IO MatMulSpec () (hattach cuda0 (hproduct allFloatingPointDTypes shapes))
      it "returns the matrix-matrix product if both arguments are 2-dimensional" $ do
        let shapes = hzip (Proxy @'[3, 2] :. HNil) (Proxy @'[2, 4] :. HNil)
        case device of
          D.Device { D.deviceType = D.CPU,  D.deviceIndex = 0 } ->
            hfoldrM @IO MatMulSpec () (hattach cpu   (hproduct standardDTypes shapes))
          D.Device { D.deviceType = D.CUDA, D.deviceIndex = 0 } ->
            hfoldrM @IO MatMulSpec () (hattach cuda0 (hproduct allFloatingPointDTypes shapes))
      it "returns the matrix-matrix product if the first argument is 1-dimensional and the second argument is 2-dimensional by temporarily adding a 1 to the dimension of the first argument" $ do
        let shapes = hzip (Proxy @'[3] :. HNil) (Proxy @'[3, 4] :. HNil)
        case device of
          D.Device { D.deviceType = D.CPU,  D.deviceIndex = 0 } ->
            hfoldrM @IO MatMulSpec () (hattach cpu   (hproduct standardDTypes shapes))
          D.Device { D.deviceType = D.CUDA, D.deviceIndex = 0 } ->
            hfoldrM @IO MatMulSpec () (hattach cuda0 (hproduct allFloatingPointDTypes shapes))
      it "returns the matrix-vector product if the first argument is 2-dimensional and the second argument is 1-dimensional" $ do
        let shapes = hzip (Proxy @'[3, 4] :. HNil) (Proxy @'[4] :. HNil)
        case device of
          D.Device { D.deviceType = D.CPU,  D.deviceIndex = 0 } ->
            hfoldrM @IO MatMulSpec () (hattach cpu   (hproduct standardDTypes shapes))
          D.Device { D.deviceType = D.CUDA, D.deviceIndex = 0 } ->
            hfoldrM @IO MatMulSpec () (hattach cuda0 (hproduct allFloatingPointDTypes shapes))
      it "returns a batched matrix-matrix product if both arguments are at least 2-dimensional and the batch (i.e. non-matrix) dimensions are broadcastable" $ do
        let shapes = hzip (Proxy @'[2, 1, 4, 3] :. HNil) (Proxy @'[3, 3, 2] :. HNil)
        case device of
          D.Device { D.deviceType = D.CPU,  D.deviceIndex = 0 } ->
            hfoldrM @IO MatMulSpec () (hattach cpu   (hproduct standardDTypes shapes))
          D.Device { D.deviceType = D.CUDA, D.deviceIndex = 0 } ->
            hfoldrM @IO MatMulSpec () (hattach cuda0 (hproduct allFloatingPointDTypes shapes))
      it "returns a batched matrix-matrix product if the first argument is 1-dimensional and the second argument has more than 2 dimensions" $ do
        let shapes = hzip (Proxy @'[3] :. HNil) (Proxy @'[2, 3, 4] :. HNil)
        case device of
          D.Device { D.deviceType = D.CPU,  D.deviceIndex = 0 } ->
            hfoldrM @IO MatMulSpec () (hattach cpu   (hproduct standardDTypes shapes))
          D.Device { D.deviceType = D.CUDA, D.deviceIndex = 0 } ->
            hfoldrM @IO MatMulSpec () (hattach cuda0 (hproduct allFloatingPointDTypes shapes))
      it "returns a batched matrix-vector product if the first argument has more than 2 dimensions and the second argument is 1-dimensional" $ do
        let shapes = hzip (Proxy @'[2, 3, 4] :. HNil) (Proxy @'[4] :. HNil)
        case device of
          D.Device { D.deviceType = D.CPU,  D.deviceIndex = 0 } ->
            hfoldrM @IO MatMulSpec () (hattach cpu   (hproduct standardDTypes shapes))
          D.Device { D.deviceType = D.CUDA, D.deviceIndex = 0 } ->
            hfoldrM @IO MatMulSpec () (hattach cuda0 (hproduct allFloatingPointDTypes shapes))

    describe "binary comparison" $ do
      let dispatch binaryCmpSpec = do
            it "works on tensors of identical shapes" $
              case device of
                D.Device { D.deviceType = D.CPU,  D.deviceIndex = 0 } ->
                  hfoldrM @IO binaryCmpSpec () (hattach cpu   (hproduct standardDTypes2  identicalShapes))
                D.Device { D.deviceType = D.CUDA, D.deviceIndex = 0 } ->
                  hfoldrM @IO binaryCmpSpec () (hattach cuda0 (hproduct almostAllDTypes2 identicalShapes))
            it "works on broadcastable tensors of different shapes" $
              case device of
                D.Device { D.deviceType = D.CPU,  D.deviceIndex = 0 } ->
                  hfoldrM @IO binaryCmpSpec () (hattach cpu   (hproduct standardDTypes2  broadcastableShapes))
                D.Device { D.deviceType = D.CUDA, D.deviceIndex = 0 } ->
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
          D.Device { D.deviceType = D.CPU,  D.deviceIndex = 0 } ->
            hfoldrM @IO ReshapeSpec () (hattach cpu   (hproduct allDTypes shapes))
          D.Device { D.deviceType = D.CUDA, D.deviceIndex = 0 } ->
            hfoldrM @IO ReshapeSpec () (hattach cuda0 (hproduct allDTypes shapes))

      it "toDevice" $ case device of
        D.Device { D.deviceType = D.CPU,  D.deviceIndex = 0 } ->
          hfoldrM @IO ToDeviceSpec () (hattach cpu   (hproduct allDTypes standardShapes))
        D.Device { D.deviceType = D.CUDA, D.deviceIndex = 0 } ->
          hfoldrM @IO ToDeviceSpec () (hattach cuda0 (hproduct allDTypes standardShapes))
      it "toType" $ case device of
        D.Device { D.deviceType = D.CPU,  D.deviceIndex = 0 } ->
          hfoldrM @IO ToTypeSpec () (hattach cpu   (hproduct (hproduct allDTypes allDTypes) standardShapes))
        D.Device { D.deviceType = D.CUDA, D.deviceIndex = 0 } ->
          hfoldrM @IO ToTypeSpec () (hattach cuda0 (hproduct (hproduct allDTypes allDTypes) standardShapes))

testTensorListFold
  :: forall device dtype shape . Tensor device dtype shape -> IO [D.ATenTensor]
testTensorListFold t = hfoldrM TensorListFold [] (t :. HNil)

testTensorListUnfold
  :: forall device dtype shape device' dtype' shape'
   . [D.ATenTensor]
  -> IO (HList '[Tensor device dtype shape, Tensor device' dtype' shape'])
testTensorListUnfold = hunfoldrM TensorListUnfold

testCast
  :: forall device dtype shape
   . HList '[Tensor device dtype shape]
  -> IO [D.ATenTensor]
testCast xs = cast xs return

testUncast
  :: forall device dtype shape
   . [D.ATenTensor]
  -> IO (HList '[Tensor device dtype shape])
testUncast xs = uncast xs return

testReplicate
  :: forall device dtype shape
   . Tensor device dtype shape
  -> HList (HReplicateR 3 (Tensor device dtype shape))
testReplicate = hreplicate @3
