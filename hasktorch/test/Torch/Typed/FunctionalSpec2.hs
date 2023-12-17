{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE NoStarIsType #-}
{-# LANGUAGE FlexibleContexts #-}
{-# OPTIONS_GHC -freduction-depth=0 #-}

module Torch.Typed.FunctionalSpec2
  ( Torch.Typed.FunctionalSpec2.spec,
  )
where

import Data.Proxy
import GHC.TypeLits
import Test.Hspec (Spec, before_, describe, it)
import Test.QuickCheck ()
import Torch.Internal.Managed.Type.Context (get_manual_seed)
import Torch.Typed
import Torch.Typed.AuxiliarySpec
import Prelude hiding
  ( abs,
    acos,
    acosh,
    all,
    any,
    asin,
    asinh,
    atan,
    atanh,
    cos,
    cosh,
    exp,
    floor,
    log,
    max,
    min,
    round,
    sin,
    sinh,
    sqrt,
    tan,
    tanh,
  )

data LossSpec
  = BinaryCrossEntropySpec
  | MSELossSpec

instance
  ( TensorOptions shape dtype device,
    TensorOptions shape' dtype device,
    KnownReduction reduction,
    shape' ~ ConditionalReduction shape reduction,
    StandardFloatingPointDTypeValidation device dtype
  ) =>
  Apply' LossSpec ((Proxy reduction, (Proxy device, (Proxy dtype, Proxy shape))), IO ()) (IO ())
  where
  apply' BinaryCrossEntropySpec (_, agg) =
    agg >> do
      let weight = ones @shape @dtype @device
          prediction = ones @shape @dtype @device
          target = ones @shape @dtype @device
          t = binaryCrossEntropy @reduction weight prediction target
      checkDynamicTensorAttributes t
  apply' MSELossSpec (_, agg) =
    agg >> do
      let prediction = ones @shape @dtype @device
          target = ones @shape @dtype @device
          t = mseLoss @reduction prediction target
      checkDynamicTensorAttributes t

data SoftmaxSpec
  = SoftmaxSpec
  | LogSoftmaxSpec

instance
  ( TensorOptions shape dtype device,
    KnownNat dim,
    DimOutOfBoundCheck shape dim,
    KnownDType dtype,
    StandardFloatingPointDTypeValidation device dtype
  ) =>
  Apply' SoftmaxSpec ((Proxy dim, (Proxy device, (Proxy dtype, Proxy shape))), IO ()) (IO ())
  where
  apply' SoftmaxSpec (_, agg) =
    agg >> do
      let t = ones @shape @dtype @device
          t' = softmax @dim t
      checkDynamicTensorAttributes t'
  apply' LogSoftmaxSpec (_, agg) =
    agg >> do
      let t = ones @shape @dtype @device
          t' = logSoftmax @dim t
      checkDynamicTensorAttributes t'

data DotSpec = DotSpec

instance
  ( TensorOptions '[size] dtype device,
    DotDTypeIsValid device dtype,
    KnownDType dtype,
    KnownDevice device
  ) =>
  Apply' DotSpec ((Proxy device, (Proxy dtype, Proxy size)), IO ()) (IO ())
  where
  apply' DotSpec (_, agg) =
    agg >> do
      let a = ones @'[size] @dtype @device
          b = ones @'[size] @dtype @device
          t = dot a b
      checkDynamicTensorAttributes t

data InverseSpec = InverseSpec

instance
  ( TensorOptions shape dtype device,
    TensorOptions shape' dtype device,
    shape' ~ Square shape,
    InverseShapeIsValid device shape,
    InverseDTypeIsValid device dtype,
    RandDTypeIsValid device dtype
  ) =>
  Apply' InverseSpec ((Proxy device, (Proxy dtype, Proxy shape)), IO ()) (IO ())
  where
  apply' InverseSpec (_, agg) =
    agg >> do
      t <- rand @shape @dtype @device
      let t' = inverse t
      checkDynamicTensorAttributes t'

data SymeigSpec = SymeigSpec | SymeigvaluesSpec

instance
  ( TensorOptions shape dtype device,
    TensorOptions shape' dtype device,
    TensorOptions shape'' dtype device,
    shape' ~ VectorOfSquare shape,
    shape'' ~ Square shape,
    SymeigDTypeIsValid device dtype,
    RandDTypeIsValid device dtype
  ) =>
  Apply' SymeigSpec ((Proxy device, (Proxy dtype, Proxy shape)), IO ()) (IO ())
  where
  apply' SymeigSpec (_, agg) =
    agg >> do
      t <- rand @shape @dtype @device
      foldMap
        ( \upper -> do
            let (t', t'') = symeig upper t
            checkDynamicTensorAttributes t'
            checkDynamicTensorAttributes t''
        )
        [Upper, Lower]
  apply' SymeigvaluesSpec (_, agg) =
    agg >> do
      t <- rand @shape @dtype @device
      foldMap
        ( \upper -> do
            let t' = symeigvalues upper t
            checkDynamicTensorAttributes t'
        )
        [Upper, Lower]

data EigSpec = EigSpec

instance
  ( TensorOptions shape dtype device,
    TensorOptions shape' dtype device,
    shape ~ '[n, n],
    shape' ~ ConditionalEigenVectors eigenvectors n,
    KnownNat n,
    KnownEigenVectors eigenvectors,
    KnownDType dtype,
    KnownDevice device,
    EigDTypeIsValid device dtype,
    RandDTypeIsValid device dtype,
    KnownDType (ToComplexNumber dtype),
    TensorOptions shape (ToComplexNumber dtype) device,
    TensorOptions shape' (ToComplexNumber dtype) device
  ) =>
  Apply' EigSpec ((Proxy eigenvectors, (Proxy device, (Proxy dtype, Proxy n))), IO ()) (IO ())
  where
  apply' EigSpec (_, agg) =
    agg >> do
      t <- rand @shape @dtype @device
      let (t', t'') = eig @eigenvectors @n @shape' @dtype @device t
      checkDynamicTensorAttributes t'
      checkDynamicTensorAttributes t''

data SVDSpec = SVDSpec

instance
  ( TensorOptions shape dtype device,
    TensorOptions shapeU dtype device,
    TensorOptions shapeS dtype device,
    TensorOptions shapeV dtype device,
    KnownReducedSVD reduced,
    '(shapeU, shapeS, shapeV) ~ SVDShapes shape reduced,
    RandDTypeIsValid device dtype,
    SVDDTypeIsValid device dtype
  ) =>
  Apply' SVDSpec ((Proxy reduced, (Proxy device, (Proxy dtype, Proxy shape))), IO ()) (IO ())
  where
  apply' SVDSpec (_, agg) =
    agg >> do
      a <- randn @shape @dtype @device
      let (u, s, v) = svd @reduced a
      checkDynamicTensorAttributes u
      checkDynamicTensorAttributes s
      checkDynamicTensorAttributes v

data CholeskySpec = CholeskySpec

instance
  ( TensorOptions shape dtype device,
    TensorOptions shape' dtype device,
    TensorOptions shape'' dtype device,
    shape' ~ Square shape,
    shape'' ~ Square (MatMul shape (Transpose shape (LastDim shape) (LastDim shape - 1))),
    1 <= LastDim shape,
    KnownNat (LastDim shape),
    KnownNat (LastDim shape - 1),
    MatMulDTypeIsValid device dtype,
    CholeskyDTypeIsValid device dtype,
    RandDTypeIsValid device dtype
  ) =>
  Apply' CholeskySpec ((Proxy device, (Proxy dtype, Proxy shape)), IO ()) (IO ())
  where
  apply' CholeskySpec (_, agg) =
    agg >> do
      t <- rand @shape @dtype @device
      let t' = t `matmul` transpose @(Backwards shape 0) @(Backwards shape 1) t
      foldMap
        ( \tri -> do
            let t'' = cholesky tri t'
            checkDynamicTensorAttributes t''
        )
        [Upper, Lower]

data CholeskyInverseSpec = CholeskyInverseSpec

instance
  ( TensorOptions shape dtype device,
    shape ~ '[n, n],
    1 <= n,
    RandDTypeIsValid device dtype,
    MatMulDTypeIsValid device dtype,
    CholeskyDTypeIsValid device dtype
  ) =>
  Apply' CholeskyInverseSpec ((Proxy device, (Proxy dtype, Proxy shape)), IO ()) (IO ())
  where
  apply' CholeskyInverseSpec (_, agg) =
    agg >> do
      t <- rand @shape @dtype @device
      let t' = t `matmul` transpose @0 @1 t
      foldMap
        ( \tri -> do
            let t'' = cholesky tri t'
            let t''' = choleskyInverse tri t''
            checkDynamicTensorAttributes t'''
        )
        [Upper, Lower]

data CholeskySolveSpec = CholeskySolveSpec

instance
  ( TensorOptions m_k dtype device,
    TensorOptions m_m dtype device,
    Square m_m ~ m_m,
    MatMul m_m (Transpose m_m (LastDim m_m) (LastDim m_m - 1)) ~ m_m,
    FstSquareDim m_m ~ FstSquareDim m_k,
    1 <= FstSquareDim m_m,
    1 <= LastDim m_m,
    KnownNat (LastDim m_m),
    KnownNat (LastDim m_m - 1),
    MatMulDTypeIsValid device dtype,
    CholeskyDTypeIsValid device dtype,
    RandDTypeIsValid device dtype
  ) =>
  Apply' CholeskySolveSpec ((Proxy device, (Proxy dtype, (Proxy m_k, Proxy m_m))), IO ()) (IO ())
  where
  apply' CholeskySolveSpec (_, agg) =
    agg >> do
      t <- rand @m_m @dtype @device
      let a = t `matmul` transpose @(Backwards m_m 0) @(Backwards m_m 1) t
      b <- rand @m_k
      foldMap
        ( \tri -> do
            let u = cholesky tri a
            checkDynamicTensorAttributes u
            let c = choleskySolve tri b u
            checkDynamicTensorAttributes c
        )
        [Upper, Lower]

data SolveSpec = SolveSpec

instance
  ( TensorOptions m_k dtype device,
    TensorOptions m_m dtype device,
    Square m_m ~ m_m,
    FstSquareDim m_m ~ FstSquareDim m_k,
    1 <= FstSquareDim m_m,
    SolveDTypeIsValid device dtype,
    RandDTypeIsValid device dtype
  ) =>
  Apply' SolveSpec ((Proxy device, (Proxy dtype, (Proxy m_k, Proxy m_m))), IO ()) (IO ())
  where
  apply' SolveSpec (_, agg) =
    agg >> do
      b <- rand @m_k @dtype @device
      a <- rand @m_m
      let c = solve b a
      checkDynamicTensorAttributes c

data TransposeSpec = TransposeSpec

instance
  ( TensorOptions shape dtype device,
    TensorOptions shape' dtype device,
    shape' ~ Transpose shape n m,
    KnownNat n,
    KnownNat m
  ) =>
  Apply' TransposeSpec (((Proxy n, Proxy m), (Proxy device, (Proxy dtype, Proxy shape))), IO ()) (IO ())
  where
  apply' TransposeSpec (_, agg) =
    agg >> do
      let t = ones @shape @dtype @device
          t' = transpose @n @m t
      checkDynamicTensorAttributes t'

data Transpose2DSpec = Transpose2DSpec

instance
  ( TensorOptions '[i, j] dtype device,
    TensorOptions '[j, i] dtype device
  ) =>
  Apply' Transpose2DSpec ((Proxy device, (Proxy dtype, Proxy '[i, j])), IO ()) (IO ())
  where
  apply' Transpose2DSpec (_, agg) =
    agg >> do
      let t = ones @'[i, j] @dtype @device
          t' = transpose2D t
      checkDynamicTensorAttributes t'

data DiagSpec = DiagSpec

instance
  ( TensorOptions shape dtype device,
    TensorOptions shape' dtype device,
    KnownTri tri,
    KnownNat index,
    StandardDTypeValidation device dtype,
    shape' ~ DiagShape tri index shape
  ) =>
  Apply' DiagSpec (((Proxy tri, Proxy index), (Proxy device, (Proxy dtype, Proxy shape))), IO ()) (IO ())
  where
  apply' DiagSpec (_, agg) =
    agg >> do
      let t = ones @shape @dtype @device
      checkDynamicTensorAttributes $ diag @tri @index t

data DiagEmbedSpec = DiagEmbedSpec

instance
  ( TensorOptions shape dtype device,
    TensorOptions shape' dtype device,
    KnownNat index,
    KnownNat dim1,
    KnownNat dim2,
    DimsDistinctAscending dim1 dim2,
    shape' ~ DiagEmbedShape index dim1 dim2 shape,
    StandardDTypeValidation device dtype
  ) =>
  Apply' DiagEmbedSpec (((Proxy index, (Proxy dim1, Proxy dim2)), (Proxy device, (Proxy dtype, Proxy shape))), IO ()) (IO ())
  where
  apply' DiagEmbedSpec (_, agg) =
    agg >> do
      let t = ones @shape @dtype @device
      foldMap
        (\tri -> checkDynamicTensorAttributes $ diagEmbed @index @dim1 @dim2 tri t)
        [Upper, Lower]

data DiagflatSpec = DiagflatSpec

instance
  ( TensorOptions shape dtype device,
    TensorOptions shape' dtype device,
    KnownNat index,
    shape' ~ DiagflatShape index shape,
    StandardDTypeValidation device dtype
  ) =>
  Apply' DiagflatSpec ((Proxy index, (Proxy device, (Proxy dtype, Proxy shape))), IO ()) (IO ())
  where
  apply' DiagflatSpec (_, agg) =
    agg >> do
      let t = ones @shape @dtype @device
      foldMap
        (\tri -> checkDynamicTensorAttributes $ diagflat @index tri t)
        [Upper, Lower]

data DiagonalSpec = DiagonalSpec

instance
  ( TensorOptions shape dtype device,
    TensorOptions shape' dtype device,
    KnownTri tri,
    KnownNat index,
    KnownNat dim1,
    KnownNat dim2,
    NDimAtLeast 2 shape,
    DimsDistinctAscending dim1 dim2,
    shape' ~ DiagonalShape tri index dim1 dim2 shape,
    StandardDTypeValidation device dtype
  ) =>
  Apply' DiagonalSpec (((Proxy tri, (Proxy index, (Proxy dim1, Proxy dim2))), (Proxy device, (Proxy dtype, Proxy shape))), IO ()) (IO ())
  where
  apply' DiagonalSpec (_, agg) =
    agg >> do
      let t = ones @shape @dtype @device
      checkDynamicTensorAttributes $ diagonal @tri @index @dim1 @dim2 t

data AnyAllSpec = AnySpec | AllSpec

instance
  ( TensorOptions shape 'Bool device,
    KnownDevice device
  ) =>
  Apply' AnyAllSpec ((Proxy device, Proxy shape), IO ()) (IO ())
  where
  apply' AnySpec (_, agg) =
    agg >> do
      let t = ones @shape @'Bool @device
          t' = any t
      checkDynamicTensorAttributes t'
  apply' AllSpec (_, agg) =
    agg >> do
      let t = ones @shape @'Bool @device
          t' = all t
      checkDynamicTensorAttributes t'

data AnyPrimeAllPrimeSpec = AnyPrimeSpec | AllPrimeSpec

instance
  ( TensorOptions shape 'Bool device,
    TensorOptions shape' 'Bool device,
    KnownNat dim,
    KnownKeepOrDropDim keepOrDropDim,
    shape' ~ ConditionalDropDimension shape dim keepOrDropDim
  ) =>
  Apply' AnyPrimeAllPrimeSpec (((Proxy dim, Proxy keepOrDropDim), (Proxy device, Proxy shape)), IO ()) (IO ())
  where
  apply' AnyPrimeSpec (_, agg) =
    agg >> do
      let t = ones @shape @'Bool @device
          t' = anyDim @dim @keepOrDropDim t
      checkDynamicTensorAttributes t'
  apply' AllPrimeSpec (_, agg) =
    agg >> do
      let t = ones @shape @'Bool @device
          t' = allDim @dim @keepOrDropDim t
      checkDynamicTensorAttributes t'

data LstmCellSpec = LstmCellSpec

instance
  ( TensorOptions '[4 * hiddenSize, inputSize] dtype device,
    TensorOptions '[4 * hiddenSize, hiddenSize] dtype device,
    TensorOptions '[4 * hiddenSize] dtype device,
    TensorOptions '[batchSize, hiddenSize] dtype device,
    TensorOptions '[batchSize, inputSize] dtype device,
    KnownNat inputSize,
    KnownNat hiddenSize,
    KnownNat batchSize
  ) =>
  Apply' LstmCellSpec ((Proxy device, (Proxy dtype, (Proxy hiddenSize, Proxy inputSize, Proxy batchSize))), IO ()) (IO ())
  where
  apply' LstmCellSpec (_, agg) =
    agg >> do
      let wi = ones @'[4 * hiddenSize, inputSize] @dtype @device
          wh = ones @'[4 * hiddenSize, hiddenSize] @dtype @device
          bi = ones @'[4 * hiddenSize] @dtype @device
          bh = ones @'[4 * hiddenSize] @dtype @device
          cc = ones @'[batchSize, hiddenSize] @dtype @device
          hc = ones @'[batchSize, hiddenSize] @dtype @device
          input = ones @'[batchSize, inputSize] @dtype @device
          (ncc, nhc) = lstmCell wi wh bi bh (cc, hc) input
      checkDynamicTensorAttributes ncc
      checkDynamicTensorAttributes nhc

data GruCellSpec = GruCellSpec

instance
  ( TensorOptions '[3 * hiddenSize, inputSize] dtype device,
    TensorOptions '[3 * hiddenSize, hiddenSize] dtype device,
    TensorOptions '[3 * hiddenSize] dtype device,
    TensorOptions '[batchSize, hiddenSize] dtype device,
    TensorOptions '[batchSize, inputSize] dtype device,
    KnownNat inputSize,
    KnownNat hiddenSize,
    KnownNat batchSize
  ) =>
  Apply' GruCellSpec ((Proxy device, (Proxy dtype, (Proxy hiddenSize, Proxy inputSize, Proxy batchSize))), IO ()) (IO ())
  where
  apply' GruCellSpec (_, agg) =
    agg >> do
      let wi = ones @'[3 * hiddenSize, inputSize] @dtype @device
          wh = ones @'[3 * hiddenSize, hiddenSize] @dtype @device
          bi = ones @'[3 * hiddenSize] @dtype @device
          bh = ones @'[3 * hiddenSize] @dtype @device
          hx = ones @'[batchSize, hiddenSize] @dtype @device
          input = ones @'[batchSize, inputSize] @dtype @device
          nhx = gruCell wi wh bi bh hx input
      checkDynamicTensorAttributes nhx

spec :: Spec
spec = before_ printSeed $ do
  foldMap spec' availableDevices
  where
    printSeed = do
      putStr "      seed:"
      get_manual_seed >>= print

spec' :: Device -> Spec
spec' device =
  describe ("for " <> show device) $ do
    let standardShapes = Proxy @'[2, 3] :. HNil -- (Proxy :: Proxy ('[] :: [Nat])) :. Proxy @'[0]  :. Proxy @'[0, 1] :. Proxy @'[1, 0] :. Proxy @'[2, 3] :. HNil
        squareShapes = Proxy @'[0, 0] :. Proxy @'[1, 1] :. Proxy @'[2, 2] :. Proxy @'[0, 0, 0] :. Proxy @'[0, 1, 1] :. Proxy @'[1, 0, 0] :. Proxy @'[3, 2, 2] :. HNil
        reductions = Proxy @ReduceNone :. Proxy @ReduceMean :. Proxy @ReduceSum :. HNil

    describe "loss functions" $ do
      let dispatch lossSpec = case device of
            Device {deviceType = CPU, deviceIndex = 0} ->
              hfoldrM @IO lossSpec () (hproduct reductions (hattach cpu (hproduct standardFloatingPointDTypes standardShapes)))
            Device {deviceType = CUDA, deviceIndex = 0} ->
              hfoldrM @IO lossSpec () (hproduct reductions (hattach cuda0 (hproduct allFloatingPointDTypes standardShapes)))
      it "binaryCrossEntropy" $ dispatch BinaryCrossEntropySpec
      it "mseLoss" $ dispatch MSELossSpec

    describe "softmax" $ do
      let softmaxDims = Proxy @0 :. Proxy @1 :. HNil
          softmaxShapes = Proxy @'[1, 0] :. Proxy @'[2, 3] :. HNil
          dispatch softmaxSpec = case device of
            Device {deviceType = CPU, deviceIndex = 0} ->
              hfoldrM @IO softmaxSpec () (hproduct softmaxDims (hattach cpu (hproduct standardFloatingPointDTypes standardShapes)))
            Device {deviceType = CUDA, deviceIndex = 0} ->
              hfoldrM @IO softmaxSpec () (hproduct softmaxDims (hattach cuda0 (hproduct allFloatingPointDTypes standardShapes)))
      it "softmax" $ dispatch SoftmaxSpec
      it "logSoftmax" $ dispatch LogSoftmaxSpec

    describe "linear algrebra" $ do
      it "dot" $ case device of
        Device {deviceType = CPU, deviceIndex = 0} ->
          hfoldrM @IO DotSpec () (hattach cpu (hproduct standardDTypes (Proxy @0 :. Proxy @1 :. Proxy @2 :. HNil)))
        Device {deviceType = CUDA, deviceIndex = 0} ->
          hfoldrM @IO DotSpec () (hattach cuda0 (hproduct allFloatingPointDTypes (Proxy @0 :. Proxy @1 :. Proxy @2 :. HNil)))
      it "inverse" $ case device of
        Device {deviceType = CPU, deviceIndex = 0} ->
          hfoldrM @IO InverseSpec () (hattach cpu (hproduct standardFloatingPointDTypes squareShapes))
        Device {deviceType = CUDA, deviceIndex = 0} ->
          hfoldrM @IO InverseSpec () (hattach cuda0 (hproduct standardFloatingPointDTypes (Proxy @'[1, 1] :. Proxy @'[2, 2] :. Proxy @'[1, 1, 1] :. Proxy @'[2, 2, 2] :. HNil)))
      let dispatchSymeigSpec symeigSpec = case device of
            Device {deviceType = CPU, deviceIndex = 0} ->
              hfoldrM @IO symeigSpec () (hattach cpu (hproduct standardFloatingPointDTypes squareShapes))
            Device {deviceType = CUDA, deviceIndex = 0} ->
              hfoldrM @IO symeigSpec () (hattach cuda0 (hproduct standardFloatingPointDTypes squareShapes))
      it "symeig" $ do
        dispatchSymeigSpec SymeigSpec
      it "symeigvalues" $ do
        dispatchSymeigSpec SymeigvaluesSpec
      it "eig" $ do
        let eigenVectors = Proxy @'EnableEigenVectors :. Proxy @'DisableEigenVectors :. HNil
            ns = Proxy @0 :. Proxy @2 :. Proxy @10 :. HNil
        case device of
          Device {deviceType = CPU, deviceIndex = 0} ->
            hfoldrM @IO EigSpec () (hproduct eigenVectors (hattach cpu (hproduct standardFloatingPointDTypes ns)))
          Device {deviceType = CUDA, deviceIndex = 0} ->
            hfoldrM @IO EigSpec () (hproduct eigenVectors (hattach cuda0 (hproduct standardFloatingPointDTypes ns)))
      it "svd" $ do
        let svdShapes = Proxy @'[1, 1] :. Proxy @'[1, 2] :. Proxy @'[2, 1] :. Proxy @'[1, 1, 1] :. Proxy @'[3, 2, 3] :. Proxy @'[3, 3, 2] :. HNil
            reducedSVD = Proxy @'ThinSVD :. Proxy @'FullSVD :. HNil
        case device of
          Device {deviceType = CPU, deviceIndex = 0} ->
            hfoldrM @IO SVDSpec () (hproduct reducedSVD (hattach cpu (hproduct standardFloatingPointDTypes svdShapes)))
          Device {deviceType = CUDA, deviceIndex = 0} ->
            hfoldrM @IO SVDSpec () (hproduct reducedSVD (hattach cuda0 (hproduct standardFloatingPointDTypes svdShapes)))
      it "cholesky" $ case device of
        Device {deviceType = CPU, deviceIndex = 0} ->
          hfoldrM @IO CholeskySpec () (hattach cpu (hproduct standardFloatingPointDTypes squareShapes))
        Device {deviceType = CUDA, deviceIndex = 0} ->
          hfoldrM @IO CholeskySpec () (hattach cuda0 (hproduct standardFloatingPointDTypes squareShapes))
      it "choleskyInverse" $ do
        let choleskyInverseShapes = Proxy @'[1, 1] :. Proxy @'[2, 2] :. HNil
        case device of
          Device {deviceType = CPU, deviceIndex = 0} ->
            hfoldrM @IO CholeskyInverseSpec () (hattach cpu (hproduct standardFloatingPointDTypes choleskyInverseShapes))
          Device {deviceType = CUDA, deviceIndex = 0} ->
            hfoldrM @IO CholeskyInverseSpec () (hattach cuda0 (hproduct standardFloatingPointDTypes choleskyInverseShapes))
      it "choleskySolve" $ do
        let choleskySolveShapes =
              hzip
                (Proxy @'[1, 0] :. Proxy @'[1, 2] :. Proxy @'[2, 1] :. Proxy @'[3, 1, 2] :. HNil)
                (Proxy @'[1, 1] :. Proxy @'[1, 1] :. Proxy @'[2, 2] :. Proxy @'[3, 1, 1] :. HNil)
        case device of
          Device {deviceType = CPU, deviceIndex = 0} ->
            hfoldrM @IO CholeskySolveSpec () (hattach cpu (hproduct standardFloatingPointDTypes choleskySolveShapes))
          Device {deviceType = CUDA, deviceIndex = 0} ->
            hfoldrM @IO CholeskySolveSpec () (hattach cuda0 (hproduct standardFloatingPointDTypes choleskySolveShapes))
      it "solve" $ do
        let solveShapes =
              hzip
                (Proxy @'[1, 0] :. Proxy @'[1, 2] :. Proxy @'[2, 1] :. Proxy @'[3, 1, 2] :. HNil)
                (Proxy @'[1, 1] :. Proxy @'[1, 1] :. Proxy @'[2, 2] :. Proxy @'[3, 1, 1] :. HNil)
        case device of
          Device {deviceType = CPU, deviceIndex = 0} ->
            hfoldrM @IO SolveSpec () (hattach cpu (hproduct standardFloatingPointDTypes solveShapes))
          Device {deviceType = CUDA, deviceIndex = 0} ->
            hfoldrM @IO SolveSpec () (hattach cuda0 (hproduct standardFloatingPointDTypes solveShapes))

    describe "boolean algebra" $ do
      do
        let dispatch anyAllSpec = case device of
              Device {deviceType = CPU, deviceIndex = 0} ->
                hfoldrM @IO anyAllSpec () (hattach cpu standardShapes)
              Device {deviceType = CUDA, deviceIndex = 0} ->
                hfoldrM @IO anyAllSpec () (hattach cuda0 standardShapes)
        it "all" $ dispatch AllSpec
        it "any" $ dispatch AnySpec
      do
        let anyPrimeAllPrimeDims = Proxy @0 :. Proxy @1 :. HNil
            keepOrDropDims = Proxy @KeepDim :. Proxy @DropDim :. HNil
            anyPrimeAllPrimeShapes = Proxy @'[0, 0] :. Proxy @'[0, 1] :. Proxy @'[1, 0] :. Proxy @'[2, 3] :. Proxy @'[0, 1, 1] :. Proxy @'[1, 0, 1] :. HNil
            dispatch anyPrimeAllPrimeSpec = case device of
              Device {deviceType = CPU, deviceIndex = 0} ->
                hfoldrM @IO
                  anyPrimeAllPrimeSpec
                  ()
                  ( hproduct
                      (hproduct anyPrimeAllPrimeDims keepOrDropDims)
                      (hattach cpu anyPrimeAllPrimeShapes)
                  )
              Device {deviceType = CUDA, deviceIndex = 0} ->
                hfoldrM @IO
                  anyPrimeAllPrimeSpec
                  ()
                  ( hproduct
                      (hproduct anyPrimeAllPrimeDims keepOrDropDims)
                      (hattach cuda0 anyPrimeAllPrimeShapes)
                  )
        it "allDim" $ dispatch AllPrimeSpec
        it "anyDim" $ dispatch AnyPrimeSpec

    describe "pooling" $
      it "maxPool2d" $ do
        let c = maxPool2d @'(1, 1) @'(1, 1) @'(0, 0) (ones :: CPUTensor 'Float '[1, 3, 4, 5])
        checkDynamicTensorAttributes c

    describe "sorting" $
      it "topk" $ do
        let (c, c') = topk @3 @1 True True (ones :: CPUTensor 'Float '[2, 3])
        checkDynamicTensorAttributes c
        checkDynamicTensorAttributes c'

    describe "upsampling" $ do
      it "upsample_nearest2d" $ do
        let c = upsample_nearest2d @5 @3 (ones :: CPUTensor 'Float '[2, 3, 2, 2])
        checkDynamicTensorAttributes c
      it "upsample_bicubic2d" $ do
        let c = upsample_bicubic2d @5 @3 False (ones :: CPUTensor 'Float '[2, 3, 2, 2])
        checkDynamicTensorAttributes c
      it "upsample_bilinear2d" $ do
        let c = upsample_bilinear2d @5 @3 False (ones :: CPUTensor 'Float '[2, 3, 2, 2])
        checkDynamicTensorAttributes c

    describe "binary native ops" $ return ()

    describe "RNNCells op" $ do
      it "lstmCell op" $ do
        let sizes =
              hzip3
                (Proxy @2 :. Proxy @4 :. Proxy @6 :. Proxy @7 :. HNil)
                (Proxy @7 :. Proxy @6 :. Proxy @5 :. Proxy @4 :. HNil)
                (Proxy @5 :. Proxy @10 :. Proxy @15 :. Proxy @20 :. HNil)
        case device of
          Device {deviceType = CPU, deviceIndex = 0} ->
            hfoldrM @IO LstmCellSpec () (hattach cpu (hproduct standardFloatingPointDTypes sizes))
          Device {deviceType = CUDA, deviceIndex = 0} ->
            hfoldrM @IO LstmCellSpec () (hattach cuda0 (hproduct standardFloatingPointDTypes sizes))
      it "gruCell op" $ do
        let sizes =
              hzip3
                (Proxy @2 :. Proxy @4 :. Proxy @6 :. Proxy @7 :. HNil)
                (Proxy @7 :. Proxy @6 :. Proxy @5 :. Proxy @4 :. HNil)
                (Proxy @5 :. Proxy @10 :. Proxy @15 :. Proxy @20 :. HNil)
        case device of
          Device {deviceType = CPU, deviceIndex = 0} ->
            hfoldrM @IO GruCellSpec () (hattach cpu (hproduct standardFloatingPointDTypes sizes))
          Device {deviceType = CUDA, deviceIndex = 0} ->
            hfoldrM @IO GruCellSpec () (hattach cuda0 (hproduct standardFloatingPointDTypes sizes))
