{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE EmptyCase #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE StandaloneKindSignatures #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}

module Torch.GraduallyTyped.NN.Transformer.Type where

import Control.Monad.Reader (MonadIO, MonadReader, ask, liftIO)
import qualified Data.Map as Map
import Data.Singletons.TH (genSingletons)
import Foreign.ForeignPtr (ForeignPtr)
import Torch.DType (DType (..))
import Torch.GraduallyTyped.DType (DataType (..), KnownDataType)
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), KnownDevice)
import Torch.GraduallyTyped.Layout (KnownLayout, Layout (..), LayoutType (..))
import Torch.GraduallyTyped.Prelude (Seq)
import Torch.GraduallyTyped.RequiresGradient (RequiresGradient (..))
import Torch.GraduallyTyped.Shape.Class (BroadcastShapesF)
import Torch.GraduallyTyped.Shape.Type (Dim (..), KnownDim (..), KnownShape (..), Name (..), Shape (..), Size (..), WithDimC (..))
import Torch.GraduallyTyped.Tensor.Creation (full)
import Torch.GraduallyTyped.Tensor.MathOperations.Comparison ((==.))
import Torch.GraduallyTyped.Tensor.Type (Tensor (..), checkedDataType, checkedDevice, checkedLayout, checkedShape)
import Torch.GraduallyTyped.Unify (type (<+>))
import qualified Torch.Internal.Type as ATen (Tensor)
import qualified Torch.Script (IValue (..))
import qualified Torch.Serialize (pickleLoad)
import qualified Torch.Tensor (Tensor (Unsafe), asTensor)

data TransformerStyle = T5 | BART | MBART | BERT | Pegasus
  deriving (Show, Eq)

genSingletons [''TransformerStyle]

type TensorDict = Map.Map String (ForeignPtr ATen.Tensor)

tensorDictFromPretrained ::
  FilePath ->
  IO TensorDict
tensorDictFromPretrained filePath = do
  iValue <- Torch.Serialize.pickleLoad filePath
  case iValue of
    Torch.Script.IVGenericDict xs -> Map.fromList <$> go xs
    _ -> fail "iValue is not a state dictionary."
  where
    go [] = pure []
    go ((Torch.Script.IVString s, Torch.Script.IVTensor (Torch.Tensor.Unsafe t)) : xs) = ((s, t) :) <$> go xs
    go ((_, Torch.Script.IVTensor _) : _) = fail "iValue is not a string."
    go ((Torch.Script.IVString _, _) : _) = fail "iValue is not a tensor."
    go _ = fail "iValue is neither a string nor a tensor."

lookupTensor ::
  forall requiresGradient layout device dataType shape m.
  ( MonadReader TensorDict m,
    MonadIO m,
    MonadFail m,
    KnownLayout layout,
    KnownDevice device,
    KnownDataType dataType,
    KnownShape shape
  ) =>
  String ->
  m (Tensor requiresGradient layout device dataType shape)
lookupTensor s = do
  tensorDict <- ask
  liftIO
    ( maybe
        (fail $ "`" <> show s <> "` is not in the state dictionary.")
        (pure . UnsafeTensor)
        (Map.lookup s tensorDict)
    )
    >>= checkedLayout
    >>= checkedDevice
    >>= checkedDataType
    >>= checkedShape

padded :: Integral n => n -> a -> [a] -> [a]
padded n p xs =
  let n' = fromIntegral n
      diff = n' - length xs
   in take n' xs ++ replicate diff p

mkTransformerInput ::
  forall batchDim seqDim m output.
  ( MonadFail m,
    WithDimC batchDim (WithDimF seqDim ([[Int]] -> m output)),
    WithDimC seqDim ([[Int]] -> m output),
    KnownDim batchDim,
    KnownDim seqDim,
    output
      ~ Tensor
          'WithoutGradient
          ('Layout 'Dense)
          ('Device 'CPU)
          ('DataType 'Int64)
          ('Shape '[batchDim, seqDim])
  ) =>
  Int ->
  WithDimF batchDim (WithDimF seqDim ([[Int]] -> m output))
mkTransformerInput padTokenId =
  withDim @batchDim $
    \(Dim batchName batchSize) ->
      withDim @seqDim @([[Int]] -> m output) $
        \(Dim seqName seqSize) xs -> do
          let emptySeq = replicate (fromIntegral seqSize) padTokenId
              paddedXs = padded batchSize emptySeq (padded seqSize padTokenId <$> xs)
          case Torch.Tensor.asTensor paddedXs of
            Torch.Tensor.Unsafe t ->
              pure (UnsafeTensor @'WithoutGradient t)
                >>= checkedLayout @('Layout 'Dense)
                >>= checkedDevice @('Device 'CPU)
                >>= checkedDataType @('DataType 'Int64)
                >>= checkedShape @('Shape '[batchDim, seqDim])

mkTransformerPaddingMask ::
  Int ->
  Tensor requiresGradient layout device dataType shape ->
  Tensor
    'WithoutGradient
    (layout <+> 'Layout 'Dense)
    (device <+> 'Device 'CPU)
    (Seq (dataType <+> 'DataType 'Int64) ('DataType 'Bool))
    (BroadcastShapesF shape ('Shape '[ 'Dim ('Name "*") ('Size 1)]))
mkTransformerPaddingMask padTokenId input =
  let padTokenId' =
        full
          @'WithoutGradient
          @('Layout 'Dense)
          @('Device 'CPU)
          @('DataType 'Int64)
          @('Shape '[ 'Dim ('Name "*") ('Size 1)])
          padTokenId
   in input ==. padTokenId'
