{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneKindSignatures #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE NoStarIsType #-}
{-# OPTIONS_GHC -v2
                -fomit-interface-pragmas
                -fplugin TypeLevel.Rewrite
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyRightAssociativeL
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL2
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL2C
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL3
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL3C
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL4
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL4C
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL5
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL5C
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL6
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL6C
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL7
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL7C
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL8
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL8C #-}

module Torch.GraduallyTyped.NN.Transformer.FeedForwardNetwork where

import Control.Monad.Indexed (ireturn, (>>>=))
import Control.Monad.Indexed.State (IxState (..))
import Control.Monad.State.Strict (MonadState (state), runState)
import Data.Kind (Type)
import GHC.TypeLits (Nat, Symbol)
import Torch.DType (DType (..))
import Torch.GraduallyTyped.DType (DataType, WithDataTypeC (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), WithDeviceC (..))
import Torch.GraduallyTyped.Layout (Layout (Layout), LayoutType (Dense))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..))
import Torch.GraduallyTyped.NN.Dropout (Dropout)
import Torch.GraduallyTyped.NN.Functional.Activation (relu)
import Torch.GraduallyTyped.NN.Functional.Linear (LinearWithoutBiasF)
import Torch.GraduallyTyped.NN.Functional.Normalization (LayerNormWithoutBiasF)
import Torch.GraduallyTyped.NN.Linear (HasInitializeLinearWithoutBiasC, Linear (..))
import Torch.GraduallyTyped.NN.Normalization (HasInitializeLayerNormWithoutBiasC, LayerNorm)
import Torch.GraduallyTyped.NN.Type (HasBias (..))
import Torch.GraduallyTyped.Random (Generator)
import Torch.GraduallyTyped.RequiresGradient (RequiresGradient (..))
import Torch.GraduallyTyped.Scalar (Scalar)
import Torch.GraduallyTyped.Shape.Class (BroadcastShapesF)
import Torch.GraduallyTyped.Shape.Type (Dim (..), KnownDim, KnownShape, Name (..), Shape (..), Size (..), WithDimC (..), WithSelectDimsC, WithShapeC (..))
import Torch.GraduallyTyped.Tensor.MathOperations.Pointwise (add)
import Torch.GraduallyTyped.Tensor.Type (Tensor)
import Torch.GraduallyTyped.Unify (type (<+>))

data
  TransformerFeedForwardNetwork
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (queryEmbedDim :: Dim (Name Symbol) (Size Nat))
    (ffnDim :: Dim (Name Symbol) (Size Nat))
    (dropoutP :: Type)
  where
  TransformerFeedForwardNetwork ::
    forall device dataType queryEmbedDim ffnDim dropoutP.
    { -- | input weight
      ffnInputWeight :: Linear 'WithoutBias device dataType queryEmbedDim ffnDim,
      -- | output weight
      ffnOutputWeight :: Linear 'WithoutBias device dataType ffnDim queryEmbedDim,
      -- | relu dropout
      ffnReluDropout :: Dropout dropoutP,
      -- | feed-forward layer norm
      ffnLayoutNorm :: LayerNorm 'WithoutBias device dataType ( 'Shape '[queryEmbedDim]),
      -- | feed-forward dropout
      ffnDropout :: Dropout dropoutP
    } ->
    TransformerFeedForwardNetwork device dataType queryEmbedDim ffnDim dropoutP

type HasInitializeTransformerFeedForwardNetworkC device dataType queryEmbedDim ffnDim dropoutP =
  ( HasInitializeLinearWithoutBiasC device dataType queryEmbedDim ffnDim,
    HasInitializeLinearWithoutBiasC device dataType ffnDim queryEmbedDim,
    HasInitializeLayerNormWithoutBiasC device dataType ( 'Shape '[queryEmbedDim]),
    Scalar dropoutP,
    WithDeviceC device (WithDataTypeF dataType (WithDimF queryEmbedDim (WithDimF ffnDim (dropoutP -> Double -> Generator device -> (TransformerFeedForwardNetwork device dataType queryEmbedDim ffnDim dropoutP, Generator device))))),
    WithDataTypeC dataType (WithDimF queryEmbedDim (WithDimF ffnDim (dropoutP -> Double -> Generator device -> (TransformerFeedForwardNetwork device dataType queryEmbedDim ffnDim dropoutP, Generator device)))),
    WithDimC queryEmbedDim (WithDimF ffnDim (dropoutP -> Double -> Generator device -> (TransformerFeedForwardNetwork device dataType queryEmbedDim ffnDim dropoutP, Generator device))),
    WithDimC ffnDim (dropoutP -> Double -> Generator device -> (TransformerFeedForwardNetwork device dataType queryEmbedDim ffnDim dropoutP, Generator device))
  )

instance
  HasInitializeTransformerFeedForwardNetworkC device dataType queryEmbedDim ffnDim dropoutP =>
  HasInitialize (TransformerFeedForwardNetwork device dataType queryEmbedDim ffnDim dropoutP)
  where
  type
    InitializeF (TransformerFeedForwardNetwork device dataType queryEmbedDim ffnDim dropoutP) =
      WithDeviceF
        device
        ( WithDataTypeF
            dataType
            ( WithDimF
                queryEmbedDim
                ( WithDimF
                    ffnDim
                    ( dropoutP ->
                      Double ->
                      Generator device ->
                      ( TransformerFeedForwardNetwork
                          device
                          dataType
                          queryEmbedDim
                          ffnDim
                          dropoutP,
                        Generator device
                      )
                    )
                )
            )
        )
  initialize =
    withDevice @device $
      \deviceType ->
        withDataType @dataType $
          \dType ->
            withDim @queryEmbedDim $
              \queryEmbedDim ->
                withDim @ffnDim @(dropoutP -> Double -> Generator device -> (TransformerFeedForwardNetwork device dataType queryEmbedDim ffnDim dropoutP, Generator device)) $
                  \ffnDim ->
                    go deviceType dType queryEmbedDim ffnDim
    where
      go deviceType dType queryEmbedDim ffnDim dropoutP eps = runState $ do
        inputWeight <-
          state $
            withoutDim @ffnDim
              ( withoutDim @queryEmbedDim
                  ( withoutDataType @dataType
                      ( withoutDevice @device
                          ( initialize @(Linear 'WithoutBias device dataType queryEmbedDim ffnDim)
                          )
                          deviceType
                      )
                      dType
                  )
                  queryEmbedDim
              )
              ffnDim
        outputWeight <-
          state $
            withoutDim @queryEmbedDim
              ( withoutDim @ffnDim
                  ( withoutDataType @dataType
                      ( withoutDevice @device
                          ( initialize @(Linear 'WithoutBias device dataType ffnDim queryEmbedDim)
                          )
                          deviceType
                      )
                      dType
                  )
                  ffnDim
              )
              queryEmbedDim
        let reluDropout = initialize @(Dropout dropoutP) dropoutP
        let layerNorm =
              withoutShape @( 'Shape '[queryEmbedDim])
                ( withoutDataType @dataType
                    ( withoutDevice @device
                        ( initialize @(LayerNorm 'WithoutBias device dataType ( 'Shape '[queryEmbedDim]))
                        )
                        deviceType
                    )
                    dType
                )
                [queryEmbedDim]
                eps
        let dropout = initialize @(Dropout dropoutP) dropoutP
        pure $ TransformerFeedForwardNetwork inputWeight outputWeight reluDropout layerNorm dropout

type FeedForwardNetworkOutputShape
  (queryEmbedDim :: Dim (Name Symbol) (Size Nat))
  (ffnDim :: Dim (Name Symbol) (Size Nat))
  (queryShape :: Shape [Dim (Name Symbol) (Size Nat)]) =
  BroadcastShapesF
    queryShape
    ( LinearWithoutBiasF
        ( 'Shape '[queryEmbedDim, ffnDim])
        ( LinearWithoutBiasF
            ( 'Shape '[ffnDim, queryEmbedDim])
            ( LayerNormWithoutBiasF
                ( 'Shape '[queryEmbedDim])
                queryShape
            )
        )
    )

instance
  ( KnownShape queryShape,
    KnownDim queryEmbedDim,
    Scalar dropoutP,
    output ~ Tensor
        'WithGradient
        (queryLayout <+> 'Layout 'Dense)
        (queryDevice <+> device <+> generatorDevice)
        (queryDataType <+> dataType)
        (FeedForwardNetworkOutputShape queryEmbedDim ffnDim queryShape),
    generatorOutput ~ Generator (device <+> queryDevice <+> generatorDevice)
  ) =>
  HasForward
    (TransformerFeedForwardNetwork device dataType queryEmbedDim ffnDim dropoutP)
    (Tensor queryRequiresGradient queryLayout queryDevice queryDataType queryShape)
    (Generator generatorDevice)
    output
    generatorOutput
  where
  forward TransformerFeedForwardNetwork {..} query =
    runIxState $
      ireturn query
        >>>= IxState . forward ffnLayoutNorm
        >>>= IxState . forward ffnInputWeight
        >>>= IxState . forward ffnReluDropout . relu
        >>>= IxState . forward ffnOutputWeight
        >>>= IxState . forward ffnDropout
        >>>= ireturn . (query `add`)