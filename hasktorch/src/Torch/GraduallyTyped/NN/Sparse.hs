{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}

module Torch.GraduallyTyped.NN.Sparse where

import Control.Monad.State.Strict (MonadState (state), runState)
import Data.Data (Proxy (..))
import GHC.TypeLits (KnownNat, Nat, Symbol, natVal)
import Torch.DType (DType (..))
import Torch.GraduallyTyped.DType (DataType (..), WithDataTypeC (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType, WithDeviceC (..))
import Torch.GraduallyTyped.Layout (KnownLayout, Layout (..), LayoutType (..), WithLayoutC (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..))
import Torch.GraduallyTyped.NN.Functional.Sparse (EmbeddingF, embedding)
import Torch.GraduallyTyped.Random (Generator)
import Torch.GraduallyTyped.RequiresGradient (RequiresGradient (..))
import Torch.GraduallyTyped.Shape (Dim (..), Name, Shape (..), Size, WithDimC (..))
import Torch.GraduallyTyped.Tensor.Creation (WithCreateC (..), randn)
import Torch.GraduallyTyped.Tensor.Type (Tensor)
import Torch.GraduallyTyped.Unify (type (<+>))
import Torch.GraduallyTyped.Prelude (Seq)

data
  Embedding
    (layout :: Layout LayoutType)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (embedNumDim :: Dim (Name Symbol) (Size Nat))
    (embedDim :: Dim (Name Symbol) (Size Nat))
    (paddingIdx :: Maybe Nat)
  where
  Embedding ::
    forall layout device dataType embedNumDim embedDim paddingIdx.
    { embeddingWeight :: Tensor 'WithGradient layout device dataType ( 'Shape '[embedNumDim, embedDim])
    } ->
    Embedding layout device dataType embedNumDim embedDim paddingIdx

type HasInitializeEmbeddingC layout device dataType embedNumDim embedDim paddingIdx =
  ( WithLayoutC layout (WithDeviceF device (WithDataTypeF dataType (WithDimF embedNumDim (WithDimF embedDim (Generator device -> (Embedding layout device dataType embedNumDim embedDim paddingIdx, Generator device)))))),
    WithDeviceC device (WithDataTypeF dataType (WithDimF embedNumDim (WithDimF embedDim (Generator device -> (Embedding layout device dataType embedNumDim embedDim paddingIdx, Generator device))))),
    WithDataTypeC dataType (WithDimF embedNumDim (WithDimF embedDim (Generator device -> (Embedding layout device dataType embedNumDim embedDim paddingIdx, Generator device)))),
    WithDimC embedNumDim (WithDimF embedDim (Generator device -> (Embedding layout device dataType embedNumDim embedDim paddingIdx, Generator device))),
    WithDimC embedDim (Generator device -> (Embedding layout device dataType embedNumDim embedDim paddingIdx, Generator device)),
    WithCreateC (Generator device -> (Tensor 'WithGradient layout device dataType ( 'Shape '[embedNumDim, embedDim]), Generator device)) 'WithGradient layout device dataType ( 'Shape '[embedNumDim, embedDim])
  )

instance
  HasInitializeEmbeddingC layout device dataType embedNumDim embedDim 'Nothing =>
  HasInitialize (Embedding layout device dataType embedNumDim embedDim 'Nothing)
  where
  type
    InitializeF (Embedding layout device dataType embedNumDim embedDim 'Nothing) =
      WithLayoutF
        layout
        ( WithDeviceF
            device
            ( WithDataTypeF
                dataType
                ( WithDimF
                    embedNumDim
                    ( WithDimF
                        embedDim
                        (Generator device -> (Embedding layout device dataType embedNumDim embedDim 'Nothing, Generator device))
                    )
                )
            )
        )
  initialize =
    withLayout @layout $
      \layoutType ->
        withDevice @device $
          \deviceType ->
            withDataType @dataType $
              \dType ->
                withDim @embedNumDim $
                  \embedNumDim ->
                    withDim @embedDim @(Generator device -> (Embedding layout device dataType embedNumDim embedDim 'Nothing, Generator device)) $
                      \embedDim ->
                        go layoutType deviceType dType embedNumDim embedDim
    where
      go layoutType deviceType dType embedNumDim embedDim = runState $ do
        weight <-
          state $
            withoutCreate @_ @ 'WithGradient @layout @device @dataType @( 'Shape '[embedNumDim, embedDim])
              (randn @ 'WithGradient @layout @device @dataType @( 'Shape '[embedNumDim, embedDim]) @device)
              WithGradient
              layoutType
              deviceType
              dType
              [embedNumDim, embedDim]
        pure $ Embedding weight

instance
  ( KnownLayout layout,
    output
      ~ Tensor
          'WithGradient
          (layout <+> layout')
          (device <+> device')
          (Seq (dataType' <+> 'DataType 'Int64) dataType)
          (EmbeddingF ( 'Shape '[embedNumDim, embedDim]) shape')
  ) =>
  HasForward
    (Embedding layout device dataType embedNumDim embedDim 'Nothing)
    (Tensor requiresGradient' layout' device' dataType' shape')
    generator
    output
    generator
  where
  forward (Embedding weight) input g = (embedding Nothing False weight input, g)

instance
  ( KnownLayout layout,
    KnownNat paddingIdx,
    output
      ~ Tensor
          'WithGradient
          (layout <+> layout')
          (device <+> device')
          (Seq (dataType' <+> 'DataType 'Int64) dataType)
          (EmbeddingF ( 'Shape '[embedNumDim, embedDim]) shape')
  ) =>
  HasForward
    (Embedding layout device dataType embedNumDim embedDim ( 'Just paddingIdx))
    (Tensor requiresGradient' layout' device' dataType' shape')
    generator
    output
    generator
  where
  forward Embedding {..} input g = (embedding (Just . fromIntegral . natVal $ Proxy @paddingIdx) False embeddingWeight input, g)
