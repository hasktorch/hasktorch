{-# LANGUAGE StandaloneKindSignatures #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE NoStarIsType #-}
{-# LANGUAGE AllowAmbiguousTypes #-}

module Torch.GraduallyTyped.NN.Transformer where

import Control.Monad.State.Strict (MonadState (state), runState)
import Data.Kind (Type)
import GHC.TypeLits (type (<=), type (*), KnownNat, Div, Nat, Symbol)
import Torch.DType (DType (..))
import Torch.GraduallyTyped.DType (UnifyDataTypeF, DataType (DataType), WithDataTypeC (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType(..), WithDeviceC (..))
import Torch.GraduallyTyped.NN.Class (forward, HasInitialize (..))
import Torch.GraduallyTyped.NN.Dropout (Dropout (Dropout))
import Torch.GraduallyTyped.NN.Linear (HasInitializeLinearC, Linear (Linear))
import Torch.GraduallyTyped.Random (generator, Generator)
import Torch.GraduallyTyped.Scalar (Scalar)
import Torch.GraduallyTyped.Shape (dimSize, Size(..), Name(..), KnownDim(..), WithShapeC(..), NumelF, By(..), SelectDim(..), Dim (..), Shape (..), WithDimC (..))
import Torch.GraduallyTyped.Tensor.MathOperations.Pointwise (divScalar, add)
import Torch.GraduallyTyped.Tensor.Type (Tensor)
import Torch.GraduallyTyped.Tensor.IndexingSlicingJoining (ReshapeF, TransposeF, reshape, transpose)
import Torch.GraduallyTyped.Tensor.MathOperations.BlasLapack (matmul)
import Torch.GraduallyTyped.NN.Functional.Linear (LinearF)
import Torch.GraduallyTyped.NN.Functional.NonLinearActivation (softmax)
import Torch.GraduallyTyped.Layout
import Torch.GraduallyTyped.RequiresGradient (RequiresGradient(Dependent))
import Torch.GraduallyTyped.Tensor.Creation (randn)

-- residual f g x = f x >>= (\x' -> g (x `add` x'))

--------------------------------------------------------------------------------
-- Multi-Headed Attention Layer
--------------------------------------------------------------------------------

data
  MultiheadAttention
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (queryEmbedDim :: Dim (Name Symbol) (Size Nat))
    (keyEmbedDim :: Dim (Name Symbol) (Size Nat))
    (valueEmbedDim :: Dim (Name Symbol) (Size Nat))
    (dropoutP :: Type)
  where
  MultiheadAttention ::
    { -- | in-projection for query
      mhaQInProj :: Linear device dataType queryEmbedDim queryEmbedDim,
      -- | in-projection for key
      mhaKInProj :: Linear device dataType keyEmbedDim queryEmbedDim,
      -- | in-projection for value
      mhaVInProj :: Linear device dataType valueEmbedDim queryEmbedDim,
      -- | out-projection
      mhaOutProj :: Linear device dataType queryEmbedDim queryEmbedDim,
      -- | dropout
      mhaDropout :: Dropout dropoutP
    } ->
    MultiheadAttention device dataType queryEmbedDim keyEmbedDim valueEmbedDim dropoutP

type HasInitializeMultiheadAttentionC device dataType queryEmbedDim keyEmbedDim valueEmbedDim dropoutP =
  ( WithDeviceC device (WithDataTypeF dataType (WithDimF queryEmbedDim (WithDimF keyEmbedDim (WithDimF valueEmbedDim (dropoutP -> Generator device -> (MultiheadAttention device dataType queryEmbedDim keyEmbedDim valueEmbedDim dropoutP, Generator device)))))),
    WithDataTypeC dataType (WithDimF queryEmbedDim (WithDimF keyEmbedDim (WithDimF valueEmbedDim (dropoutP -> Generator device -> (MultiheadAttention device dataType queryEmbedDim keyEmbedDim valueEmbedDim dropoutP, Generator device))))),
    WithDimC queryEmbedDim (WithDimF keyEmbedDim (WithDimF valueEmbedDim (dropoutP -> Generator device -> (MultiheadAttention device dataType queryEmbedDim keyEmbedDim valueEmbedDim dropoutP, Generator device)))),
    WithDimC keyEmbedDim (WithDimF valueEmbedDim (dropoutP -> Generator device -> (MultiheadAttention device dataType queryEmbedDim keyEmbedDim valueEmbedDim dropoutP, Generator device))),
    WithDimC valueEmbedDim (dropoutP -> Generator device -> (MultiheadAttention device dataType queryEmbedDim keyEmbedDim valueEmbedDim dropoutP, Generator device)),
    WithDimC queryEmbedDim (Generator device -> (Linear device dataType queryEmbedDim queryEmbedDim, Generator device)),
    HasInitializeLinearC device dataType queryEmbedDim queryEmbedDim,
    HasInitializeLinearC device dataType keyEmbedDim queryEmbedDim,
    HasInitializeLinearC device dataType valueEmbedDim queryEmbedDim,
    Scalar dropoutP
  )

instance
  HasInitializeMultiheadAttentionC device dataType queryEmbedDim keyEmbedDim valueEmbedDim dropoutP =>
  HasInitialize (MultiheadAttention device dataType queryEmbedDim keyEmbedDim valueEmbedDim dropoutP)
  where
  type
    InitializeF (MultiheadAttention device dataType queryEmbedDim keyEmbedDim valueEmbedDim dropoutP) =
      WithDeviceF
        device
        ( WithDataTypeF
            dataType
            ( WithDimF
                queryEmbedDim
                ( WithDimF
                    keyEmbedDim
                    ( WithDimF
                        valueEmbedDim
                        (dropoutP -> Generator device -> (MultiheadAttention device dataType queryEmbedDim keyEmbedDim valueEmbedDim dropoutP, Generator device))
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
                withDim @keyEmbedDim $
                  \keyEmbedDim ->
                    withDim @valueEmbedDim @(dropoutP -> Generator device -> (MultiheadAttention device dataType queryEmbedDim keyEmbedDim valueEmbedDim dropoutP, Generator device)) $
                      \valueEmbedDim ->
                        go deviceType dType queryEmbedDim keyEmbedDim valueEmbedDim
    where
      go deviceType dType queryEmbedDim keyEmbedDim valueEmbedDim dropoutP = runState $ do
        qInProj <-
          state $
            withoutDim @queryEmbedDim
              ( withoutDim @queryEmbedDim
                  ( withoutDataType @dataType
                      ( withoutDevice @device
                          ( initialize @(Linear device dataType queryEmbedDim queryEmbedDim)
                          )
                          deviceType
                      )
                      dType
                  )
                  queryEmbedDim
              )
              queryEmbedDim
        kInProj <-
          state $
            withoutDim @queryEmbedDim
              ( withoutDim @keyEmbedDim
                  ( withoutDataType @dataType
                      ( withoutDevice @device
                          ( initialize @(Linear device dataType keyEmbedDim queryEmbedDim)
                          )
                          deviceType
                      )
                      dType
                  )
                  keyEmbedDim
              )
              queryEmbedDim
        vInProj <-
          state $
            withoutDim @queryEmbedDim
              ( withoutDim @valueEmbedDim
                  ( withoutDataType @dataType
                      ( withoutDevice @device
                          ( initialize @(Linear device dataType valueEmbedDim queryEmbedDim)
                          )
                          deviceType
                      )
                      dType
                  )
                  valueEmbedDim
              )
              queryEmbedDim
        outProj <-
          state $
            withoutDim @queryEmbedDim
              ( withoutDim @queryEmbedDim
                  ( withoutDataType @dataType
                      ( withoutDevice @device
                          ( initialize @(Linear device dataType queryEmbedDim queryEmbedDim)
                          )
                          deviceType
                      )
                      dType
                  )
                  queryEmbedDim
              )
              queryEmbedDim
        dropout <-
          pure $ initialize @(Dropout dropoutP) dropoutP
        pure $ MultiheadAttention qInProj kInProj vInProj outProj dropout

reshape' ::
  forall (batchDim :: Dim (Name Symbol) (Size Nat)) (seqDim :: Dim (Name Symbol) (Size Nat)) (headDim :: Dim (Name Symbol) (Size Nat)) (headEmbedDim :: Dim (Name Symbol) (Size Nat)) requiresGradient layout device dataType shape.
  ( WithShapeC
      ( 'Shape '[batchDim, seqDim, headDim, headEmbedDim])
      ( Tensor requiresGradient layout device dataType shape ->
        Tensor
          requiresGradient
          layout
          device
          dataType
          ( ReshapeF
              (NumelF shape)
              (NumelF ( 'Shape '[batchDim, seqDim, headDim, headEmbedDim]))
              shape
              ( 'Shape '[batchDim, seqDim, headDim, headEmbedDim])
          )
      ),
    ( WithDimC
        batchDim
        ( WithDimF
            seqDim
            ( WithDimF
                headDim
                ( WithDimF
                    headEmbedDim
                    ( Tensor requiresGradient layout device dataType shape ->
                      Tensor
                        requiresGradient
                        layout
                        device
                        dataType
                        ( ReshapeF
                            (NumelF shape)
                            ( NumelF
                                ( 'Shape
                                    '[batchDim, seqDim, headDim, headEmbedDim]
                                )
                            )
                            shape
                            ( 'Shape
                                '[batchDim, seqDim, headDim, headEmbedDim]
                            )
                        )
                    )
                )
            )
        )
    ),
    ( WithDimC
        seqDim
        ( WithDimF
            headDim
            ( WithDimF
                headEmbedDim
                ( Tensor requiresGradient layout device dataType shape ->
                  Tensor
                    requiresGradient
                    layout
                    device
                    dataType
                    ( ReshapeF
                        (NumelF shape)
                        (NumelF ( 'Shape '[batchDim, seqDim, headDim, headEmbedDim]))
                        shape
                        ( 'Shape '[batchDim, seqDim, headDim, headEmbedDim])
                    )
                )
            )
        )
    ),
    ( WithDimC
        headDim
        ( WithDimF
            headEmbedDim
            ( Tensor requiresGradient layout device dataType shape ->
              Tensor
                requiresGradient
                layout
                device
                dataType
                ( ReshapeF
                    (NumelF shape)
                    (NumelF ( 'Shape '[batchDim, seqDim, headDim, headEmbedDim]))
                    shape
                    ( 'Shape '[batchDim, seqDim, headDim, headEmbedDim])
                )
            )
        )
    ),
    ( WithDimC
        headEmbedDim
        ( Tensor requiresGradient layout device dataType shape ->
          Tensor
            requiresGradient
            layout
            device
            dataType
            ( ReshapeF
                (NumelF shape)
                (NumelF ( 'Shape '[batchDim, seqDim, headDim, headEmbedDim]))
                shape
                ( 'Shape '[batchDim, seqDim, headDim, headEmbedDim])
            )
        )
    )
  ) =>
  WithDimF
    batchDim
    ( WithDimF
        seqDim
        ( WithDimF
            headDim
            ( WithDimF
                headEmbedDim
                ( Tensor requiresGradient layout device dataType shape ->
                  Tensor
                    requiresGradient
                    layout
                    device
                    dataType
                    ( ReshapeF
                        (NumelF shape)
                        (NumelF ( 'Shape '[batchDim, seqDim, headDim, headEmbedDim]))
                        shape
                        ( 'Shape '[batchDim, seqDim, headDim, headEmbedDim])
                    )
                )
            )
        )
    )
reshape' =
  withDim @batchDim $
    \batchDim ->
      withDim @seqDim $
        \seqDim ->
          withDim @headDim $
            \headDim ->
              withDim @headEmbedDim
                @( Tensor requiresGradient layout device dataType shape ->
                   Tensor
                     requiresGradient
                     layout
                     device
                     dataType
                     ( ReshapeF
                         (NumelF shape)
                         (NumelF ( 'Shape '[batchDim, seqDim, headDim, headEmbedDim]))
                         shape
                         ( 'Shape '[batchDim, seqDim, headDim, headEmbedDim])
                     )
                 )
                $ \headEmbedDim ->
                  go batchDim seqDim headDim headEmbedDim
  where
    go batchDim seqDim headDim headEmbedDim input =
      --transpose @( 'SelectDim ( 'ByIndex 1)) @( 'SelectDim ( 'ByIndex 2))
      ( withoutShape
          @( 'Shape '[batchDim, seqDim, headDim, headEmbedDim])
          @( Tensor requiresGradient layout device dataType shape ->
             Tensor
               requiresGradient
               layout
               device
               dataType
               ( ReshapeF
                   (NumelF shape)
                   (NumelF ( 'Shape '[batchDim, seqDim, headDim, headEmbedDim]))
                   shape
                   ( 'Shape '[batchDim, seqDim, headDim, headEmbedDim])
               )
           )
          (reshape @( 'Shape '[batchDim, seqDim, headDim, headEmbedDim]) @requiresGradient @layout @device @dataType @shape)
          [batchDim, seqDim, headDim, headEmbedDim]
          input
      )

-- batchDim ~ GetDimF ('SelectDim ('ByIndex 0)) queryShape
-- batchDim ~ GetDimF ('SelectDim ('ByIndex 0)) keyShape
-- batchDim ~ GetDimF ('SelectDim ('ByIndex 0)) valueShape
-- querySeqDim ~ GetDimF ('SelectDim ('ByIndex 1)) queryShape
-- keySeqDim ~ GetDimF ('SelectDim ('ByIndex 1)) keyShape
-- keySeqDim ~ GetDimF ('SelectDim ('ByIndex 1)) valueShape
-- queryEmbedDim ~ GetDimF ('SelectDim ('ByIndex 2)) queryShape
-- keyEmbedDim ~ GetDimF ('SelectDim ('ByIndex 2)) keyShape
-- valueEmbedDim ~ GetDimF ('SelectDim ('ByIndex 2)) valueShape
-- headDim
-- headEmbedDim
multiheadAttention ::
  forall (batchDim :: Dim (Name Symbol) (Size Nat)) (querySeqDim :: Dim (Name Symbol) (Size Nat)) (keySeqDim :: Dim (Name Symbol) (Size Nat)) (headDim :: Dim (Name Symbol) (Size Nat)) (headEmbedDim :: Dim (Name Symbol) (Size Nat)) device dataType queryEmbedDim keyEmbedDim valueEmbedDim dropoutP requiresGradient queryLayout queryDevice queryDataType queryShape keyLayout keyDevice keyDataType keyShape valueLayout valueDevice valueDataType valueShape generatorDevice outputLayout outputDevice outputDataType outputShape.
  _ =>
  -- | multi-head attention model
  MultiheadAttention device dataType queryEmbedDim keyEmbedDim valueEmbedDim dropoutP ->
  -- | query representation
  Tensor requiresGradient queryLayout queryDevice queryDataType queryShape ->
  -- | key representation
  Tensor requiresGradient keyLayout keyDevice keyDataType keyShape ->
  -- | value representation
  Tensor requiresGradient valueLayout valueDevice valueDataType valueShape ->
  -- | random generator
  Generator generatorDevice ->
  -- | attention and random generator
  ( Tensor requiresGradient outputLayout outputDevice outputDataType outputShape,
    Generator outputDevice
  )
multiheadAttention MultiheadAttention {..} query key value g =
  let batchDim = case dimVal @batchDim of
        Dim (Name name) (Size size) -> Dim name size
        Dim _ _ -> undefined
      querySeqDim = case dimVal @querySeqDim of
        Dim (Name name) (Size size) -> Dim name size
        Dim _ _ -> undefined
      keySeqDim = case dimVal @keySeqDim of
        Dim (Name name) (Size size) -> Dim name size
        Dim _ _ -> undefined
      headDim = case dimVal @headDim of
        Dim (Name name) (Size size) -> Dim name size
        Dim _ _ -> undefined
      headEmbedDim = case dimVal @headEmbedDim of
        Dim (Name name) (Size size) -> Dim name size
        Dim _ _ -> undefined
      queryEmbedDim = case dimVal @queryEmbedDim of
        Dim (Name name) (Size size) -> Dim name size
        Dim _ _ -> undefined
      scaling :: Double = sqrt . fromIntegral . dimSize $ headDim
      q =
        transpose @( 'SelectDim ( 'ByIndex 1)) @( 'SelectDim ( 'ByIndex 2))
          . reshape' @batchDim @querySeqDim @headDim @headEmbedDim [batchDim, querySeqDim, headDim, headEmbedDim]
          . flip divScalar scaling
          . forward mhaQInProj
          $ query
      k = transpose @( 'SelectDim ( 'ByIndex 1)) @( 'SelectDim ( 'ByIndex 2))
          . reshape' @batchDim @keySeqDim @headDim @headEmbedDim [batchDim, keySeqDim, headDim, headEmbedDim]
          . forward mhaKInProj $ key
      qk = q `matmul` transpose @( 'SelectDim ( 'ByIndex 2)) @( 'SelectDim ( 'ByIndex 3)) k
      (weights, g') = forward @_ @_ @(Generator generatorDevice) mhaDropout (softmax @( 'SelectDim ( 'ByIndex 3)) qk) g
      v = transpose @( 'SelectDim ( 'ByIndex 1)) @( 'SelectDim ( 'ByIndex 2))
          . reshape' @batchDim @keySeqDim @headDim @headEmbedDim  [batchDim, keySeqDim, headDim, headEmbedDim]
          . forward mhaVInProj $ value
   in (forward mhaOutProj . reshape'' @batchDim @querySeqDim @queryEmbedDim [batchDim, querySeqDim, queryEmbedDim] . transpose @( 'SelectDim ( 'ByIndex 1)) @( 'SelectDim ( 'ByIndex 2)) $ weights `matmul` v, g')
  where
    reshape' ::
      forall batchDim seqDim headDim headEmbedDim requiresGradient layout device dataType shape.
      WithShapeC
        ( 'Shape '[batchDim, seqDim, headDim, headEmbedDim])
        ( Tensor requiresGradient layout device dataType shape ->
          Tensor
            requiresGradient
            layout
            device
            dataType
            ( ReshapeF
                (NumelF shape)
                (NumelF ( 'Shape '[batchDim, seqDim, headDim, headEmbedDim]))
                shape
                ( 'Shape '[batchDim, seqDim, headDim, headEmbedDim])
            )
        ) =>
      [Dim String Integer] ->
      Tensor requiresGradient layout device dataType shape ->
      Tensor
        requiresGradient
        layout
        device
        dataType
        ( ReshapeF
            (NumelF shape)
            (NumelF ( 'Shape '[batchDim, seqDim, headDim, headEmbedDim]))
            shape
            ( 'Shape '[batchDim, seqDim, headDim, headEmbedDim])
        )
    reshape' [batchDim, seqDim, headDim, headEmbedDim] input =
      withoutShape
        @( 'Shape '[batchDim, seqDim, headDim, headEmbedDim])
        @( Tensor requiresGradient layout device dataType shape ->
           Tensor
             requiresGradient
             layout
             device
             dataType
             ( ReshapeF
                 (NumelF shape)
                 (NumelF ( 'Shape '[batchDim, seqDim, headDim, headEmbedDim]))
                 shape
                 ( 'Shape '[batchDim, seqDim, headDim, headEmbedDim])
             )
         )
        (reshape @( 'Shape '[batchDim, seqDim, headDim, headEmbedDim]) @requiresGradient @layout @device @dataType @shape)
        [batchDim, seqDim, headDim, headEmbedDim]
        input
    reshape''
      :: forall batchDim seqDim embedDim requiresGradient layout device dataType shape.
      WithShapeC
        ( 'Shape '[batchDim, seqDim, embedDim])
        ( Tensor requiresGradient layout device dataType shape ->
          Tensor
            requiresGradient
            layout
            device
            dataType
            ( ReshapeF
                (NumelF shape)
                (NumelF ( 'Shape '[batchDim, seqDim, embedDim]))
                shape
                ( 'Shape '[batchDim, seqDim, embedDim])
            )
        ) =>
      [Dim String Integer] ->
      Tensor requiresGradient layout device dataType shape ->
      Tensor
        requiresGradient
        layout
        device
        dataType
        ( ReshapeF
            (NumelF shape)
            (NumelF ( 'Shape '[batchDim, seqDim, embedDim]))
            shape
            ( 'Shape '[batchDim, seqDim, embedDim])
        )
    reshape'' [batchDim, seqDim, embedDim] input =
      withoutShape
        @( 'Shape '[batchDim, seqDim, embedDim])
        @( Tensor requiresGradient layout device dataType shape ->
           Tensor
             requiresGradient
             layout
             device
             dataType
             ( ReshapeF
                 (NumelF shape)
                 (NumelF ( 'Shape '[batchDim, seqDim, embedDim]))
                 shape
                 ( 'Shape '[batchDim, seqDim, embedDim])
             )
         )
        (reshape @( 'Shape '[batchDim, seqDim, embedDim]) @requiresGradient @layout @device @dataType @shape)
        [batchDim, seqDim, embedDim]
        input 

type TestDevice :: Device (DeviceType Nat)
type TestDevice = 'Device 'CPU
type TestLayout = 'Layout 'Dense
type TestDataType = 'DataType 'Float
type TestQuerySeqDim = 'Dim ('Name "querySeq") ('Size 64)
type TestQueryEmbedDim = 'Dim ('Name "queryEmbed") ('Size 32)
type TestKeySeqDim = 'Dim ('Name "keySeq") ('Size 48)
type TestKeyEmbedDim = 'Dim ('Name "keyEmbed") ('Size 16)
type TestValueEmbedDim = 'Dim ('Name "valueEmbed") ('Size 24)
type TestHeadDim = 'Dim ('Name "head") ('Size 8)
type TestHeadEmbedDim = 'Dim ('Name "headEmbed") ('Size 4)
type TestBatchDim = 'Dim ('Name "batch") ('Size 4)

testmha ::
  IO
    ( Tensor
        'Dependent
        ( 'Layout 'Dense)
        ( 'Device 'CPU)
        ( 'DataType 'Float)
        ( 'Shape '[TestBatchDim, TestQuerySeqDim, TestQueryEmbedDim])
    )
testmha = do
  g <- generator @TestDevice 0
  let (mha, g') = initialize @(MultiheadAttention TestDevice TestDataType TestQueryEmbedDim TestKeyEmbedDim TestValueEmbedDim Float) 0.0 g
      (query, g'') = randn @ 'Dependent @TestLayout @TestDevice @TestDataType @( 'Shape '[TestBatchDim, TestQuerySeqDim, TestQueryEmbedDim]) g'
      (key, g''') = randn @ 'Dependent @TestLayout @TestDevice @TestDataType @( 'Shape '[TestBatchDim, TestKeySeqDim, TestKeyEmbedDim]) g''
      (value, g'''') = randn @ 'Dependent @TestLayout @TestDevice @TestDataType @( 'Shape '[TestBatchDim, TestKeySeqDim, TestValueEmbedDim]) g'''
      (result, _) = multiheadAttention @TestBatchDim @TestQuerySeqDim @TestKeySeqDim @TestHeadDim @TestHeadEmbedDim mha query key value g''''
  pure result
      


-- multiheadAttention ::
--   forall device dataType embedSize kEmbedSize vEmbedSize dropoutP requiresGradient layout batchSize seqLen seqLen' headSize.
--   (KnownNat embedSize, KnownNat headSize, KnownNat batchSize, ((Div embedSize headSize) * headSize) ~ embedSize, 1 <= headSize,
--    UnifyDataTypeF (UnifyDataTypeF dataType dataType) dataType ~ dataType
--    --(LinearF ( 'Shape '[outputFeatures, inputFeatures]) ( 'Shape '[outputFeatures]) shape' ~ )
--   ) =>
--   -- | multi-head attention model
--   MultiheadAttention device dataType ('Dim ( 'NamedSized "embed" embedSize)) ('Dim ( 'NamedSized "keyEmbed" kEmbedSize)) ('Dim ( 'NamedSized "valueEmbed" vEmbedSize)) dropoutP ->
--   -- | optional attention mask
--   Maybe (Tensor requiresGradient layout device dataType ( 'Shape '[ 'Dim ( 'NamedSized "batch" batchSize), 'Dim ( 'Sized seqLen'), 'Dim ( 'Sized seqLen)])) ->
--   -- | optional key padding mask
--   Maybe (Tensor requiresGradient layout device ( 'DataType 'Bool) ( 'Shape '[ 'Dim ( 'NamedSized "batch" batchSize), 'Dim ( 'Sized seqLen)])) ->
--   -- | optional key relations
--   Maybe (Tensor requiresGradient layout device dataType ( 'Shape '[ 'Dim ( 'NamedSized "batch" batchSize), 'Dim ( 'Sized seqLen'), 'Dim ( 'Sized seqLen), 'Dim ( 'Sized headSize)])) ->
--   -- | optional value relations
--   Maybe (Tensor requiresGradient layout device dataType ( 'Shape '[ 'Dim ( 'NamedSized "batch" batchSize), 'Dim ( 'Sized seqLen'), 'Dim ( 'Sized seqLen), 'Dim ( 'Sized headSize)])) ->
--   -- | query representation
--   Tensor requiresGradient layout device dataType ( 'Shape '[ 'Dim ( 'NamedSized "batch" batchSize), 'Dim ( 'Sized seqLen'), 'Dim ( 'NamedSized "embed" embedSize)]) ->
--   -- | key representation
--   Tensor requiresGradient layout device dataType ( 'Shape '[ 'Dim ( 'NamedSized "batch" batchSize), 'Dim ( 'Sized seqLen), 'Dim ( 'NamedSized "keyEmbed" kEmbedSize)]) ->
--   -- | value representation
--   Tensor requiresGradient layout device dataType ( 'Shape '[ 'Dim ( 'NamedSized "batch" batchSize), 'Dim ( 'Sized seqLen), 'Dim ( 'NamedSized "valueEmbed" vEmbedSize)]) ->
--   -- | attention and attention averaged over heads
--   Generator device ->
--   ( ( Tensor requiresGradient layout device dataType ( 'Shape '[ 'Dim ( 'NamedSized "batch" batchSize), 'Dim ( 'Sized seqLen'), 'Dim ( 'NamedSized "embed" embedSize)]),
--       Tensor requiresGradient layout device dataType ( 'Shape '[ 'Dim ( 'NamedSized "batch" batchSize), 'Dim ( 'Sized seqLen'), 'Dim ( 'Sized seqLen)])
--     ),
--     Generator device
--   )
-- multiheadAttention MultiheadAttention {..} attentionMask keyPaddingMask keyRelations valueRelations query key value =
--   runState $ do
--   -- weights <- state (\t g -> forward mhaDropout t g)
--   pure undefined
--   where
--     reshape' ::
--       forall seqLen''.
--       (KnownNat seqLen'') =>
--       Tensor requiresGradient layout device dataType ( 'Shape '[ 'Dim ('NamedSized "batch" batchSize), 'Dim ( 'Sized seqLen''), 'Dim ( 'NamedSized "embed" embedSize)]) ->
--       Tensor requiresGradient layout device dataType ( 'Shape '[ 'Dim ('NamedSized "batch" batchSize), 'Dim ( 'Sized (Div embedSize headSize)), 'Dim ( 'Sized seqLen''), 'Dim ( 'Sized headSize)])
--     reshape' = 
--       transpose @('SelectDim ('ByIndex 1)) @('SelectDim ('ByIndex 2)) . 
--         reshape @( 'Shape '[ 'Dim ('NamedSized "batch" batchSize), 'Dim ( 'Sized seqLen''), 'Dim ( 'Sized (Div embedSize headSize)), 'Dim ( 'Sized headSize)])
--     attend attentionWeights =
--       let v = reshape' . forward mhaVInProj $ value
--           attention = transpose @('SelectDim ('ByIndex 1)) @('SelectDim ('ByIndex 2)) $ attentionWeights `matmul` v
--           attention' = case valueRelations of
--             Nothing -> attention
--             Just vr -> attention `add` ((transpose @('SelectDim ('ByIndex 1)) @('SelectDim ('ByIndex 2)) attentionWeights) `matmul` vr)
--        in forward mhaOutProj . reshape @( 'Shape '[ 'Dim ('NamedSized "batch" batchSize), 'Dim ('Sized seqLen'), 'Dim ( 'NamedSized "embed" embedSize)]) $ attention'
--   -- pure (_attention weights, averageOverHeads weights)
--   -- where 
--   --   _attentionWeights =
--   --     let scaling = Prelude.sqrt . fromIntegral $ _headDim :: Double
--   --         q = reshape' . divScalar scaling . forward mhaQInProj $ query
--   --         k = reshape' . forward mhaKInProj $ key
--   --         weights = matmul q (transpose @2 @3 k)
--   --         weights' = case keyRelations of
--   --           Nothing -> weights
--   --           Just kr -> weights `add` transpose @1 @2 ((transpose @1 @2 q) `matmul` (transpose @2 @3 kr))
--   --      in weights'
--   --   _maskAttention attentionWeights =
--   --     case attentionMask of
--   --       Nothing -> attentionWeights
--   --       Just am -> attentionWeights `add` unsqueeze @1 am
--   --   _maskKeyPaddings attentionWeights =
--   --     case keyPaddingMask of
--   --       Nothing -> attentionWeights
--   --       Just kpm ->
--   --         let keyPaddingMask' = unsqueeze @2 . unsqueeze @1 $ kpm
--   --          in maskedFill keyPaddingMask' (-1 / 0 :: Double) attentionWeights
--   --   _attention attentionWeights =
--   --     let v = reshape' . forward mhaVInProj $ value
--   --         attention = transpose @1 @2 $ matmul attentionWeights v
--   --         attention' = case valueRelations of
--   --           Nothing -> attention
--   --           Just vr -> attention `add` (matmul (transpose @1 @2 attentionWeights) vr)
--   --      in forward mhaOutProj . reshape @'[batchSize, seqLen', queryEmbedDim] $ attention'
--   --   averageOverHeads =
--   --     let numHeads' = natValI @numHeads
--   --      in divScalar numHeads' . sumDim @1

-- -- multiheadAttention ::
-- --   forall queryEmbedDim keyEmbedDim valueEmbedDim numHeads seqLen seqLen' batchSize headDim dtype device.
-- --   ( 1 <= numHeads,
-- --     queryEmbedDim ~ (headDim * numHeads),
-- --     All KnownNat '[queryEmbedDim, keyEmbedDim, valueEmbedDim, numHeads, seqLen, seqLen', batchSize, headDim],
-- --     KnownDType dtype,
-- --     StandardFloatingPointDTypeValidation device dtype,
-- --     MatMulDTypeIsValid device dtype,
-- --     BasicArithmeticDTypeIsValid device dtype,
-- --     dtype ~ SumDType dtype,
-- --     SumDTypeIsValid device dtype,
-- --     KnownDevice device
-- --   ) =>
-- --   -- | multi-head attention model ADT
-- --   MultiheadAttention queryEmbedDim keyEmbedDim valueEmbedDim numHeads dtype device ->
-- --   -- | switch between training mode and evaluation mode (turns random dropout on and off)
-- --   Bool ->
-- --   -- | optional attention mask
-- --   Maybe (Tensor device dtype '[batchSize, seqLen', seqLen]) ->
-- --   -- | optional key padding mask
-- --   Maybe (Tensor device 'D.Bool '[batchSize, seqLen]) ->
-- --   -- | optional key relations
-- --   Maybe (Tensor device dtype '[batchSize, seqLen', seqLen, headDim]) ->
-- --   -- | optional value relations
-- --   Maybe (Tensor device dtype '[batchSize, seqLen', seqLen, headDim]) ->
-- --   -- | query representation
-- --   Tensor device dtype '[batchSize, seqLen', queryEmbedDim] ->
-- --   -- | key representation
-- --   Tensor device dtype '[batchSize, seqLen, keyEmbedDim] ->
-- --   -- | value representation
-- --   Tensor device dtype '[batchSize, seqLen, valueEmbedDim] ->
-- --   -- | attention and attention averaged over heads
-- --   IO
-- --     ( Tensor device dtype '[batchSize, seqLen', queryEmbedDim],
-- --       Tensor device dtype '[batchSize, seqLen', seqLen]
-- --     )
-- -- multiheadAttention MultiheadAttention {..} train attentionMask keyPaddingMask keyRelations valueRelations query key value = do
-- --   weights <-
-- --     dropoutForward mhaDropout train
-- --       . softmax @3
-- --       . _maskKeyPaddings
-- --       . _maskAttention
-- --       $ _attentionWeights
-- --   pure (_attention weights, averageOverHeads weights)
-- --   where
-- --     _attentionWeights =
-- --       let scaling = Prelude.sqrt . fromIntegral $ natValI @headDim :: Double
-- --           q = reshape' . divScalar scaling . forward mhaQInProj $ query
-- --           k = reshape' . forward mhaKInProj $ key
-- --           weights = matmul q (transpose @2 @3 k)
-- --           weights' = case keyRelations of
-- --             Nothing -> weights
-- --             Just kr -> weights `add` transpose @1 @2 ((transpose @1 @2 q) `matmul` (transpose @2 @3 kr))
-- --        in weights'
-- --     _maskAttention attentionWeights =
-- --       case attentionMask of
-- --         Nothing -> attentionWeights
-- --         Just am -> attentionWeights `add` unsqueeze @1 am
-- --     _maskKeyPaddings attentionWeights =
-- --       case keyPaddingMask of
-- --         Nothing -> attentionWeights
-- --         Just kpm ->
-- --           let keyPaddingMask' = unsqueeze @2 . unsqueeze @1 $ kpm
-- --            in maskedFill keyPaddingMask' (-1 / 0 :: Double) attentionWeights
-- --     _attention attentionWeights =
-- --       let v = reshape' . forward mhaVInProj $ value
-- --           attention = transpose @1 @2 $ matmul attentionWeights v
-- --           attention' = case valueRelations of
-- --             Nothing -> attention
-- --             Just vr -> attention `add` (matmul (transpose @1 @2 attentionWeights) vr)
-- --        in forward mhaOutProj . reshape @'[batchSize, seqLen', queryEmbedDim] $ attention'
-- --     averageOverHeads =
-- --       let numHeads' = natValI @numHeads
-- --        in divScalar numHeads' . sumDim @1
-- --     reshape' ::
-- --       forall seqLen''.
-- --       KnownNat seqLen'' =>
-- --       Tensor device dtype '[batchSize, seqLen'', queryEmbedDim] ->
-- --       Tensor device dtype '[batchSize, numHeads, seqLen'', headDim]
-- --     reshape' = transpose @1 @2 . reshape @'[batchSize, seqLen'', numHeads, headDim]

-- -- instance
-- --   ( All KnownNat '[queryEmbedDim, keyEmbedDim, valueEmbedDim, numHeads],
-- --     KnownDType dtype,
-- --     KnownDevice device,
-- --     RandDTypeIsValid device dtype
-- --   ) =>
-- --   A.Randomizable
-- --     (MultiheadAttentionSpec queryEmbedDim keyEmbedDim valueEmbedDim numHeads dtype device)
-- --     (MultiheadAttention queryEmbedDim keyEmbedDim valueEmbedDim numHeads dtype device)
-- --   where
-- --   sample (MultiheadAttentionSpec mhaDropoutSpec) =
-- --     MultiheadAttention
-- --       <$> A.sample LinearSpec
-- --       <*> A.sample LinearSpec
-- --       <*> A.sample LinearSpec
-- --       <*> A.sample LinearSpec
-- --       <*> A.sample mhaDropoutSpec

-- -- --------------------------------------------------------------------------------
-- -- -- Transformer MLP Layer
-- -- --------------------------------------------------------------------------------

-- -- data
-- --   TransformerMLPSpec
-- --     (queryEmbedDim :: Nat)
-- --     (ffnDim :: Nat)
-- --     (dtype :: D.DType)
-- --     (device :: (D.DeviceType, Nat))
-- --   where
-- --   TransformerMLPSpec ::
-- --     forall queryEmbedDim ffnDim dtype device.
-- --     { -- | spec for relu dropout
-- --       dropout0Spec :: DropoutSpec,
-- --       -- | spec for other dropout
-- --       dropout1Spec :: DropoutSpec,
-- --       -- | epsilon for layer norm
-- --       epsSpec :: Double
-- --     } ->
-- --     TransformerMLPSpec queryEmbedDim ffnDim dtype device
-- --   deriving (Show, Eq)

-- -- data
-- --   TransformerMLP
-- --     (queryEmbedDim :: Nat)
-- --     (ffnDim :: Nat)
-- --     (dtype :: D.DType)
-- --     (device :: (D.DeviceType, Nat))
-- --   where
-- --   TransformerMLP ::
-- --     forall queryEmbedDim ffnDim dtype device.
-- --     { -- | first fully connected layer
-- --       linear0 :: Linear queryEmbedDim ffnDim dtype device,
-- --       -- | second fully connected layer
-- --       linear1 :: Linear ffnDim queryEmbedDim dtype device,
-- --       -- | relu dropout
-- --       dropout0 :: Dropout,
-- --       -- | other dropout
-- --       dropout1 :: Dropout,
-- --       -- | layer norm
-- --       ln :: LayerNorm '[queryEmbedDim] dtype device
-- --     } ->
-- --     TransformerMLP queryEmbedDim ffnDim dtype device
-- --   deriving (Show, Generic, Parameterized)

-- -- transformerMLP ::
-- --   forall queryEmbedDim ffnDim seqLen batchSize dtype device.
-- --   ( BasicArithmeticDTypeIsValid device dtype,
-- --     StandardFloatingPointDTypeValidation device dtype,
-- --     KnownNat queryEmbedDim,
-- --     IsSuffixOf '[queryEmbedDim] '[seqLen, batchSize, queryEmbedDim]
-- --   ) =>
-- --   -- | MLP model ADT for transformer
-- --   TransformerMLP queryEmbedDim ffnDim dtype device ->
-- --   -- | switch between training mode and evaluation mode (turns random dropout on and off)
-- --   Bool ->
-- --   Tensor device dtype '[seqLen, batchSize, queryEmbedDim] -> -- input
-- --   IO (Tensor device dtype '[seqLen, batchSize, queryEmbedDim]) -- output
-- -- transformerMLP TransformerMLP {..} train input =
-- --   residual f (pure . forward ln) input
-- --   where
-- --     f x =
-- --       dropoutForward dropout1 train
-- --         . forward linear1
-- --         =<< dropoutForward dropout0 train
-- --           . relu
-- --           . forward linear0
-- --         =<< pure x

-- -- instance
-- --   ( All KnownNat '[queryEmbedDim, ffnDim],
-- --     KnownDType dtype,
-- --     KnownDevice device,
-- --     RandDTypeIsValid device dtype
-- --   ) =>
-- --   A.Randomizable
-- --     (TransformerMLPSpec queryEmbedDim ffnDim dtype device)
-- --     (TransformerMLP queryEmbedDim ffnDim dtype device)
-- --   where
-- --   sample TransformerMLPSpec {..} =
-- --     TransformerMLP
-- --       <$> A.sample LinearSpec
-- --       <*> A.sample LinearSpec
-- --       <*> A.sample dropout0Spec
-- --       <*> A.sample dropout1Spec
-- --       <*> A.sample (LayerNormSpec epsSpec)

-- -- --------------------------------------------------------------------------------
-- -- -- Relation-Aware Transformer Layer
-- -- --------------------------------------------------------------------------------

-- -- data
-- --   TransformerLayerSpec
-- --     (queryEmbedDim :: Nat)
-- --     (keyEmbedDim :: Nat)
-- --     (valueEmbedDim :: Nat)
-- --     (numHeads :: Nat)
-- --     (ffnDim :: Nat)
-- --     (dtype :: D.DType)
-- --     (device :: (D.DeviceType, Nat))
-- --   where
-- --   TransformerLayerSpec ::
-- --     forall queryEmbedDim keyEmbedDim valueEmbedDim numHeads ffnDim dtype device.
-- --     { mhaSpec :: MultiheadAttentionSpec queryEmbedDim keyEmbedDim valueEmbedDim numHeads dtype device,
-- --       attnDropoutSpec :: DropoutSpec,
-- --       epsSpec' :: Double,
-- --       mlpSpec :: TransformerMLPSpec queryEmbedDim ffnDim dtype device
-- --     } ->
-- --     TransformerLayerSpec queryEmbedDim keyEmbedDim valueEmbedDim numHeads ffnDim dtype device
-- --   deriving (Show, Eq)

-- -- data
-- --   TransformerLayer
-- --     (queryEmbedDim :: Nat)
-- --     (keyEmbedDim :: Nat)
-- --     (valueEmbedDim :: Nat)
-- --     (numHeads :: Nat)
-- --     (ffnDim :: Nat)
-- --     (dtype :: D.DType)
-- --     (device :: (D.DeviceType, Nat))
-- --   where
-- --   TransformerLayer ::
-- --     forall queryEmbedDim keyEmbedDim valueEmbedDim numHeads ffnDim dtype device.
-- --     { -- | multi-head attention
-- --       transformerLayer_mha :: MultiheadAttention queryEmbedDim keyEmbedDim valueEmbedDim numHeads dtype device,
-- --       -- | dropout
-- --       transformerLayer_attnDropout :: Dropout,
-- --       -- | layer norm
-- --       transformerLayer_ln :: LayerNorm '[queryEmbedDim] dtype device,
-- --       -- | MLP
-- --       transformerLayer_mlp :: TransformerMLP queryEmbedDim ffnDim dtype device
-- --     } ->
-- --     TransformerLayer queryEmbedDim keyEmbedDim valueEmbedDim numHeads ffnDim dtype device
-- --   deriving (Show, Generic, Parameterized)

-- -- transformerLayer ::
-- --   forall (numHeads :: Nat) (ffnDim :: Nat) (queryEmbedDim :: Nat) (keyEmbedDim :: Nat) (valueEmbedDim :: Nat) (headDim :: Nat) (seqLen :: Nat) (seqLen' :: Nat) (batchSize :: Nat) dtype device.
-- --   ( 1 <= numHeads,
-- --     queryEmbedDim ~ (headDim * numHeads),
-- --     All KnownNat '[queryEmbedDim, keyEmbedDim, valueEmbedDim, numHeads, seqLen, seqLen', batchSize, headDim],
-- --     IsSuffixOf '[queryEmbedDim] '[batchSize, seqLen', queryEmbedDim],
-- --     KnownDType dtype,
-- --     dtype ~ SumDType dtype,
-- --     StandardFloatingPointDTypeValidation device dtype,
-- --     MatMulDTypeIsValid device dtype,
-- --     BasicArithmeticDTypeIsValid device dtype,
-- --     SumDTypeIsValid device dtype,
-- --     KnownDevice device
-- --   ) =>
-- --   -- | transformer layer model ADT
-- --   TransformerLayer queryEmbedDim keyEmbedDim valueEmbedDim numHeads ffnDim dtype device ->
-- --   -- | switch between training mode and evaluation mode (turns random dropout on and off)
-- --   Bool ->
-- --   -- | optional attention mask
-- --   Maybe (Tensor device dtype '[batchSize, seqLen', seqLen]) ->
-- --   -- | optional key padding mask
-- --   Maybe (Tensor device 'D.Bool '[batchSize, seqLen]) ->
-- --   -- | optional key relations
-- --   Maybe (Tensor device dtype '[batchSize, seqLen', seqLen, headDim]) ->
-- --   -- | optional value relations
-- --   Maybe (Tensor device dtype '[batchSize, seqLen', seqLen, headDim]) ->
-- --   -- | query representation
-- --   Tensor device dtype '[batchSize, seqLen', queryEmbedDim] ->
-- --   -- | key representation
-- --   Tensor device dtype '[batchSize, seqLen, keyEmbedDim] ->
-- --   -- | value representation
-- --   Tensor device dtype '[batchSize, seqLen, valueEmbedDim] ->
-- --   -- | transformer layer output representation
-- --   IO (Tensor device dtype '[batchSize, seqLen', queryEmbedDim])
-- -- transformerLayer TransformerLayer {..} train attentionMask keyPaddingMask keyRelations valueRelations query key value =
-- --   let f query' =
-- --         multiheadAttention transformerLayer_mha train attentionMask keyPaddingMask keyRelations valueRelations query' key value
-- --           >>= dropoutForward transformerLayer_attnDropout train . fst
-- --    in residual f (pure . forward transformerLayer_ln) query >>= transformerMLP transformerLayer_mlp train

-- -- instance
-- --   ( All KnownNat '[queryEmbedDim, keyEmbedDim, valueEmbedDim, numHeads, ffnDim],
-- --     KnownDType dtype,
-- --     KnownDevice device,
-- --     RandDTypeIsValid device dtype
-- --   ) =>
-- --   A.Randomizable
-- --     (TransformerLayerSpec queryEmbedDim keyEmbedDim valueEmbedDim numHeads ffnDim dtype device)
-- --     (TransformerLayer queryEmbedDim keyEmbedDim valueEmbedDim numHeads ffnDim dtype device)
-- --   where
-- --   sample TransformerLayerSpec {..} =
-- --     TransformerLayer
-- --       <$> A.sample mhaSpec
-- --       <*> A.sample attnDropoutSpec
-- --       <*> A.sample (LayerNormSpec epsSpec')
-- --       <*> A.sample mlpSpec

-- -- --------------------------------------------------------------------------------
-- -- -- Transformer Language Model (GPT-2)
-- -- --------------------------------------------------------------------------------

-- -- data
-- --   TransformerLMSpec
-- --     (numAttnLayers :: Nat)
-- --     (numHeads :: Nat)
-- --     (ffnDim :: Nat)
-- --     (paddingIdx :: Nat)
-- --     (numEmbeds :: Nat)
-- --     (queryEmbedDim :: Nat)
-- --     (dtype :: D.DType)
-- --     (device :: (D.DeviceType, Nat))
-- --   where
-- --   TransformerLMSpec ::
-- --     forall numAttnLayers numHeads ffnDim paddingIdx numEmbeds queryEmbedDim dtype device.
-- --     { -- | dropout spec
-- --       lmDropoutSpec :: DropoutSpec,
-- --       -- | spec for each and every transformer layer
-- --       lmLayerSpec :: TransformerLayerSpec queryEmbedDim queryEmbedDim queryEmbedDim numHeads ffnDim dtype device
-- --     } ->
-- --     TransformerLMSpec numAttnLayers numHeads ffnDim paddingIdx numEmbeds queryEmbedDim dtype device
-- --   deriving (Show, Eq)

-- -- data
-- --   TransformerLM
-- --     (numAttnLayers :: Nat)
-- --     (numHeads :: Nat)
-- --     (ffnDim :: Nat)
-- --     (paddingIdx :: Nat)
-- --     (numEmbeds :: Nat)
-- --     (queryEmbedDim :: Nat)
-- --     (dtype :: D.DType)
-- --     (device :: (D.DeviceType, Nat))
-- --   where
-- --   TransformerLM ::
-- --     forall numAttnLayers numHeads ffnDim paddingIdx numEmbeds queryEmbedDim dtype device.
-- --     { -- | token embedding
-- --       tEmbedding :: Embedding ('Just paddingIdx) numEmbeds queryEmbedDim 'Learned dtype device,
-- --       -- | positional embedding
-- --       tPosEmbedding :: Embedding 'Nothing 2048 queryEmbedDim 'Constant dtype device,
-- --       -- | transformer dropout
-- --       tDropout :: Dropout,
-- --       -- | transformer layers
-- --       tLayers :: HList (HReplicateR numAttnLayers (TransformerLayer queryEmbedDim queryEmbedDim queryEmbedDim numHeads ffnDim dtype device)),
-- --       -- | final output projection
-- --       tProj :: Linear queryEmbedDim numEmbeds dtype device
-- --     } ->
-- --     TransformerLM numAttnLayers numHeads ffnDim paddingIdx numEmbeds queryEmbedDim dtype device
-- --   deriving (Generic)

-- -- deriving instance
-- --   ( Show
-- --       ( HList
-- --           ( HReplicateR
-- --               numAttnLayers
-- --               ( TransformerLayer
-- --                   queryEmbedDim
-- --                   queryEmbedDim
-- --                   queryEmbedDim
-- --                   numHeads
-- --                   ffnDim
-- --                   dtype
-- --                   device
-- --               )
-- --           )
-- --       )
-- --   ) =>
-- --   Show (TransformerLM numAttnLayers numHeads ffnDim paddingIdx numEmbeds queryEmbedDim dtype device)

-- -- instance
-- --   ( layers
-- --       ~ ( HReplicateR
-- --             numAttnLayers
-- --             ( TransformerLayer
-- --                 queryEmbedDim
-- --                 queryEmbedDim
-- --                 queryEmbedDim
-- --                 numHeads
-- --                 ffnDim
-- --                 dtype
-- --                 device
-- --             )
-- --         ),
-- --     Parameterized
-- --       ( HList
-- --           layers
-- --       ),
-- --     HAppendFD
-- --       (Parameters (HList layers))
-- --       '[ Parameter device dtype '[numEmbeds, queryEmbedDim],
-- --          Parameter device dtype '[numEmbeds]
-- --        ]
-- --       ( Parameters (HList layers)
-- --           ++ '[ Parameter device dtype '[numEmbeds, queryEmbedDim],
-- --                 Parameter device dtype '[numEmbeds]
-- --               ]
-- --       )
-- --   ) =>
-- --   Parameterized (TransformerLM numAttnLayers numHeads ffnDim paddingIdx numEmbeds queryEmbedDim dtype device)

-- -- data
-- --   FoldLayers
-- --     (batchSize :: Nat)
-- --     (seqLen :: Nat)
-- --     (dtype :: D.DType)
-- --     (device :: (D.DeviceType, Nat)) = FoldLayers
-- --   { -- | switch between training mode and evaluation mode (turns random dropout on and off)
-- --     flTrain :: Bool,
-- --     -- | optional attention mask
-- --     flAttentionMask :: Maybe (Tensor device dtype '[batchSize, seqLen, seqLen]),
-- --     -- | optional key padding mask
-- --     flKeyPaddingMask :: Maybe (Tensor device 'D.Bool '[batchSize, seqLen])
-- --   }

-- -- instance
-- --   ( 1 <= numHeads,
-- --     queryEmbedDim ~ (headDim * numHeads),
-- --     All KnownNat '[queryEmbedDim, numHeads, seqLen, batchSize, headDim],
-- --     IsSuffixOf '[queryEmbedDim] '[batchSize, seqLen, queryEmbedDim],
-- --     KnownDType dtype,
-- --     StandardFloatingPointDTypeValidation device dtype,
-- --     MatMulDTypeIsValid device dtype,
-- --     BasicArithmeticDTypeIsValid device dtype,
-- --     dtype ~ SumDType dtype,
-- --     SumDTypeIsValid device dtype,
-- --     KnownDevice device
-- --   ) =>
-- --   Apply'
-- --     (FoldLayers batchSize seqLen dtype device)
-- --     ( TransformerLayer queryEmbedDim queryEmbedDim queryEmbedDim numHeads ffnDim dtype device,
-- --       IO (Tensor device dtype '[batchSize, seqLen, queryEmbedDim])
-- --     )
-- --     (IO (Tensor device dtype '[batchSize, seqLen, queryEmbedDim]))
-- --   where
-- --   apply' FoldLayers {..} (layer, mx) = mx >>= \x -> transformerLayer layer flTrain flAttentionMask flKeyPaddingMask Nothing Nothing x x x

-- -- transformerLM ::
-- --   forall
-- --     numAttnLayers
-- --     numHeads
-- --     ffnDim
-- --     paddingIdx
-- --     numEmbeds
-- --     queryEmbedDim
-- --     seqLen
-- --     batchSize
-- --     dtype
-- --     device.
-- --   ( All KnownNat '[paddingIdx, queryEmbedDim, seqLen, batchSize],
-- --     paddingIdx + 1 <= numEmbeds,
-- --     1 <= seqLen,
-- --     HFoldrM
-- --       IO
-- --       (FoldLayers batchSize seqLen dtype device)
-- --       (Tensor device dtype '[batchSize, seqLen, queryEmbedDim])
-- --       (HReplicateR numAttnLayers (TransformerLayer queryEmbedDim queryEmbedDim queryEmbedDim numHeads ffnDim dtype device))
-- --       (Tensor device dtype '[batchSize, seqLen, queryEmbedDim]),
-- --     BasicArithmeticDTypeIsValid device dtype,
-- --     ComparisonDTypeIsValid device dtype,
-- --     ComparisonDTypeIsValid device 'D.Int64,
-- --     KnownDType dtype,
-- --     KnownDevice device
-- --   ) =>
-- --   TransformerLM numAttnLayers numHeads ffnDim paddingIdx numEmbeds queryEmbedDim dtype device ->
-- --   Bool ->
-- --   Tensor device 'D.Int64 '[batchSize, seqLen] ->
-- --   IO (Tensor device dtype '[batchSize, seqLen, numEmbeds])
-- -- transformerLM TransformerLM {..} train xTokens = do
-- --   let x = embed tEmbedding xTokens
-- --       positions =
-- --         expand @'[batchSize, seqLen, queryEmbedDim] True
-- --           . embed tPosEmbedding
-- --           . Torch.Typed.Tensor.toDType @D.Int64
-- --           . linspace @seqLen (0 :: Int)
-- --           $ natValI @(seqLen - 1)
-- --   x' <- dropoutForward tDropout train (x `add` positions)
-- --   let attentionMask =
-- --         unsqueeze @0
-- --           . Torch.Typed.Tensor.toDType @D.Bool
-- --           . triu 1
-- --           $ ones @'[seqLen, seqLen] @D.Int8 @device
-- --       attentionMask' =
-- --         pure . maskedFill attentionMask (-1 / 0 :: Double) $
-- --           zeros @'[batchSize, seqLen, seqLen] @dtype @device
-- --   let keyPaddingMask = pure $ xTokens ==. (fromInteger . natVal $ Proxy @paddingIdx :: Tensor device 'D.Int64 '[])
-- --   y <- hfoldrM (FoldLayers train attentionMask' keyPaddingMask) x' tLayers
-- --   return $ forward tProj y

-- -- instance
-- --   ( All KnownNat '[paddingIdx, queryEmbedDim, seqLen, batchSize],
-- --     paddingIdx + 1 <= numEmbeds,
-- --     1 <= seqLen,
-- --     HFoldrM
-- --       IO
-- --       (FoldLayers batchSize seqLen dtype device)
-- --       (Tensor device dtype '[batchSize, seqLen, queryEmbedDim])
-- --       (HReplicateR numAttnLayers (TransformerLayer queryEmbedDim queryEmbedDim queryEmbedDim numHeads ffnDim dtype device))
-- --       (Tensor device dtype '[batchSize, seqLen, queryEmbedDim]),
-- --     BasicArithmeticDTypeIsValid device dtype,
-- --     ComparisonDTypeIsValid device dtype,
-- --     ComparisonDTypeIsValid device 'D.Int64,
-- --     KnownDType dtype,
-- --     KnownDevice device
-- --   ) =>
-- --   HasForward (TransformerLM numAttnLayers numHeads ffnDim paddingIdx numEmbeds queryEmbedDim dtype device) (Tensor device 'D.Int64 '[batchSize, seqLen]) (Tensor device dtype '[batchSize, seqLen, numEmbeds])
-- --   where
-- --   forward model input = unsafePerformIO $ transformerLM model False input
-- --   forwardStoch model input = transformerLM model True input

-- -- sinusoidal ::
-- --   forall numEmbeds queryEmbedDim device.
-- --   ( All KnownNat '[numEmbeds, queryEmbedDim],
-- --     1 <= numEmbeds,
-- --     1 <= Div queryEmbedDim 2,
-- --     (Div queryEmbedDim 2 * 2) ~ queryEmbedDim,
-- --     StandardFloatingPointDTypeValidation device 'D.Float,
-- --     BasicArithmeticDTypeIsValid device 'D.Float,
-- --     KnownDevice device
-- --   ) =>
-- --   Tensor device 'D.Float '[numEmbeds, queryEmbedDim]
-- -- sinusoidal =
-- --   let positions =
-- --         unsqueeze @1
-- --           . linspace @numEmbeds (0 :: Int)
-- --           $ natValI @(numEmbeds - 1)
-- --       scalingFactors =
-- --         exp
-- --           . mulScalar (- log (10000 :: Double) / (fromInteger . natVal $ Proxy @(Div queryEmbedDim 2)))
-- --           . linspace @(Div queryEmbedDim 2) (0 :: Int)
-- --           $ natValI @((Div queryEmbedDim 2) - 1)
-- --       radians = mul positions scalingFactors
-- --       weights = stack @2 (sin radians :. cos radians :. HNil)
-- --    in reshape weights

-- -- instance
-- --   ( paddingIdx <= numEmbeds,
-- --     1 <= numEmbeds - paddingIdx,
-- --     (((numEmbeds - paddingIdx) - 1) + (1 + paddingIdx)) ~ numEmbeds,
-- --     (Div queryEmbedDim 2 * 2) ~ queryEmbedDim,
-- --     All KnownNat '[ffnDim, paddingIdx, numEmbeds, queryEmbedDim],
-- --     HReplicate numAttnLayers (TransformerLayerSpec queryEmbedDim queryEmbedDim queryEmbedDim numHeads ffnDim dtype device),
-- --     A.Randomizable
-- --       (HList (HReplicateR numAttnLayers (TransformerLayerSpec queryEmbedDim queryEmbedDim queryEmbedDim numHeads ffnDim dtype device)))
-- --       (HList (HReplicateR numAttnLayers (TransformerLayer queryEmbedDim queryEmbedDim queryEmbedDim numHeads ffnDim dtype device))),
-- --     KnownDType dtype,
-- --     RandDTypeIsValid device dtype,
-- --     StandardFloatingPointDTypeValidation device 'D.Float,
-- --     BasicArithmeticDTypeIsValid device 'D.Float,
-- --     KnownDevice device
-- --   ) =>
-- --   A.Randomizable
-- --     (TransformerLMSpec numAttnLayers numHeads ffnDim paddingIdx numEmbeds queryEmbedDim dtype device)
-- --     (TransformerLM numAttnLayers numHeads ffnDim paddingIdx numEmbeds queryEmbedDim dtype device)
-- --   where
-- --   sample TransformerLMSpec {..} =
-- --     TransformerLM
-- --       <$> A.sample (LearnedEmbeddingWithRandomInitSpec @('Just paddingIdx))
-- --       <*> A.sample (ConstEmbeddingSpec @'Nothing (Torch.Typed.Tensor.toDType sinusoidal))
-- --       <*> A.sample lmDropoutSpec
-- --       <*> A.sample (hreplicate @numAttnLayers lmLayerSpec)
-- --       <*> A.sample LinearSpec
