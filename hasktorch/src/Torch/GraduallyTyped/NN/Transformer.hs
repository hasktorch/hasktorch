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

module Torch.GraduallyTyped.NN.Transformer
  ( module Torch.GraduallyTyped.NN.Transformer,
    module Torch.GraduallyTyped.NN.Transformer.Block,
    module Torch.GraduallyTyped.NN.Transformer.CrossAttention,
    module Torch.GraduallyTyped.NN.Transformer.Decoder,
    module Torch.GraduallyTyped.NN.Transformer.DecoderBlock,
    module Torch.GraduallyTyped.NN.Transformer.DecoderStack,
    module Torch.GraduallyTyped.NN.Transformer.Encoder,
    module Torch.GraduallyTyped.NN.Transformer.FeedForwardNetwork,
    module Torch.GraduallyTyped.NN.Transformer.MultiHeadAttention,
    module Torch.GraduallyTyped.NN.Transformer.SelfAttention,
    module Torch.GraduallyTyped.NN.Transformer.SequenceToSequence,
    module Torch.GraduallyTyped.NN.Transformer.Stack,
  )
where

import Control.Monad.State.Strict (MonadState (state), runState)
import GHC.TypeLits (Nat)
import Torch.GraduallyTyped.DType (DataType (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..))
import Torch.GraduallyTyped.Layout (Layout (..), LayoutType (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..))
import Torch.GraduallyTyped.NN.Transformer.Block
import Torch.GraduallyTyped.NN.Transformer.CrossAttention
import Torch.GraduallyTyped.NN.Transformer.Decoder
import Torch.GraduallyTyped.NN.Transformer.DecoderBlock
import Torch.GraduallyTyped.NN.Transformer.DecoderStack
import Torch.GraduallyTyped.NN.Transformer.Encoder
import Torch.GraduallyTyped.NN.Transformer.FeedForwardNetwork
import Torch.GraduallyTyped.NN.Transformer.MultiHeadAttention
import Torch.GraduallyTyped.NN.Transformer.SelfAttention
import Torch.GraduallyTyped.NN.Transformer.SequenceToSequence
import Torch.GraduallyTyped.NN.Transformer.Stack
import Torch.GraduallyTyped.Random (mkGenerator)
import Torch.GraduallyTyped.RequiresGradient (RequiresGradient (..))
import Torch.GraduallyTyped.Shape.Type (Dim (..), Name (..), Shape (..), Size (..))
import Torch.GraduallyTyped.Tensor.Creation (randn)
import Torch.GraduallyTyped.Tensor.Type (Tensor)
import Torch.Typed (DType (..))

type TestDevice :: Device (DeviceType Nat)

type TestDevice = 'Device 'CPU

type TestLayout = 'Layout 'Dense

type TestDataType = 'DataType 'Float

type TestHeadDim = 'Dim ( 'Name "head") ( 'Size 12)

type TestHeadEmbedDim = 'Dim ( 'Name "headEmbed") ( 'Size 64)

type TestEmbedDim = 'Dim ( 'Name "*") ( 'Size 768)

type TestQueryEmbedDim = 'Dim ( 'Name "*") ( 'Size 512)

type TestKeyEmbedDim = 'Dim ( 'Name "*") ( 'Size 2048)

type TestValueEmbedDim = 'Dim ( 'Name "*") ( 'Size 1024)

type TestFFNDim = 'Dim ( 'Name "*") ( 'Size 256)

type TestBatchDim = 'Dim ( 'Name "*") ( 'Size 4)

type TestQuerySeqDim = 'Dim ( 'Name "*") ( 'Size 32)

type TestKeySeqDim = 'Dim ( 'Name "*") ( 'Size 48)

type TestInputEmbedDim = 'Dim ( 'Name "*") ( 'Size 512)

type TestDecoderInputEmbedDim = 'Dim ( 'Name "*") ( 'Size 1024)

type TestInputSeqDim = 'Dim ( 'Name "*") ( 'Size 32)

type TestDecoderInputSeqDim = 'Dim ( 'Name "*") ( 'Size 48)

-- testmha = do
--   g <- mkGenerator @TestDevice 0
--   let (result, _) =
--         runState
--           ( do
--               mha <- state $ initialize @(MultiHeadAttention TestDevice TestDataType TestHeadDim TestHeadEmbedDim TestEmbedDim TestQueryEmbedDim TestKeyEmbedDim TestValueEmbedDim Float) 0.0
--               query <- state $ randn @ 'Dependent @TestLayout @TestDevice @TestDataType @( 'Shape '[TestBatchDim, TestQuerySeqDim, TestQueryEmbedDim])
--               key <- state $ randn @ 'Dependent @TestLayout @TestDevice @TestDataType @( 'Shape '[TestBatchDim, TestKeySeqDim, TestKeyEmbedDim])
--               value <- state $ randn @ 'Dependent @TestLayout @TestDevice @TestDataType @( 'Shape '[TestBatchDim, TestKeySeqDim, TestValueEmbedDim])
--               attentionMask <- state $ randn @ 'Dependent @TestLayout @TestDevice @TestDataType @( 'Shape '[TestBatchDim, TestQuerySeqDim, TestKeySeqDim])
--               state $ forward mha (query, key, value, attentionMask)
--           )
--           g
--   pure result

-- testsa = do
--   g <- mkGenerator @TestDevice 0
--   let (result, _) =
--         runState
--           ( do
--               sa <- state $ initialize @(SelfAttention TestDevice TestDataType TestHeadDim TestHeadEmbedDim TestEmbedDim TestQueryEmbedDim Float) 0.0 1e-6
--               query <- state $ randn @ 'Dependent @TestLayout @TestDevice @TestDataType @( 'Shape '[TestBatchDim, TestQuerySeqDim, TestQueryEmbedDim])
--               attentionMask <- state $ randn @ 'Dependent @TestLayout @TestDevice @TestDataType @( 'Shape '[TestBatchDim, TestQuerySeqDim, TestQuerySeqDim])
--               state $ forward sa (query, attentionMask)
--           )
--           g
--   pure result

-- testca = do
--   g <- mkGenerator @TestDevice 0
--   let (result, _) =
--         runState
--           ( do
--               ca <- state $ initialize @(CrossAttention TestDevice TestDataType TestHeadDim TestHeadEmbedDim TestEmbedDim TestQueryEmbedDim TestKeyEmbedDim Float) 0.0 1e-6
--               query <- state $ randn @ 'Dependent @TestLayout @TestDevice @TestDataType @( 'Shape '[TestBatchDim, TestQuerySeqDim, TestQueryEmbedDim])
--               key <- state $ randn @ 'Dependent @TestLayout @TestDevice @TestDataType @( 'Shape '[TestBatchDim, TestKeySeqDim, TestKeyEmbedDim])
--               attentionMask <- state $ randn @ 'Dependent @TestLayout @TestDevice @TestDataType @( 'Shape '[TestBatchDim, TestQuerySeqDim, TestKeySeqDim])
--               state $ forward ca (query, key, attentionMask)
--           )
--           g
--   pure result

-- testffn = do
--   g <- mkGenerator @TestDevice 0
--   let (result, _) =
--         runState
--           ( do
--               ffn <- state $ initialize @(TransformerFeedForwardNetwork TestDevice TestDataType TestQueryEmbedDim TestFFNDim Float) 0.0 1e-6
--               query <- state $ randn @ 'Dependent @TestLayout @TestDevice @TestDataType @( 'Shape '[TestBatchDim, TestQuerySeqDim, TestQueryEmbedDim])
--               state $ forward ffn query
--           )
--           g
--   pure result

-- testBlock = do
--   g <- mkGenerator @TestDevice 0
--   let (result, _) =
--         runState
--           ( do
--               block <- state $ initialize @(TransformerBlock TestDevice TestDataType TestHeadDim TestHeadEmbedDim TestEmbedDim TestQueryEmbedDim TestFFNDim Float) 0.0 1e-6
--               query <- state $ randn @ 'Dependent @TestLayout @TestDevice @TestDataType @( 'Shape '[TestBatchDim, TestQuerySeqDim, TestQueryEmbedDim])
--               attentionMask <- state $ randn @ 'Dependent @TestLayout @TestDevice @TestDataType @( 'Shape '[TestBatchDim, TestQuerySeqDim, TestQuerySeqDim])
--               state $ forward block (query, attentionMask)
--           )
--           g
--   pure result

-- testDecoderBlock = do
--   g <- mkGenerator @TestDevice 0
--   let (result, _) =
--         runState
--           ( do
--               decoderBlock <- state $ initialize @(TransformerDecoderBlock TestDevice TestDataType TestHeadDim TestHeadEmbedDim TestEmbedDim TestQueryEmbedDim TestKeyEmbedDim TestFFNDim Float) 0.0 1e-6
--               query <- state $ randn @ 'Dependent @TestLayout @TestDevice @TestDataType @( 'Shape '[TestBatchDim, TestQuerySeqDim, TestQueryEmbedDim])
--               key <- state $ randn @ 'Dependent @TestLayout @TestDevice @TestDataType @( 'Shape '[TestBatchDim, TestKeySeqDim, TestKeyEmbedDim])
--               decoderAttentionMask <- state $ randn @ 'Dependent @TestLayout @TestDevice @TestDataType @( 'Shape '[TestBatchDim, TestQuerySeqDim, TestQuerySeqDim])
--               crossAttentionMask <- state $ randn @ 'Dependent @TestLayout @TestDevice @TestDataType @( 'Shape '[TestBatchDim, TestQuerySeqDim, TestKeySeqDim])
--               state $ forward decoderBlock (query, key, decoderAttentionMask, crossAttentionMask)
--           )
--           g
--   pure result

-- testStack = do
--   g <- mkGenerator @TestDevice 0
--   let (result, _) =
--         runState
--           ( do
--               stack <- state $ initialize @(TransformerStack 2 TestDevice TestDataType TestHeadDim TestHeadEmbedDim TestEmbedDim TestQueryEmbedDim TestFFNDim Float) 0.0 1e-6
--               query <- state $ randn @ 'Dependent @TestLayout @TestDevice @TestDataType @( 'Shape '[TestBatchDim, TestQuerySeqDim, TestQueryEmbedDim])
--               attentionMask <- state $ randn @ 'Dependent @TestLayout @TestDevice @TestDataType @( 'Shape '[TestBatchDim, TestQuerySeqDim, TestQuerySeqDim])
--               state $ forward stack (query, attentionMask)
--           )
--           g
--   pure result

-- testDecoderStack = do
--   g <- mkGenerator @TestDevice 0
--   let (result, _) =
--         runState
--           ( do
--               decoderStack <- state $ initialize @(TransformerDecoderStack 3 TestDevice TestDataType TestHeadDim TestHeadEmbedDim TestEmbedDim TestQueryEmbedDim TestKeyEmbedDim TestFFNDim Float) 0.0 1e-6
--               query <- state $ randn @ 'Dependent @TestLayout @TestDevice @TestDataType @( 'Shape '[TestBatchDim, TestQuerySeqDim, TestQueryEmbedDim])
--               key <- state $ randn @ 'Dependent @TestLayout @TestDevice @TestDataType @( 'Shape '[TestBatchDim, TestKeySeqDim, TestKeyEmbedDim])
--               decoderAttentionMask <- state $ randn @ 'Dependent @TestLayout @TestDevice @TestDataType @( 'Shape '[TestBatchDim, TestQuerySeqDim, TestQuerySeqDim])
--               crossAttentionMask <- state $ randn @ 'Dependent @TestLayout @TestDevice @TestDataType @( 'Shape '[TestBatchDim, TestQuerySeqDim, TestKeySeqDim])
--               state $ forward decoderStack (query, key, decoderAttentionMask, crossAttentionMask)
--           )
--           g
--   pure result

-- testEncoder = do
--   g <- mkGenerator @TestDevice 0
--   let (result, _) =
--         runState
--           ( do
--               encoder <- state $ initialize @(TransformerEncoder 2 TestDevice TestDataType TestHeadDim TestHeadEmbedDim TestEmbedDim TestInputEmbedDim TestFFNDim Float) 0.0 1e-6
--               input <- state $ randn @ 'Dependent @TestLayout @TestDevice @TestDataType @( 'Shape '[TestBatchDim, TestInputSeqDim, TestInputEmbedDim])
--               attentionMask <- state $ randn @ 'Dependent @TestLayout @TestDevice @TestDataType @( 'Shape '[TestBatchDim, TestInputSeqDim, TestInputSeqDim])
--               state $ forward encoder (input, attentionMask)
--           )
--           g
--   pure result

-- testDecoder = do
--   g <- mkGenerator @TestDevice 0
--   let (result, _) =
--         runState
--           ( do
--               decoder <- state $ initialize @(TransformerDecoder 1 TestDevice TestDataType TestHeadDim TestHeadEmbedDim TestEmbedDim TestDecoderInputEmbedDim TestInputEmbedDim TestFFNDim Float) 0.0 1e-6
--               decoderInput <- state $ randn @ 'Dependent @TestLayout @TestDevice @TestDataType @( 'Shape '[TestBatchDim, TestDecoderInputSeqDim, TestDecoderInputEmbedDim])
--               encoderOutput <- state $ randn @ 'Dependent @TestLayout @TestDevice @TestDataType @( 'Shape '[TestBatchDim, TestInputSeqDim, TestInputEmbedDim])
--               decoderAttentionMask <- state $ randn @ 'Dependent @TestLayout @TestDevice @TestDataType @( 'Shape '[TestBatchDim, TestDecoderInputSeqDim, TestDecoderInputSeqDim])
--               crossAttentionMask <- state $ randn @ 'Dependent @TestLayout @TestDevice @TestDataType @( 'Shape '[TestBatchDim, TestDecoderInputSeqDim, TestInputSeqDim])
--               state $ forward decoder (decoderInput, encoderOutput, decoderAttentionMask, crossAttentionMask)
--           )
--           g
--   pure result

-- testSequenceToSequence = do
--   g <- mkGenerator @TestDevice 0
--   let (result, _) =
--         runState
--           ( do
--               sequenceToSequence <- state $ initialize @(SequenceToSequenceTransformer 3 2 TestDevice TestDataType TestHeadDim TestHeadEmbedDim TestEmbedDim TestInputEmbedDim TestDecoderInputEmbedDim TestFFNDim Float) 0.0 1e-6
--               input <- state $ randn @ 'Dependent @TestLayout @TestDevice @TestDataType @( 'Shape '[TestBatchDim, TestInputSeqDim, TestInputEmbedDim])
--               decoderInput <- state $ randn @ 'Dependent @TestLayout @TestDevice @TestDataType @( 'Shape '[TestBatchDim, TestDecoderInputSeqDim, TestDecoderInputEmbedDim])
--               attentionMask <- state $ randn @ 'Dependent @TestLayout @TestDevice @TestDataType @( 'Shape '[TestBatchDim, TestInputSeqDim, TestInputSeqDim])
--               decoderAttentionMask <- state $ randn @ 'Dependent @TestLayout @TestDevice @TestDataType @( 'Shape '[TestBatchDim, TestDecoderInputSeqDim, TestDecoderInputSeqDim])
--               crossAttentionMask <- state $ randn @ 'Dependent @TestLayout @TestDevice @TestDataType @( 'Shape '[TestBatchDim, TestDecoderInputSeqDim, TestInputSeqDim])
--               state $ forward sequenceToSequence (input, decoderInput, attentionMask, decoderAttentionMask, crossAttentionMask)
--           )
--           g
--   pure result
