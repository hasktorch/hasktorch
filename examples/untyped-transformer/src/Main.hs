{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE ExistentialQuantification #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}

import Torch as T

import GHC.Generics
import System.IO.Unsafe (unsafePerformIO)

import Experimental

--
-- Encoder / Decoder
--

data EncoderDecoder = EncoderDecoder {
  encoder :: Encoder,
  decoder :: Decoder,
  srcEmbed :: Tensor -> Tensor,
  tgtEmbed :: Tensor -> Tensor,
  generator :: Generator
}

encodeDecode :: 
  EncoderDecoder ->
  -- | srcMask
  Tensor -> 
  -- | tgtMask
  Tensor -> 
  -- | source
  Tensor -> 
  -- | target
  Tensor ->
  Tensor
encodeDecode EncoderDecoder{..} srcMask tgtMask source target =
  decode decoder memory (tgtEmbed target) srcMask tgtMask memory
  where 
    memory = encode encoder (srcEmbed source) srcMask

--
-- Generator
--

data TFGenerator = TFGenerator {
  proj :: Linear
} deriving (Show, Generic, Parameterized)

tfgenerator TFGenerator{..} x =
  logSoftmax (Dim (-1)) $ linearForward proj x

--
-- Decoder
--

data Decoder = Decoder {
  decLayers :: [DecoderLayer],
  decNorm :: LayerNorm
} deriving (Show, Generic, Parameterized)

decode ::
  Decoder ->
  -- | memory (encoder output)
  Tensor ->
  -- | tgtEmbed
  Tensor ->
  -- | srcMask
  Tensor ->
  -- | tgtMask
  Tensor ->
  -- | input tensor
  Tensor ->
  -- | output
  Tensor
decode Decoder{..} memory tgtEmbed srcMask tgtMask t = 
  layerNorm decNorm layerOutputs
  where 
    layerOutputs = foldl (\accum f -> (f accum)) t decApplied  
    decApplied = fmap  
        (\layer -> decodeLayer layer tgtMask srcMask memory) decLayers

--
-- DecoderLayer 
-- 

data DecoderLayer = DecoderLayer {
  declSize :: [Int],
  declSelfAttn :: MHAttention,
  declSrcAttn :: MHAttention,
  declFF :: Linear,
  declSubLayer :: (SubLayerConnection, SubLayerConnection)

  -- dlSelfAttn
  -- dlSrcAttn
  -- dlfeedFwd
  -- dlSublayer
} deriving (Show, Generic, Parameterized)

decodeLayer :: 
  DecoderLayer ->
  Tensor ->
  Tensor ->
  Tensor ->
  Tensor ->
  Tensor
decodeLayer DecoderLayer{..} memory srcMask tgtMask t =
  undefined
  -- unsafePerformIO $ sublayer (fst declSubLayer) 
  

--
-- Encoder
-- 

data Encoder = Encoder {
  encLayers :: [EncoderLayer],
  encNorm :: !LayerNorm
} deriving (Show, Generic, Parameterized)

encode :: Encoder -> Tensor -> Tensor -> Tensor
encode Encoder{..} srcMask t = 
  layerNorm encNorm layerOutputs
  where
    layerOutputs = foldl (\accum f -> f accum) t encApplied
    encApplied = fmap
      (\layer -> encodeLayer layer srcMask) encLayers

--
-- EncodeLayer
-- 

data EncoderLayer = EncoderLayer {
  enclSelfAttn :: MHAttention,
  enclFF :: Linear,
  encSubLayers :: [SubLayerConnection],
  size :: Int
} deriving (Show, Generic, Parameterized)

encodeLayer = undefined

--
-- LayerNorm
-- 

data LayerNorm = LayerNorm {
  a2 :: !Parameter,
  b2 :: !Parameter,
  eps :: !Float
} deriving (Show, Generic, Parameterized)

layerNorm :: LayerNorm -> Tensor -> Tensor
layerNorm LayerNorm{..} x = a2' * (x - m) / (s + eps') + b2'
  where
    eps' = asTensor eps
    a2' = toDependent a2
    b2' = toDependent b2
    m = meanDim (Dim (-1)) KeepDim Float x
    s = meanDim (Dim (-1)) KeepDim Float x

--
-- SubLayer
-- 

-- placeholder type
data Dropout = Dropout {
} deriving (Show, Generic, Parameterized)

data SubLayerConnection  = SubLayerConnection {
  slcNorm :: LayerNorm,
  slcDropout :: Dropout
} deriving (Show, Generic, Parameterized)

sublayer ::
  SubLayerConnection ->
  -- | sublayer computation - either MHA, 
  (Tensor -> Tensor) ->
  -- | input
  Tensor ->
  -- | output
  IO Tensor
sublayer SubLayerConnection{..} subForward x = do
  -- TODO - what's the default dropout prob?
  fwd <- dropout 0.5 True (subForward $ layerNorm slcNorm x)
  pure (x + fwd)

--
-- Multi-headed Attention
-- 

data MHAttention = MHAttention {
  mhaDimK :: Int,
  mhaHeads :: Int,
  mhaLinears :: [Linear],
  mhaDropout :: Dropout
} deriving (Show, Generic, Parameterized)

attention :: 
  -- | query
  Tensor ->
  -- | key
  Tensor ->
  -- | value
  Tensor ->
  -- | mask
  Tensor -> 
  -- | dropout
  Dropout -> 
  -- | output
  IO (Tensor, Tensor)
attention query key value mask dropout = do
  let dk = asTensor $ T.size (-1) query
      keyTranspose = transpose (Dim (-2)) (Dim (-1)) key
      scores = (matmul query keyTranspose) / (T.sqrt dk)
      pAttn = softmax (Dim (-1)) scores
      -- TODO: dropout
  pure (matmul pAttn value, pAttn)

mha :: 
  -- | params
  MHAttention -> 
  -- | query
  Tensor ->
  -- | key
  Tensor ->
  -- | value
  Tensor ->
  -- | mask
  Tensor ->
  -- | output
  IO Tensor
mha MHAttention{..} query key value mask = do
  let mask' = unsqueeze (Dim 1) mask :: Tensor
  let nbatches = (T.size 0 query) :: Int

  let (query', key', value') = undefined

  (x :: Tensor, attn) <- attention query' key' value' mask mhaDropout
  let x' = view [nbatches, (-1), mhaHeads * mhaDimK] $ contiguous $ transpose (Dim 1) (Dim 2) x
  pure $ forward (last mhaLinears) x'

--
-- Position-wise Feed Forward
-- 

-- Position-wise Feed Forward 
data PositionwiseFF = PositionwiseFF {
  ffW1 :: Linear,
  ffW2 :: Linear,
  ffDropout:: Dropout
} deriving (Show, Generic, Parameterized)

-- TODO - add dropout
positionwiseFF ::
  PositionwiseFF ->
  Tensor ->
  Tensor
positionwiseFF PositionwiseFF{..} x =
  forward ffW2 $ relu $ forward ffW1 x

data Embeddings = Embeddings {
  embLut :: Embedding, -- look-up table
  embDModel :: Int
} deriving (Show, Generic, Parameterized)

--
-- Positional Encoding
-- 

data PositionalEncoding = PositionalEncoding {
  peDropout :: Dropout
  -- TODO
}

peForward = undefined



main = do
  putStrLn "Done"
