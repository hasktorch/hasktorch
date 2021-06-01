{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}

import Torch 

import GHC.Generics

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
  declSubLayer :: [SubLayerConnection]

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

data SubLayerConnection  = SubLayerConnection {
  slcNorm :: LayerNorm,
  slcDropout :: Dropout
} deriving (Show, Generic, Parameterized)


--
-- Multi-headed Attention
-- 

data MHAttention = MHAttention {
  mhaDimK :: Int,
  mhaHeads :: Int,
  mhaLinears :: [Linear],
  mhaAttention :: Attention,
  mhaDropout :: Dropout
} deriving (Show, Generic, Parameterized)

--
-- Attention
-- 

data Attention = Attention {

} deriving (Show, Generic, Parameterized)

data Dropout = Dropout {
} deriving (Show, Generic, Parameterized)

main = do
  putStrLn "Done"
