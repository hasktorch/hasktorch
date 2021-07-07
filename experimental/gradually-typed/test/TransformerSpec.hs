module TransformerSpec where

import Test.Hspec
import Torch.GraduallyTyped.NN.Transformer.Block (testBlock)
import Torch.GraduallyTyped.NN.Transformer.CrossAttention (testCA)
import Torch.GraduallyTyped.NN.Transformer.Decoder (testDecoder)
import Torch.GraduallyTyped.NN.Transformer.DecoderBlock (testDecoderBlock)
import Torch.GraduallyTyped.NN.Transformer.DecoderStack (testDecoderStack)
import Torch.GraduallyTyped.NN.Transformer.Encoder (testEncoder)
import Torch.GraduallyTyped.NN.Transformer.MultiHeadAttention (testMHA)
import Torch.GraduallyTyped.NN.Transformer.SelfAttention (testSA)
import Torch.GraduallyTyped.NN.Transformer.SequenceToSequence (testSeqToSeq)
import Torch.GraduallyTyped.NN.Transformer.Stack (testStack)

spec :: Spec
spec = describe "Transformer" $ do
  context "multi-head attention" $ do
    it "minimal" $ do
      _ <- testMHA
      pure ()
  context "self-attention" $ do
    it "minimal" $ do
      _ <- testSA
      pure ()
  context "cross-attention" $ do
    it "minimal" $ do
      _ <- testCA
      pure ()
  context "block" $ do
    it "minimal" $ do
      _ <- testBlock
      pure ()
  context "decoder block" $ do
    it "minimal" $ do
      _ <- testDecoderBlock
      pure ()
  context "stack" $ do
    it "minimal" $ do
      _ <- testStack
      pure ()
  context "encoder" $ do
    it "minimal" $ do
      _ <- testEncoder
      pure ()
  context "decoder stack" $ do
    it "minimal" $ do
      _ <- testDecoderStack
      pure ()
  context "decoder" $ do
    it "minimal" $ do
      _ <- testDecoder
      pure ()
  context "sequence-to-sequence" $ do
    it "minimal" $ do
      _ <- testSeqToSeq
      pure ()
