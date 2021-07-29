module TransformerSpec where

import Test.Hspec
import Torch.GraduallyTyped.NN.Transformer.GBlock (testEncoderBlock, testDecoderBlock)
import Torch.GraduallyTyped.NN.Transformer.GCrossAttention (testCA)
import Torch.GraduallyTyped.NN.Transformer.GTransformer (testEncoder, testDecoder)
import Torch.GraduallyTyped.NN.Transformer.GMultiHeadAttention (testMHA)
import Torch.GraduallyTyped.NN.Transformer.GSelfAttention (testSA)
import Torch.GraduallyTyped.NN.Transformer.GEncoderDecoder (testEncoderDecoderTransformer)
import Torch.GraduallyTyped.NN.Transformer.GStack (testEncoderStack, testDecoderStack)
import Torch.GraduallyTyped.NN.Transformer.GLMHead (testLMHead)

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
  context "encoder block" $ do
    it "minimal" $ do
      _ <- testEncoderBlock
      pure ()
  context "decoder block" $ do
    it "minimal" $ do
      _ <- testDecoderBlock
      pure ()
  context "encoder stack" $ do
    it "minimal" $ do
      _ <- testEncoderStack
      pure ()
  context "decoder stack" $ do
    it "minimal" $ do
      _ <- testDecoderStack
      pure ()
  context "encoder" $ do
    it "minimal" $ do
      _ <- testEncoder
      pure ()
  context "decoder" $ do
    it "minimal" $ do
      _ <- testDecoder
      pure ()
  context "encoder-decoder" $ do
    it "minimal" $ do
      _ <- testEncoderDecoderTransformer
      pure ()
  context "lm-head" $ do
    it "minimal" $ do
      _ <- testLMHead
      pure ()
