module Torch.GraduallyTyped.NN.TransformerSpec where

import Test.Hspec (Spec, context, describe, it)
import Torch.GraduallyTyped.NN.Transformer.GBlockSpec (testDecoderBlock, testEncoderBlock)
import Torch.GraduallyTyped.NN.Transformer.GCrossAttentionSpec (testCA)
import Torch.GraduallyTyped.NN.Transformer.GEncoderDecoderSpec (testEncoderDecoderTransformer)
import Torch.GraduallyTyped.NN.Transformer.GEncoderOnlySpec (testEncoderOnlyTransformer)
import Torch.GraduallyTyped.NN.Transformer.GLMHeadSpec (testLMHead)
import Torch.GraduallyTyped.NN.Transformer.GMultiHeadAttentionSpec (testMHA)
import Torch.GraduallyTyped.NN.Transformer.GSelfAttentionSpec (testSA)
import Torch.GraduallyTyped.NN.Transformer.GStackSpec (testDecoderStack, testEncoderStack)
import Torch.GraduallyTyped.NN.Transformer.GTransformerSpec (testDecoder, testEncoder)

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
  context "encoder-only" $ do
    it "minimal" $ do
      _ <- testEncoderOnlyTransformer
      pure ()
  context "encoder-decoder" $ do
    it "minimal" $ do
      _ <- testEncoderDecoderTransformer
      pure ()
  context "lm-head" $ do
    it "minimal" $ do
      _ <- testLMHead
      pure ()
