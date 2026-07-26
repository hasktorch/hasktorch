{-# LANGUAGE Arrows #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE NoStarIsType #-}

-- | Networks as arrows: a CNN wired with '>>>', a ResNet-style skip via
-- 'residual', and @proc@ notation — with no @PartialTypeSignatures@ and no
-- per-example @HasForward@ instances, which is the point.
module Torch.Typed.ArrowSpec (spec) where

import Control.Arrow
import Control.Category (id, (.), (>>>))
import Control.Monad (foldM)
import GHC.Generics (Generic)
import Test.Hspec
import Torch.Internal.Managed.Type.Context (manual_seed_L)
import qualified Torch.DType as D
import qualified Torch.Device as D
import qualified Torch.Tensor as UT
import Torch.Typed hiding (length, replicate, residual)
import Torch.Typed.NN.Arrow
import Prelude hiding (id, (.))

type Dev = '(D.CPU, 0)

type T shape = Tensor Dev 'D.Float shape

type B = 4

--------------------------------------------------------------------------------
-- A CNN as an arrow: parameters in a record, wiring as a pipeline
--------------------------------------------------------------------------------

data CNN = CNN
  { c1 :: Conv2d 1 8 3 3 'D.Float Dev,
    c2 :: Conv2d 8 16 3 3 'D.Float Dev,
    fc :: Linear 784 10 'D.Float Dev
  }
  deriving (Generic, Parameterized)

data CNNSpec = CNNSpec

instance Randomizable CNNSpec CNN where
  sample CNNSpec = CNN <$> sample Conv2dSpec <*> sample Conv2dSpec <*> sample LinearSpec

-- 28 -conv3x3/s1/p1-> 28 -pool2-> 14 -conv-> 14 -pool2-> 7 -flatten-> 784 -fc-> 10
cnn :: CNN -> Net (T '[B, 1, 28, 28]) (T '[B, 10])
cnn CNN {..} =
  arr (conv2dForward @'(1, 1) @'(1, 1) c1)
    >>> arr (mulScalar (0.1 :: Float)) -- tame the unscaled default init
    >>> arr relu
    >>> arr (maxPool2d @'(2, 2) @'(2, 2) @'(0, 0))
    >>> arr (conv2dForward @'(1, 1) @'(1, 1) c2)
    >>> arr (mulScalar (0.1 :: Float))
    >>> arr relu
    >>> arr (maxPool2d @'(2, 2) @'(2, 2) @'(0, 0))
    >>> arr (reshape @'[B, 784])
    >>> layer fc

--------------------------------------------------------------------------------
-- Tests
--------------------------------------------------------------------------------

lossOf :: CNN -> T '[B, 1, 28, 28] -> Tensor Dev 'D.Int64 '[B] -> IO (T '[])
lossOf m x y = do
  logits <- runNet (cnn m) x
  pure (nllLoss @ReduceMean ones (-100) (logSoftmax @1 logits) y)

spec :: Spec
spec = describe "networks as arrows" $ beforeAll_ (manual_seed_L 42) $ do
  it "a CNN wired with >>> produces the right shape" $ do
    model <- sample CNNSpec
    x <- randn :: IO (T '[B, 1, 28, 28])
    out <- runNet (cnn model) x
    shape out `shouldBe` [4, 10]
  it "residual computes f x + x" $ do
    let t = ones :: T '[2, 3]
    out <- runNet (residual (arr (mulScalar (2 :: Float)))) t
    (UT.asValue (UT.reshape [-1] (toDynamic out)) :: [Float]) `shouldBe` replicate 6 3
  it "proc notation works" $ do
    let net :: Net (T '[2]) (T '[2])
        net = proc x -> do
          y <- arr (mulScalar (2 :: Float)) -< x
          z <- residual id -< y
          returnA -< z
    out <- runNet net (ones :: T '[2])
    (UT.asValue (toDynamic out) :: [Float]) `shouldBe` [4, 4]
  it "trains end to end through the arrow" $ do
    model <- sample CNNSpec
    x <- randn :: IO (T '[B, 1, 28, 28])
    let y = UnsafeMkTensor (UT.asTensor ([0, 1, 2, 3] :: [Int])) :: Tensor Dev 'D.Int64 '[B]
        optim = mkAdam 0 0.9 0.999 (flattenParameters model)
    loss0 <- toFloat <$> lossOf model x y
    (trained, _) <-
      foldM
        ( \(m, o) _ -> do
            loss <- lossOf m x y
            runStep m o loss 1e-4
        )
        (model, optim)
        [1 .. 30 :: Int]
    loss1 <- toFloat <$> lossOf trained x y
    loss1 `shouldSatisfy` (< loss0)
