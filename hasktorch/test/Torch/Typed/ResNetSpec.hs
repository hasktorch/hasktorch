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

-- | A miniature ResNet and an Inception-style block, wired with arrows.
--
-- The earlier @feature/arrow@ branch needed 475 lines, per-example
-- 'HasForward' instances, config tuples, a @MutableTensor@ hack for batch
-- norm, and @PartialTypeSignatures@ — and its @deriving Parameterized@ was
-- commented out.  Here: parameters are ordinary derivable records, batch
-- norm's statefulness lives honestly in the arrow's 'IO', skip connections
-- are 'residual'/'residualWith', and multi-branch blocks are @proc@
-- notation.  No partial signatures anywhere.
module Torch.Typed.ResNetSpec (spec) where

import Control.Arrow
import Control.Category (id, (.), (>>>))
import Control.Monad (foldM)
import GHC.Generics (Generic)
import GHC.TypeLits
import Test.Hspec
import Torch.Internal.Managed.Type.Context (manual_seed_L)
import qualified Torch.DType as D
import qualified Torch.Device as D
import qualified Torch.Tensor as UT
import Torch.Typed hiding (length, replicate, residual)
import Torch.Typed.NN.Arrow
import Torch.Typed.NN.BatchNorm
import Prelude hiding (id, (.))

type Dev = '(D.CPU, 0)

type T shape = Tensor Dev 'D.Float shape

type B = 2

--------------------------------------------------------------------------------
-- Building blocks: derivable parameter records + arrow wiring
--------------------------------------------------------------------------------

data Block (c :: Nat) = Block
  { k1 :: Conv2d c c 3 3 'D.Float Dev,
    n1 :: BatchNorm2d c 'D.Float Dev,
    k2 :: Conv2d c c 3 3 'D.Float Dev,
    n2 :: BatchNorm2d c 'D.Float Dev
  }
  deriving (Generic, Parameterized)

data BlockSpec (c :: Nat) = BlockSpec

instance (KnownNat c) => Randomizable (BlockSpec c) (Block c) where
  sample BlockSpec =
    Block <$> sample Conv2dSpec <*> sample BatchNorm2dSpec <*> sample Conv2dSpec <*> sample BatchNorm2dSpec

-- conv3x3 -> bn -> relu -> conv3x3 -> bn, wrapped in a skip, then relu
identityBlock ::
  forall b c h w.
  ( All KnownNat '[b, c, h, w],
    ConvSideCheck h 3 1 1 h,
    ConvSideCheck w 3 1 1 w
  ) =>
  Bool ->
  Block c ->
  Net (T '[b, c, h, w]) (T '[b, c, h, w])
identityBlock train Block {..} =
  residual
    ( arr (conv2dForward @'(1, 1) @'(1, 1) k1)
        >>> Net (batchNorm2dForward n1 train)
        >>> arr relu
        >>> arr (conv2dForward @'(1, 1) @'(1, 1) k2)
        >>> Net (batchNorm2dForward n2 train)
    )
    >>> arr relu

data Down (cin :: Nat) (cout :: Nat) = Down
  { dk1 :: Conv2d cin cout 3 3 'D.Float Dev,
    dn1 :: BatchNorm2d cout 'D.Float Dev,
    dk2 :: Conv2d cout cout 3 3 'D.Float Dev,
    dn2 :: BatchNorm2d cout 'D.Float Dev,
    dproj :: Conv2d cin cout 1 1 'D.Float Dev,
    dnp :: BatchNorm2d cout 'D.Float Dev
  }
  deriving (Generic, Parameterized)

data DownSpec (cin :: Nat) (cout :: Nat) = DownSpec

instance (KnownNat cin, KnownNat cout) => Randomizable (DownSpec cin cout) (Down cin cout) where
  sample DownSpec =
    Down
      <$> sample Conv2dSpec
      <*> sample BatchNorm2dSpec
      <*> sample Conv2dSpec
      <*> sample BatchNorm2dSpec
      <*> sample Conv2dSpec
      <*> sample BatchNorm2dSpec

-- stride-2 block: the main path halves the resolution and changes channels,
-- so the skip needs a 1x1 stride-2 projection
downBlock ::
  forall b cin cout h w h' w'.
  ( All KnownNat '[b, cin, cout, h, w, h', w'],
    ConvSideCheck h 3 2 1 h',
    ConvSideCheck w 3 2 1 w',
    ConvSideCheck h' 3 1 1 h',
    ConvSideCheck w' 3 1 1 w',
    ConvSideCheck h 1 2 0 h',
    ConvSideCheck w 1 2 0 w'
  ) =>
  Bool ->
  Down cin cout ->
  Net (T '[b, cin, h, w]) (T '[b, cout, h', w'])
downBlock train Down {..} =
  residualWith
    (arr (conv2dForward @'(2, 2) @'(0, 0) dproj) >>> Net (batchNorm2dForward dnp train))
    ( arr (conv2dForward @'(2, 2) @'(1, 1) dk1)
        >>> Net (batchNorm2dForward dn1 train)
        >>> arr relu
        >>> arr (conv2dForward @'(1, 1) @'(1, 1) dk2)
        >>> Net (batchNorm2dForward dn2 train)
    )
    >>> arr relu

--------------------------------------------------------------------------------
-- The model: 32x32 -> stem -> 2 blocks at 8 -> down to 16 -> block -> head
--------------------------------------------------------------------------------

data ResNet = ResNet
  { stem :: Conv2d 3 8 3 3 'D.Float Dev,
    stemBn :: BatchNorm2d 8 'D.Float Dev,
    b1 :: Block 8,
    b2 :: Block 8,
    d1 :: Down 8 16,
    b3 :: Block 16,
    fc :: Linear 16 10 'D.Float Dev
  }
  deriving (Generic, Parameterized)

data ResNetSpec = ResNetSpec

instance Randomizable ResNetSpec ResNet where
  sample ResNetSpec =
    ResNet
      <$> sample Conv2dSpec
      <*> sample BatchNorm2dSpec
      <*> sample BlockSpec
      <*> sample BlockSpec
      <*> sample DownSpec
      <*> sample BlockSpec
      <*> sample LinearSpec

resnet :: Bool -> ResNet -> Net (T '[B, 3, 32, 32]) (T '[B, 10])
resnet train ResNet {..} =
  arr (conv2dForward @'(1, 1) @'(1, 1) stem)
    >>> Net (batchNorm2dForward stemBn train)
    >>> arr relu
    >>> identityBlock train b1
    >>> identityBlock train b2
    >>> downBlock train d1
    >>> identityBlock train b3
    >>> arr (adaptiveAvgPool2d @'(1, 1))
    >>> arr (reshape @'[B, 16])
    >>> layer fc

--------------------------------------------------------------------------------
-- An Inception-style block: parallel branches in proc notation
--------------------------------------------------------------------------------

data Inception = Inception
  { p1 :: Conv2d 8 4 1 1 'D.Float Dev,
    p3 :: Conv2d 8 4 3 3 'D.Float Dev,
    p5 :: Conv2d 8 4 5 5 'D.Float Dev
  }
  deriving (Generic, Parameterized)

data InceptionSpec = InceptionSpec

instance Randomizable InceptionSpec Inception where
  sample InceptionSpec = Inception <$> sample Conv2dSpec <*> sample Conv2dSpec <*> sample Conv2dSpec

-- three parallel convolutions at different receptive fields, concatenated
-- along the channel dimension
inception :: Inception -> Net (T '[B, 8, 16, 16]) (T '[B, 12, 16, 16])
inception Inception {..} = proc x -> do
  b1' <- arr (conv2dForward @'(1, 1) @'(0, 0) p1) -< x
  b3' <- arr (conv2dForward @'(1, 1) @'(1, 1) p3) -< x
  b5' <- arr (conv2dForward @'(1, 1) @'(2, 2) p5) -< x
  returnA -< cat @1 (b1' :. b3' :. b5' :. HNil)

--------------------------------------------------------------------------------
-- Tests
--------------------------------------------------------------------------------

lossOf :: ResNet -> T '[B, 3, 32, 32] -> Tensor Dev 'D.Int64 '[B] -> IO (T '[])
lossOf m x y = do
  logits <- runNet (resnet True m) x
  pure (nllLoss @ReduceMean ones (-100) (logSoftmax @1 logits) y)

spec :: Spec
spec = describe "a miniature ResNet as arrows" $ beforeAll_ (manual_seed_L 42) $ do
  it "runs forward with the right shape" $ do
    model <- sample ResNetSpec
    x <- randn :: IO (T '[B, 3, 32, 32])
    out <- runNet (resnet False model) x
    shape out `shouldBe` [2, 10]
  it "trains, updating batch-norm statistics in place along the way" $ do
    model <- sample ResNetSpec
    x <- randn :: IO (T '[B, 3, 32, 32])
    let y = UnsafeMkTensor (UT.asTensor ([3, 7] :: [Int])) :: Tensor Dev 'D.Int64 '[B]
        optim = mkAdam 0 0.9 0.999 (flattenParameters model)
    loss0 <- toFloat <$> lossOf model x y
    (trained, _) <-
      foldM
        ( \(m, o) _ -> do
            loss <- lossOf m x y
            runStep m o loss 1e-4
        )
        (model, optim)
        [1 .. 15 :: Int]
    loss1 <- toFloat <$> lossOf trained x y
    loss1 `shouldSatisfy` (< loss0)
  it "an Inception-style block branches and concatenates in proc notation" $ do
    model <- sample InceptionSpec
    x <- randn :: IO (T '[B, 8, 16, 16])
    out <- runNet (inception model) x
    shape out `shouldBe` [2, 12, 16, 16]
