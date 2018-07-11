{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE TemplateHaskell #-}
module Torch.Static.NN.LinearSpec where

import GHC.TypeLits
import Control.Monad (join, void)
import Data.Function ((&))
import Data.Foldable
import Debug.Trace
import GHC.Generics (Generic)
import Test.Hspec
import Lens.Micro.Platform
import Numeric.Backprop
import qualified Numeric.Backprop as B

import Torch.Double as Torch
import Torch.Double.NN.Linear
import qualified Torch.Long as Long

xavier :: forall d . Dimensions d => IO (Tensor d)
xavier = case (fromIntegral <$> listDims (dims :: Dims d)) of
  [] -> empty
  a:_ -> pure $ constant (1 / realToFrac (fromIntegral a))


data FF2Network i h o = FF2Network
  { _layer1 :: Linear i h
  , _layer2 :: Linear h o
  } deriving (Generic, Show)

weightsL :: Lens' (Linear i o) (Tensor '[i, o])
weightsL = lens weights $ \(Linear (_, b)) w' -> Linear (w', b)

biasL :: Lens' (Linear i o) (Tensor '[o])
biasL = lens bias $ \(Linear (w, _)) b' -> Linear (w, b')

update
  :: forall i h o
  .  All KnownDim '[i, h, o]
  => FF2Network i h o
  -> FF2Network i h o
  -> FF2Network i h o
update i g = FF2Network
  { _layer1 = B.add (_layer1 i) (_layer1 g)
  , _layer2 = B.add (_layer2 i) (_layer2 g)
  }

makeLenses ''FF2Network
instance (KnownDim i, KnownDim h, KnownDim o) => Backprop (FF2Network i h o)

-- ========================================================================= --

main :: IO ()
main = hspec spec

spec :: Spec
spec = do
  describe "a single linear layer" singleLayer
  describe "a two-layer feed forward network" $ do
    describe "with xavier initialization" twoLayerXavier
    describe "forcing ReLU activity"      twoLayerForceReLU
    describe "overfitting to [0, 1]" $ do
      describe "with softmax and binary cross-entropy" $
        twoLayerOverfit softmax (bCECriterion (unsafeVector [0,1])) id
      describe "with logSoftmax and multiclass log-loss" $
        twoLayerOverfit logSoftMax (classNLLCriterion (Long.unsafeVector [0,1])) Torch.exp

-- ========================================================================= --

singleLayer :: Spec
singleLayer = do
  ll :: Linear 3 2 <- runIO $ mkLinear xavier
  xavierPurityCheck ll $ do
    describe "the forward pass" $ do
      let y = constant 5 :: Tensor '[3]
          o = evalBP2 (linear undefined) ll y

      it "performs matrix multipication as you would expect" $ do
        o =##= ((5/3)*3) + 1/2
        o `elementsSatisfy` ((== 2) . length)

    describe "the backward pass" $ do
      let y = constant 1 :: Tensor '[3]
          lr = 1.0
          (_, (ll', o)) = backprop2 (linear lr) ll y

      it "returns updated weights" $ weights ll' =##= 1/2
      it "returns updated bias"    $ bias    ll' =##= 1/2

      it "returns the updated output tensor" $ o =##= 1/3

-- ========================================================================= --

mkXavierNetwork :: All KnownDim '[i,h,o] => IO (FF2Network i h o)
mkXavierNetwork =
  FF2Network
    <$> mkLinear xavier
    <*> mkLinear xavier


mkGaussianNetwork :: All KnownDim '[i,h,o] => IO (FF2Network i h o)
mkGaussianNetwork = do
  g <- newRNG
  manualSeed g 1
  let Just std = positive 2

  FF2Network
    <$> mkLinear (normal g 0 std)
    <*> mkLinear (normal g 0 std)


ff2network
  :: forall s i h o
  .  Reifies s W
  => All KnownDim '[i,h,o]
  => (forall s . Reifies s W => BVar s (Tensor '[o]) -> BVar s (Tensor '[o]))
  -> Double
  -> BVar s (FF2Network i h o)  -- ^ ff2network architecture
  -> BVar s (Tensor '[i])       -- ^ input
  -> BVar s (Tensor '[o])       -- ^ output
ff2network final lr arch inp
  = linear lr (arch ^^. layer1) inp
  & relu
  & linear lr (arch ^^. layer2)
  & final

twoLayerXavier :: Spec
twoLayerXavier = do
  ll :: FF2Network 4 6 2 <- runIO mkXavierNetwork
  describe "the forward pass" $ do
    describe "with xavier instantiation" $ do
      xavierPurityCheck (ll ^. layer1) $
        xavierPurityCheck (ll ^. layer2) $
          describe "with input all positive input" $ do
            let y = constant 4 :: Tensor '[4]
                (o, _) = backprop2 (ff2network softmax undefined) ll y

            it "performs matrix multipication as you would expect" $ o `approx` [1/2, 1/2]

      describe "with input that drops all values via ReLU" $ do
        let y = constant (-1) :: Tensor '[4]
            (o, _) = backprop2 (ff2network softmax undefined) ll y

        it "performs matrix multipication as you would expect" $ o `approx` [1/2, 1/2]
  where
    approx = lapproximately 0.0001


twoLayerForceReLU :: Spec
twoLayerForceReLU = do
  describe "operations that force relu activity" $ do
    let o1 = evalBP2 (relu    .: linear 1) (ff2 ^. layer1) oneInput
        o2 = evalBP2 (           linear 1) (ff2 ^. layer2) o1
        (out, (gff2, gin)) = backprop2 (ff2network logSoftMax 1) ff2 oneInput

    describe "dropping half the gradient during ReLU" $ do
      describe "the forward pass" $ do
        it "returns [0,0,0,4,4,4] after the first layer" $
          tensordata o1 >>= (`shouldBe` [0,0,0,4,4,4])

        it "returns [-12,0] after the second layer" $
          tensordata o2 >>= (`shouldBe` [-12, 0])

        it "returns [0, 1] as the output" $ do
          Torch.exp out `lapprox` [0, 1]

      describe "the backward pass" $ do
        it "returns a half zero-d out layer 1 gradient" $ do
          (gff2 ^. layer1 . weightsL) `approx` l1weightgrad

        it "returns a quarter zero-d out layer 2 gradient" $ do
          (gff2 ^. layer2 . weightsL) `approx` l2weightgrad

        it "returns a [3,3,3] input gradient" $ do
          gin `lapprox` replicate 4 (-3)

 where
  eps = 0.0001
  approx = approximately eps
  lapprox = lapproximately eps

  ff2 :: FF2Network 4 6 2
  ff2 = FF2Network
    (Linear (unsafeMatrix $ replicate 4 [ -1, -1, -1, 1, 1, 1], constant 0))
    (Linear (unsafeMatrix $ replicate 6                [-1, 0], constant 0))

  oneInput :: Tensor '[4]
  oneInput = constant 1

  l1weightgrad :: Tensor '[4, 6]
  l1weightgrad = unsafeMatrix $ replicate 4 [ 0, 0, 0,-1,-1,-1]

  l2weightgrad :: Tensor '[6, 2]
  l2weightgrad = unsafeMatrix $ replicateN 3
    [ [ 0, 0]
    , [ 4,-4]
    ]

twoLayerOverfit
  :: (forall s . Reifies s W => BVar s (Tensor '[2]) -> BVar s (Tensor '[2]))
  -> (forall s . Reifies s W => BVar s (Tensor '[2]) -> BVar s (Tensor '[1]))
  -> (Tensor '[2] -> Tensor '[2])
  -> Spec
twoLayerOverfit finalLayer loss postproc = do
  it "overfits on a single input with lr=0.3 and 100 steps" . void $ do
    net0 <- mkGaussianNetwork
    [l0, r0] <- tensordata $ infer net0 y
    (fnet, (fl, fr)) <-
      foldlM (\(net, (l, r)) i -> do
        let speedup = 2
            lr = 0.01
        let net' = train lr net y -- gradBP2 (bCECriterion True True Nothing t .: ff2network lr) net y
        -- print (grad ^. layer2 . weightsL)
        -- let net' = update net grad

        [l', r'] <- tensordata (infer net' y)
        -- print [l', r']
        (i, l, l') `shouldSatisfy` (\(_, l, l') -> l' < l || (l' - (0.2 * speedup * 2)) < l)
        (i, r, r') `shouldSatisfy` (\(_, r, r') -> r' > r || (r' + (0.2 * speedup * 2)) > r)
        pure (net', (l', r'))
        ) (net0, (l0, r0)) [1..50]
    let fin = (unsafeVector [fl, fr] :: Tensor '[2])
    -- print fin
    lapproximately 0.08 fin [0, 1]
  where
    eps = 0.0001
    approx     = approximately eps
    notApprox  = notCloseTo eps
    lapprox    = lapproximately eps
    lnotApprox = lnotCloseTo eps

    y :: Tensor '[4]
    y = constant 1

    -- t = Long.unsafeVector [0.01, 0.99] :: Long.Tensor '[2]
    -- t = unsafeVector [0, 1] :: Tensor '[2]

    train
      :: Double
      -> FF2Network 4 6 2  -- ^ ff2network architecture
      -> Tensor '[4]       -- ^ input
      -> FF2Network 4 6 2  -- ^ new ff2network architecture
    train lr arch inp = update arch grad
      where
        (grad, _) = gradBP2 (loss .: ff2network' lr) arch inp

    infer
      :: FF2Network 4 6 2  -- ^ ff2network architecture
      -> Tensor '[4]       -- ^ input
      -> Tensor '[2]       -- ^ output
    infer = postproc .: evalBP2 (ff2network' undefined)

    ff2network'
      :: Reifies s W
      => Double
      -> BVar s (FF2Network 4 6 2)  -- ^ ff2network architecture
      -> BVar s (Tensor '[4])       -- ^ input
      -> BVar s (Tensor '[2])       -- ^ output
    ff2network' = ff2network finalLayer


xavierPurityCheck :: forall i o . (KnownDim i, KnownDim o) => Linear i o -> Spec -> Spec
xavierPurityCheck ll tests =
  it (header ++ "initializes with xavier correctly")
    (  weights ll =##= 1/i
    >> bias    ll =##= 1/o)
  >> tests
  >> it (header ++ "leaves weights unchanged") (weights ll =##= 1/i)
  >> it (header ++ "leaves bias unchanged")    (bias    ll =##= 1/o)
 where
  header :: String
  header = "[ref-check] " ++ unwords ["Linear", show (truncate i), show (truncate o)] ++ ": "

  i, o :: Double
  i = fromIntegral (dimVal (dim :: Dim i))
  o = fromIntegral (dimVal (dim :: Dim o))


_lapproximately :: ([Double] -> Bool) -> Double -> Tensor d -> [Double] -> IO ()
_lapproximately pred e o dist = tensordata o >>= \os ->
  zipWith (Prelude.abs .: (-)) os dist `shouldSatisfy` pred

_approximately :: ([Double] -> Bool) -> Double -> Tensor d -> Tensor d -> IO ()
_approximately pred e o dist = tensordata dist >>= _lapproximately pred e o

approximately  e = _approximately  (all (< e)) e
notCloseTo     e = _approximately  (all (> e)) e
lapproximately e = _lapproximately (all (< e)) e
lnotCloseTo    e = _lapproximately (all (> e)) e

elementsSatisfy :: Tensor d -> ([Double] -> Bool) -> IO ()
elementsSatisfy o pred = tensordata o >>= (`shouldSatisfy` pred)

replicateN :: Int -> [a] -> [a]
replicateN inner = concatMap (replicate inner)

(=##=) :: Tensor d -> Double -> IO ()
(=##=) o v = elementsSatisfy o (all (== v))

infixl 2 =##=


