{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE NoStarIsType #-}

-- | A Child-Sum Tree-LSTM, showing where 'dimUp' and 'dimDown' earn their
-- keep.  (The example follows hasktorch-naperian's Tree-LSTM showcase.)
--
-- A tree has no fixed shape, so the recursion over it can never be a tensor
-- dimension — it stays plain Haskell.  Around that recursion, dimensions
-- cross the tensor\/structure border in both directions:
--
-- * 'dimUp': the four LSTM gates are computed with /one/ fused matrix
--   multiplication into a @'[Gates, Vector H]@ tensor; @dimUp@ lifts the
--   gate axis into a @Gates { gi, gf, go, gu }@ record so each gate gets its
--   own nonlinearity /by field name/ instead of by slice arithmetic.
--
-- * 'dimDown': the recursion hands back the children's states as Haskell
--   structure (a @Vector 2@ of tensors); @dimDown@ packs them into a
--   @'[Vector 2, Vector H]@ axis so all per-child forget gates and the
--   \(\sum_k f_k \odot c_k\) reduction run as bulk tensor math.
module Torch.Typed.TreeLSTMSpec (spec) where

import Data.Default.Class (Default)
import Data.Vector.Sized (Vector)
import qualified Data.Vector.Sized as V
import GHC.Generics (Generic)
import GHC.TypeLits
import Test.Hspec
import qualified Torch.DType as D
import qualified Torch.Device as D
import qualified Torch.Tensor as UT
import Torch.Typed hiding (length, replicate)
import Torch.Typed.Representable (dimDown, dimUp)
import Prelude hiding (tanh)

type H = 8

type Dev = '(D.CPU, 0)

type T shape = Tensor Dev 'D.Float shape

-- the LSTM gate axis, as a record instead of slice offsets
data Gates a = Gates
  { gi :: a, -- input gate
    gf :: a, -- forget gate (for the summed child state)
    go :: a, -- output gate
    gu :: a -- candidate update
  }
  deriving (Show, Eq, Generic, Default)

-- a tree with no fixed shape: this can never be a tensor dimension
data Tree = Leaf Int | Node Tree Tree

data Params = Params
  { emb :: T '[4, H], -- token embeddings (vocabulary of 4)
    uG :: T '[4 * H, H], -- fused gate weights, gates from the summed child state
    uF :: T '[H, H] -- per-child forget-gate weights
  }

-- | One node of the Child-Sum Tree-LSTM.  The recursion is ordinary Haskell;
-- the marked lines are where dimensions change sides.
treeLSTM :: Params -> Tree -> (T '[H], T '[H])
treeLSTM Params {..} (Leaf tok) =
  let e = UnsafeMkTensor (UT.select 0 tok (toDynamic emb)) :: T '[H]
   in (tanh e, e)
treeLSTM p@Params {..} (Node l r') =
  let children = V.fromTuple (treeLSTM p l, treeLSTM p r') :: V.Vector 2 (T '[H], T '[H])
      -- dimDown: children (Haskell structure) -> a tensor axis
      hs = toUnnamed (dimDown (fmap (fromUnnamed . fst) children :: Vector 2 (NamedTensor Dev 'D.Float '[Vector H]))) :: T '[2, H]
      cs = toUnnamed (dimDown (fmap (fromUnnamed . snd) children :: Vector 2 (NamedTensor Dev 'D.Float '[Vector H]))) :: T '[2, H]
      hsum = sumDim @0 hs :: T '[H]
      -- one fused matmul computes all four gates ...
      pre = uG `matmul` hsum :: T '[4 * H]
      -- ... and dimUp names the gate axis, so each gate can be treated
      -- differently by field instead of by slice offsets
      Gates {..} = dimUp (fromUnnamed (reshape @'[4, H] pre) :: NamedTensor Dev 'D.Float '[Gates, Vector H])
      i = sigmoid (toUnnamed gi)
      o = sigmoid (toUnnamed go)
      u = tanh (toUnnamed gu)
      -- per-child forget gates, batched over the packed child axis
      fs = sigmoid (hs `matmul` uF + expand @'[2, H] False (toUnnamed gf)) :: T '[2, H]
      c = i * u + sumDim @0 (fs * cs)
   in (o * tanh c, c)

params :: Params
params =
  Params
    { emb = UnsafeMkTensor (UT.reshape [4, 8] (UT.asTensor (map (/ 32) [0 .. 31 :: Float]))),
      uG = UnsafeMkTensor (UT.reshape [32, 8] (UT.asTensor (map (\x -> Prelude.sin x / 8) [0 .. 255 :: Float]))),
      uF = UnsafeMkTensor (UT.reshape [8, 8] (UT.asTensor (map (\x -> Prelude.cos x / 8) [0 .. 63 :: Float])))
    }

elems :: T shape -> [Float]
elems = UT.asValue . UT.reshape [-1] . toDynamic

spec :: Spec
spec = describe "a Child-Sum Tree-LSTM via dimUp and dimDown" $ do
  it "runs over trees of different shapes" $ do
    let (h1, _) = treeLSTM params (Node (Leaf 0) (Leaf 1))
        (h2, _) = treeLSTM params (Node (Node (Leaf 0) (Leaf 1)) (Leaf 2))
    length (elems h1) `shouldBe` 8
    length (elems h2) `shouldBe` 8
    elems h1 `shouldNotBe` elems h2 -- the tree shape matters
  it "dimUp splits the fused gate axis in declaration order" $ do
    let pre = fromUnnamed (reshape @'[4, H] (UnsafeMkTensor (UT.asTensor [0 .. 31 :: Float]) :: T '[32])) :: NamedTensor Dev 'D.Float '[Gates, Vector H]
        Gates {..} = dimUp pre
    elems (toUnnamed gi) `shouldBe` [0 .. 7]
    elems (toUnnamed gf) `shouldBe` [8 .. 15]
    elems (toUnnamed go) `shouldBe` [16 .. 23]
    elems (toUnnamed gu) `shouldBe` [24 .. 31]
  it "dimDown packs children in order" $ do
    let ha = UnsafeMkTensor (UT.asTensor (replicate 8 (1 :: Float))) :: T '[H]
        hb = UnsafeMkTensor (UT.asTensor (replicate 8 (2 :: Float))) :: T '[H]
        packed = toUnnamed (dimDown (fmap fromUnnamed (V.fromTuple (ha, hb)) :: Vector 2 (NamedTensor Dev 'D.Float '[Vector H]))) :: T '[2, H]
    elems packed `shouldBe` (replicate 8 1 ++ replicate 8 2)
