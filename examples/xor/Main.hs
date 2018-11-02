{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TupleSections #-}
module Main where

-- Other modules
import Dense3 (linearBatchIO, reluIO)

-- Dependencies
import Text.Printf (printf)
import Control.Monad (replicateM, when)
import Control.Monad.ST (ST, runST)
import Data.Maybe (fromJust)
import System.IO (hFlush, stdout)
import System.Random.MWC (GenST)
import qualified System.Random.MWC as MWC (toSeed, Seed, save, restore, uniformR)
import qualified Data.Vector as V (fromList)
import Lens.Micro

-- Torch deps
import Torch.Double
  ( Tensor, HsReal                    -- Torch.Sig.Types
  , unsafeMatrix, unsafeVector, get1d -- Torch.Indef.Static.Tensor
  , constant                          -- Torch.Indef.Static.Tensor.Math
  , (^*)                              -- Torch.Indef.Static.Pairwise
  , uniform, ord2Tuple                -- Torch.Indef.Math.Random.TH
  , positive, positiveValue, Positive -- Torch.Indef.Math.Random.TH
  , manualSeed, newRNG                -- Torch.Core.Random
  , eqTensor                          -- Torch.Indef.Static.CompareT
  , allOf                             -- Torch.Indef.Mask
  , mSECriterionIO                    -- Torch.Indef.Static.NN.Criterion
  )
import Torch.Double.NN.Linear (Linear(Linear), getTensors)

type MLP2 i h o = (Linear i h, Linear h o)

type XORArch = MLP2 2 2 1

main :: IO ()
main = do
  section "Deterministic Inference" deterministic
  section "Stochastic Training"     stochastic

  where
    section title action = do
      putStrLn ""
      putStrLn $ replicate 20 '='
      putStrLn title
      putStrLn $ replicate 20 '='
      putStrLn ""
      action

    deterministic = do
      (xs, ys) <- mkExactData
      net0 <- mkExact
      putStrLn "Inferring with the exact XOR function. True values:"
      print ys
      (ys', _) <- xorForward net0 xs
      putStrLn "Inferred values:"
      print ys'
      printf "All values are identical? %s\n" (show . allOf $ eqTensor ys ys')
      (l, _) <- mSECriterionIO ys ys'
      printf "Mean-squared error: %s\n" (show (get1d l 0))

    seed0 = MWC.toSeed . V.fromList $ [0..256]

    stochastic = do
      net0 <- mkUniform
      let Just lr = positive 0.2
      (_, _, net) <-
        trainer lr 10000 (0, seed0, net0)

      (xs, ys) <- mkExactData
      (ys', _) <- xorForward net xs
      putStrLn "\nInferred values:"
      print ys'

      (l, _) <- mSECriterionIO ys ys'
      printf "Mean-squared error: %s\n" (show (get1d l 0))

    trainer lr n (c, s, net)
      | c >= n = pure (c, s, net)
      | otherwise = do
        (s', xs, ys) <- xorBatch s
        (o, grad) <- backward net ys xs
        trainer lr n (c+1, s', update net (lr, grad))


-- Forward + AD
xorForward
  :: XORArch
  -> Tensor '[4, 2]  -- ^ input
  -> IO (Tensor '[4, 1], Tensor '[4, 1] -> IO (XORArch, Tensor '[4, 2]))
xorForward (l1, l2) i = do
  (fc1out,    getl1gin) <- linearBatchIO l1 i
  (reluout, getrelugin) <- reluIO fc1out
  (fc2out,    getl2gin) <- linearBatchIO l2 reluout

  pure (fc2out, \gout -> do
    (l2, gin2) <- getl2gin gout
    (l1, gin1) <- getl1gin =<< getrelugin gin2
    pure ((l1, l2), gin1))

-- Forward + AD composed with the loss calculation
backward :: XORArch -> Tensor '[4, 1] -> Tensor '[4, 2] -> IO (Tensor '[1], XORArch)
backward net ys xs = do
  (out, getArchGrad) <- xorForward net xs
  (loss, getLossGrad) <- mSECriterionIO ys out

  printf "\rloss: %f" (fromJust $ get1d loss 0)
  hFlush stdout

  gnet <- fmap fst . getArchGrad =<< getLossGrad loss
  pure (loss, gnet)

-- Simple way to update a network with a multiple of the gradient
update :: XORArch -> (Positive HsReal, XORArch) -> XORArch
update (l1, l2) (plr, (g1, g2)) = (l1 - (g1 ^* lr), l2 - (g2 ^* lr))
  where
    lr = positiveValue plr

-- ========================================================================= --
-- Initialize an architecture
-- ========================================================================= --

-- make the exact solution to this architecture
mkExact :: IO XORArch
mkExact = do
  b1 <- unsafeVector [0, -1]
  w2 <- unsafeMatrix [[1], [-2]]
  pure (Linear (constant 1, b1), Linear (w2, constant 0))

-- make the stochastic initial weights for this architecture
--
-- FIXME: may require tuning
mkUniform :: IO XORArch
mkUniform = do
  g <- newRNG
  manualSeed g 1
  let Just rg = ord2Tuple (0, 1)
  l1 <- fmap Linear $ (,) <$> uniform g rg <*> uniform g rg
  l2 <- fmap Linear $ (,) <$> uniform g rg <*> uniform g rg
  pure (l1, l2)

-- ========================================================================= --
-- Build datasets
-- ========================================================================= --

-- make an exact datapoint of all 4 examples of XOR
mkExactData :: IO (Tensor '[4, 2], Tensor '[4, 1])
mkExactData =
  (,)
  <$> unsafeMatrix
    [ [0, 0]
    , [0, 1]
    , [1, 0]
    , [1, 1]
    ]
  <*>  unsafeMatrix
    [ [0]
    , [1]
    , [1]
    , [0]
    ]

-- make a trainable batch of XOR samples (with uniform distribution of the 4 examples)
xorBatch
  :: MWC.Seed
  -> IO (MWC.Seed, Tensor '[4, 2], Tensor '[4, 1])
xorBatch s =
  (s',,) <$> unsafeMatrix xs <*> unsafeMatrix ys
 where
  (s', xs, ys) = runST $ do
    g <- MWC.restore s
    (xs, ys) <- xorLists g 4
    (, xs, ys) <$> MWC.save g

  -- Make n-number of XOR examples in ST
  xorLists :: GenST s -> Int -> ST s ([[HsReal]], [[HsReal]])
  xorLists g n =
    foldl
      (\memo ((x0, x1), y) -> ([x0,x1]:fst memo, [y]:snd memo))
      ([], [])
    <$>
      replicateM n (mkRandomXOR g)

  -- make a uniform random datapoint of XOR
  mkRandomXOR :: GenST s -> ST s ((HsReal, HsReal), HsReal)
  mkRandomXOR g = do
    x0 <- MWC.uniformR (0, 1) g
    x1 <- MWC.uniformR (0, 1) g
    pure ((fromIntegral x0, fromIntegral x1), x0 `xor` x1)
   where
    xor :: Int -> Int -> HsReal
    xor a b
      = fromIntegral
      . fromEnum
      . odd
      $ a + b


