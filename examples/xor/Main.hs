{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TupleSections #-}
module Main where

-- Other modules
import Dense3 (linearBatchIO, reluIO)
import Criterion (criterion)

-- Dependencies
import Text.Printf
import Control.Monad
import Control.Monad.ST
import System.Random.MWC (GenST)
import qualified System.Random.MWC as MWC
import qualified Data.Vector as V

-- Torch imports
import Torch.Double hiding (update)
import Torch.Double.NN.Linear (Linear(..))
import qualified Torch.Double.NN           as NN
import qualified Torch.Double.NN.Linear    as Linear
import qualified Torch.Long                as Long
import qualified Torch.Long.Dynamic        as LDynamic

type MLP2 i h o = (Linear i h, Linear h o)

type XORArch = MLP2 2 2 1

xorForward
  :: HsReal
  -> XORArch
  -> Tensor '[4, 2]  -- ^ input
  -> IO (Tensor '[4, 1], Tensor '[4, 1] -> IO (XORArch, Tensor '[4, 2]))
xorForward lr (l1, l2) i = do
  (fc1out,    getl1gin) <- linearBatchIO 1 l1 i
  (reluout, getrelugin) <- reluIO fc1out
  (fc2out,    getl2gin) <- linearBatchIO 1 l2 reluout

  pure (fc2out, \gout -> do
    (l2, gin2) <- getl2gin gout
    (l1, gin1) <- getl1gin =<< getrelugin gin2
    pure ((l1, l2), gin1))

loss :: HsReal -> XORArch -> Tensor '[4, 1] -> Tensor '[4, 2] -> IO (Tensor '[1], XORArch)
loss = criterion mSECriterionIO xorForward

update :: XORArch -> (HsReal, XORArch) -> XORArch
update (l1, l2) (lr, (g1, g2)) = (l1 + (g1 ^* lr), l2 + (g2 ^* lr))

mkExact :: IO XORArch
mkExact = do
  b1 <- unsafeVector [0, -1]
  w2 <- unsafeMatrix [[1], [-2]]
  pure (Linear (constant 1, b1), Linear (w2, constant 0))

mkUniform :: IO XORArch
mkUniform = do
  g <- newRNG
  manualSeed g 1
  let Just rg = ord2Tuple (-3, 3)
  l1 <- fmap Linear $ (,) <$> uniform g rg <*> uniform g rg
  l2 <- fmap Linear $ (,) <$> uniform g rg <*> uniform g rg
  pure (l1, l2)

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

-- the formatting is a little funky
xorLists :: GenST s -> Int -> ST s ([[HsReal]], [[HsReal]])
xorLists g n =
  foldl
    (\memo ((x0, x1), y) -> ([x0,x1]:fst memo, [y]:snd memo))
    ([], [])
  <$>
    replicateM n (mkRandomXOR g)

xorBatchList
  :: MWC.Seed -> Int -> (MWC.Seed, [[HsReal]], [[HsReal]])
xorBatchList s n = runST $ do
  g <- MWC.restore s
  (xs, ys) <- xorLists g 4
  s' <- MWC.save g
  pure (s', xs, ys)

xorBatch
  :: MWC.Seed -> IO (MWC.Seed, Tensor '[4, 2], Tensor '[4, 1])
xorBatch s =
  (s',,) <$> unsafeMatrix xs <*> unsafeMatrix ys
 where
  (s', xs, ys) = xorBatchList s 4


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
      putStrLn "computing exact XOR function. True values:"
      print ys
      (ys', _) <- xorForward undefined net0 xs
      putStrLn "Inferred values:"
      print ys'

      printf "All values are identical? %s\n" (show . allOf $ eqTensor ys ys')
      (l, _) <- loss undefined net0 ys xs
      printf "Mean-squared error: %s\n" (show (get1d l 0))

    seed0 = MWC.toSeed . V.fromList $ [0..256]

    stochastic = do
      net0 <- mkUniform
      (_, s, net) <-
        trainer 0.01 4 (0, seed0, net0)
      (_, xs, ys) <- xorBatch s
      (ys', _) <- xorForward undefined net xs
      putStrLn "Inferred values:"
      print ys'

      printf "All values are identical? %s\n" (show . allOf $ eqTensor ys ys')
      (l, _) <- loss undefined net0 ys xs
      printf "Mean-squared error: %s\n" (show (get1d l 0))

    trainer lr n (c, s, net)
      | c > n = pure (c, s, net)
      | otherwise = do
        (s', xs, ys) <- xorBatch s
        (_, grad) <- loss lr net ys xs
        trainer lr n (c+1, s', update net (lr, grad))



