{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TupleSections #-}
module Main where

-- Other modules
import Dense3 (linearBatchIO, reluIO)
import Criterion (criterion)

-- Dependencies
import Text.Printf

-- Torch imports
import Torch.Double
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

mkExact :: IO XORArch
mkExact = do
  b1 <- unsafeVector [0, -1]
  w2 <- unsafeMatrix [[1], [-2]]
  pure (Linear (constant 1, b1), Linear (w2, constant 0))

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

main :: IO ()
main = do
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


