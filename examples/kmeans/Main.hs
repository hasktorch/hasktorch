{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE RecordWildCards #-}

module Main where
import Control.Monad (foldM)
import Torch
import Torch.Functional as F hiding (take)
import qualified Torch.Functional.Internal as FI
import Prelude hiding (exp, take)
import qualified Prelude as P
import qualified Control.Monad.State.Strict as S
import Data.Maybe
import Control.Monad.IO.Class
import GHC.Float

import Plot (plot)

type StateIO = S.StateT (Maybe Generator) IO

trigger :: StateIO a -> IO a 
trigger m = flip S.evalStateT Nothing m


-- using broadcast mechanism to calculate pairwise ecludian distance of data
-- the input data is N*M matrix, where M is the dimension
-- we first expand the N*M matrix into N*1*M matrix A and 1*N*M matrix B
-- then a simple elementwise operation of A and B will handle the pairwise operation of points represented by data
pairwiseDistance :: Tensor -> Tensor -> Tensor
pairwiseDistance d1 d2 = res
  where
    a = F.unsqueeze (F.Dim 1) d1 
    b = F.unsqueeze (F.Dim 0) d2
    dis = F.pow (2.0 :: Float) (a - b)
    res = F.sumDim (F.Dim 2) RemoveDim Float dis

forgy :: Int -> Tensor -> StateIO Tensor
forgy nclusters x = do 
  let n = shape x !! 0
  inds <- randintTensor 0 n [nclusters]
  let nx = indexSelect 0 inds x
  return nx

lloyd :: Int -> Float -> Tensor -> StateIO (Tensor, Tensor)
lloyd nclusters tol x = do 
  centers <- forgy nclusters x 
  lloydStep nclusters tol x centers

lloydStep :: Int -> Float -> Tensor -> Tensor -> StateIO (Tensor, Tensor)
lloydStep nclusters tol x centers = do
  let 
    dis = pairwiseDistance x centers
    choiceCluster = F.argmin dis 1 False 
    ncenters = F.cat (F.Dim 0) $ map 
      (\index -> let
        selectedInd = F.squeezeDim 1 $ F.nonzero (choiceCluster ==. (asValue (fromIntegral index)))
        selected = indexSelect 0 selectedInd x
        n = (shape selected) !! 0
        retValue = F.meanDim (F.Dim 0) F.KeepDim Float selected
        in retValue
      ) [0 .. nclusters - 1]
    
    progress = asValue
      $ F.pow (2.0 :: Float) 
      $ F.sumAll 
      $ F.sqrt 
      $ F.sumDim (F.Dim 1) F.RemoveDim Float 
      $ F.pow (2.0 :: Float) (centers - ncenters)
  liftIO $ print $ "one step: " ++ show progress ++ " tol: " ++ show tol
  if progress < tol 
    then return (choiceCluster, ncenters)   -- we stop here
    else lloydStep nclusters tol x ncenters -- we continue the next iteration

initGen :: StateIO ()
initGen = do 
  let device = Device CPU 0
  gen <- liftIO $ mkGenerator device 0 
  S.put (Just gen)

randomTensorGen :: ([Int] -> Generator -> (Tensor, Generator)) -> [Int] -> StateIO Tensor 
randomTensorGen f dims = do 
  gen <- fromJust <$> S.get 
  let (t, ngen) = f dims gen
  S.put (Just ngen)
  return t

randnTensor :: [Int] -> StateIO Tensor 
randnTensor = randomTensorGen randn'

type High = Int 
type Low = Int
randintTensor :: Low -> High -> [Int] -> StateIO Tensor
randintTensor l h = randomTensorGen 
  (\dims gen -> randint l h dims (withDType Int64 defaultOpts) gen) 
  -- we need a LongTensor, thus (withDType Int64 defaultOpts) should be used

prog :: StateIO ()
prog = do 
  let nclusters = 3
  initGen
  [a1, a2, a3] <- mapM (\_ -> randnTensor [1000, 2]) (P.take 3 $ P.repeat 0)
  let 
    na2 = a2 + 3.0
    na3 = a3 + 6.0
    a = F.cat (F.Dim 0) [a1, na2, na3]
  (choiceCluster, centers) <- lloyd nclusters 1e-4 a
  let 
    xs = map float2Double $ ((asValue $ select 1 0 a) :: [Float])
    ys = map float2Double $ ((asValue $ select 1 1 a) :: [Float])
    cs = map (\x -> "c" ++ show x) $ (asValue choiceCluster :: [Int])
  liftIO $ plot xs ys cs

main :: IO ()
main = do
  trigger prog