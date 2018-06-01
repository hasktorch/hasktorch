{-# LANGUAGE TypeApplications #-}
module Train where

import Text.Printf
import Control.Monad.IO.Class

import Torch.Double as Math

import LeNet
import DataLoader
-- import Numeric.Opto

main :: IO ()
main = do
  net0 <- newLeNet
  undefined

epoch :: Int -> IO ()
epoch e = do
  liftIO $ printf "[Epoch %d]\n" e

testNet :: [(Tensor '[3, 32, 32], Tensor '[10])] -> LeNet -> Double
testNet xs n = sum (map (uncurry test) xs) / fromIntegral (length xs)
  where
    test x (H.extract->t)
        | HM.maxIndex t == HM.maxIndex (H.extract r) = 1
        | otherwise                                  = 0
      where
        r = evalBP (`runNet` constVar x) n

loss :: (Tensor '[2,N], Tensor '[N]) -> Tensor '[1, 2] -> IO Precision
loss (x, y) param = do
  x' <- (y -) <$> resizeAs (param !*! x)
  (realToFrac . Math.sumall) <$> Math.square x'


gradient
  :: forall n . (KnownNatDim n)
  => (Tensor '[2, n], Tensor '[n]) -> Tensor '[1, 2] -> IO (Tensor '[1, 2])
gradient (x, y) param = do
  y' :: Tensor '[1, n] <- resizeAs y
  x' :: Tensor '[n, 2] <- newTranspose2d x
  m  :: Tensor '[1, 2] <- resizeAs (err y' !*! x')
  pure $ (-2 / nsamp) *^ m

  where
    err :: Tensor '[1, n] -> Tensor '[1, n]
    err y' = y' - (param !*! x)

    nsamp :: Precision
    nsamp = realToFrac (natVal (Proxy :: Proxy n))

gradientDescent
  :: (Tensor '[2, N], Tensor '[N])
  -> Precision
  -> Precision
  -> Tensor '[1, 2]
  -> IO [(Tensor '[1, 2], Precision, Tensor '[1, 2])]
gradientDescent (x, y) rate eps = go 0 []
 where
  go :: Int -> [(Tensor '[1, 2], Precision, Tensor '[1, 2])] -> Tensor '[1, 2] -> IO [(Tensor '[1, 2], Precision, Tensor '[1, 2])]
  go i res param = do
    g <- gradient (x, y) param
    diff <- (realToFrac . Math.sumall) <$> Math.abs g
    if diff < eps
    then pure res
    else do
      j <- loss (x, y) param
      let param' = param ^-^ (g ^* rate)
      go (i+1) ((param, j, g):res) param'

runN :: [(Tensor '[1, 2], Precision, Tensor '[1, 2])] -> Int -> IO (Tensor '[1,2])
runN lazyIters nIter = do
  let final = last $ take nIter lazyIters
  g <- Math.sumall <$> Math.abs (final ^. _3)
  let j = (^. _2) final
  let p = (^. _1) final
  putStrLn $ "Gradient magnitude after " <> show nIter <> " steps"
  print g
  putStrLn $ "Loss after " <> show nIter <> " steps"
  print j
  putStrLn $ "Parameter estimate after " <> show nIter <> " steps:"
  print p
  pure p

runExample :: IO (Tensor '[1,2])
runExample = do
  -- Generate data w/ ground truth params
  putStrLn "True parameters"
  let Just trueParam = fromList [3.5, -4.4]
  print trueParam

  dat <- genData trueParam

  -- Setup GD
  let Just (p0 :: Tensor '[1, 2]) = fromList [0, 0]
  iters <- gradientDescent dat 0.0005 0.0001 p0

  -- Results
  x <- runN iters (fromIntegral (natVal (Proxy :: Proxy N)))
  pure x




{-
main :: IO ()
main = MWC.withSystemRandom $ \g -> do
    flip evalStateT []
        . runConduit
        $ forM_ [0..] (\e -> liftIO (printf "[Epoch %d]\n" (e :: Int))
                          >> C.yieldMany train .| shuffling g
                      )
       .| C.iterM (modify . (:))      -- add to state stack for train eval
       .| runOptoConduit_
            (RO' Nothing Nothing)
            net0
            (adam @_ @(MutVar _ _) def
              (modelGradStoch crossEntropy noReg mnistNet g)
            )
       .| mapM_ (report 2500) [0..]
       .| C.sinkNull

  where
    report n b = do
          liftIO $ printf "(Batch %d)\n" (b :: Int)
          t0 <- liftIO getCurrentTime
          C.drop (n - 1)
          net' <- mapM (liftIO . evaluate . force) =<< await
          t1 <- liftIO getCurrentTime
          case net' of
            Nothing  -> liftIO $ putStrLn "Done!"
            Just net -> do
              chnk <- lift . state $ (,[])
              liftIO $ do
                printf "Trained on %d points in %s.\n"
                  (length chnk)
                  (show (t1 `diffUTCTime` t0))
                let trainScore = testModelAll maxIxTest mnistNet (J_I net) chnk
                    testScore  = testModelAll maxIxTest mnistNet (J_I net) test
                printf "Training error:   %.2f%%\n" ((1 - trainScore) * 100)
                printf "Validation error: %.2f%%\n" ((1 - testScore ) * 100)


-}
