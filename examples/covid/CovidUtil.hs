module CovidUtil where

import Torch as T

initializeEmbedding :: Int -> Tensor -> IO Tensor
initializeEmbedding embedDim t =
  randIO' [nUniq, embedDim]
  where 
    (uniqT, _, _) = (T.uniqueDim 0 True False False t)
    nUniq = shape uniqT !! 0

-- | Convert a series into a sparkline string (from clisparkline library)
series2sparkline :: RealFrac a => [a] -> String
series2sparkline vs =
  let maxv = if null vs then 0 else maximum vs
  in map (num2sparkchar maxv) vs
  where
    sparkchars = "_▁▂▃▄▅▆▇█" 
    num2sparkchar maxv curv =
      sparkchars !!
        (Prelude.floor $ (curv / maxv) * (fromIntegral (length sparkchars - 2)))
  

tensorSparkline :: Tensor -> IO ()
tensorSparkline t = putStrLn $ (series2sparkline (asValue t' :: [Float])) ++ (" | Max: " ++ show (asValue maxValue :: Float))
  where 
    t' = toDType Float t
    maxValue = T.max t'
