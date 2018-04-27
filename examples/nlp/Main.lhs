\usepackage{hyperref}

This tutorial is intended to familiarize a new user of haskell and deep learning
with the current state of haskell bindings to torch, a deep learning library.

While this literate haskell file is an introduction to NLP, most of the content
is taken from Robert Guthrie's
\href{https://github.com/rguthrie3/DeepLearningForNLPInPytorch}{DeepLearningForNLPInPytorch}.

We start with our language extensions:

\begin{code}
{-# LANGUAGE DataKinds, ScopedTypeVariables #-}
\end{code}

And our imports:
\begin{code}
import Torch.Cuda
import qualified Torch.Core.Random as Random
import qualified Torch.Cuda as TH
\end{code}

At present, the hasktorch library doesn't export a default tensor type, so
it can be helpful to indicate what level of precision we want here.
\begin{code}
type Tensor = DoubleTensor
\end{code}

\begin{code}
tensorIntro = do
  v :: Tensor '[2]    <- fromList [1..2]
  print v
  m :: Tensor '[2, 3] <- fromList [1..2*3]
  print m
  t :: Tensor '[2, 3, 2]    <- fromList [1..2*3*2]
  print t
  r'4 :: Tensor '[2, 3, 2, 3] <- fromList [1..2*3*2*3]
  print r'4

tensorEdgecases = do
  putStrLn "keep in mind:"
  m :: Tensor '[2, 3] <- fromList []
  print m
  v :: Tensor '[2] <- fromList [1..5]
  print v
\end{code}

\section{Indexing}

\begin{code}
tensorIndexing = do
  v :: Tensor '[2]    <- fromList [1..2]
  print v
  print (v TH.!! 0 :: Tensor '[1])
  m :: Tensor '[2, 3] <- fromList [1..2*3]
  print m
  print (m TH.!! 1 :: Tensor '[3])
  t :: Tensor '[2, 3, 2]    <- fromList [1..2*3*2]
  print t
  print (t TH.!! 2  :: Tensor '[2,3])
  r'4 :: Tensor '[2, 3, 2, 3] <- fromList [1..2*3*2*3]
  print (r'4 TH.!! 0  :: Tensor '[3,2,3])
  print (r'4 TH.!! 1  :: Tensor '[2,2,3])
  print (r'4 TH.!! 2  :: Tensor '[2,3,3])
  print (r'4 TH.!! 3  :: Tensor '[2,3,2])
\end{code}

\begin{code}
main :: IO ()
main = do
  g <- Random.new
  Random.manualSeed g 1
  tensorIntro
  tensorEdgecases
  putStrLn ""
  putStrLn "INDEXING"
  putStrLn "================"
  tensorIndexing
\end{code}
