This tutorial is intended to familiarize a new user of haskell and deep learning
with the current state of haskell bindings to torch, a deep learning library.

That said, while this literate haskell file also serves as a to learn a bit more about NLP,which

\begin{code}
import Torch.Cuda

type Tensor = DoubleTensor
\end{code}


\begin{code}
tensorIntro = do
  v_data :: Tensor '[2]    <- fromList [1..2]
  print v_data
  v_data :: Tensor '[2, 3] <- fromList [1..2*3]
  print v_data
  v_data :: Tensor '[2, 3, 2]    <- fromList [1..2*3*2]
  print v_data
  v_data :: Tensor '[2, 3, 2, 3] <- fromList [1..2*3*2*3]
  print v_data

tensorEdgecases = do
  putStrLn "keep in mind:"
  v_data :: Tensor '[2, 3] <- fromList []
  print v_data
  v_data :: Tensor '[2] <- fromList [1..5]
  print v_data
\end{code}


\begin{code}
main :: IO ()
main = do
  tensorIntro
  tensorEdgecases
\end{code}
