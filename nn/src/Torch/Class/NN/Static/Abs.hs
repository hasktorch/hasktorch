module Torch.Class.NN.Static.Abs where

import Torch.Class.Tensor.Static
import Torch.Dimensions
import Control.Monad

class IsTensor t => Abs (t :: [Nat] -> *) where
  _abs_updateOutput
    :: t d   -- ^ input
    -> t d   -- ^ output
    -> IO ()
  _abs_updateGradInput
    :: (Product d ~ Product d')
    => t d   -- ^ input
    -> t d'  -- ^ gradOutput
    -> t d   -- ^ gradInput
    -> IO ()

abs_updateOutput :: Abs t => t d -> IO (t d)
abs_updateOutput i = empty >>= \o -> _abs_updateOutput i o >> pure o

abs_updateGradInput :: (Product d ~ Product d', Abs t) => t d -> t d' -> IO (t d)
abs_updateGradInput i gout = empty >>= \gin -> _abs_updateGradInput i gout gin >> pure gin

