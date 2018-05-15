{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleContexts #-}
module MaxPooling where

import Numeric.Backprop
import Control.Monad.Trans
import Control.Monad.Trans.Maybe

import Torch.Double as Torch
import Conv2d
import qualified Torch.Long as Ix
import qualified Utils
import qualified Torch.Double.NN.Conv2d     as NN
import qualified Torch.Double.NN.Activation as NN


lenetLayer
  :: Reifies s W
  => Step2d D D
  -> Padding2d Pad Pad
  -> Double
  -> BVar s (Conv2d F O KW KH)
  -> BVar s (Tensor '[F, H, Wid])
  -> BVar s (Tensor '[2, 7, 15])
lenetLayer steps pad lr conv inp = relu (NN.conv2dMM steps pad lr conv inp)

main :: IO ()
main = Utils.section "Using Backpack" $ do
  g <- newRNG
  manualSeed g 1
  c :: Conv2d F O KW KH <- initConv2d g
  Utils.printFullConv2d "initial conv1d state" c

  -- do a forward pass
  Just (input :: Tensor InputDims) <- runMaybeT Utils.mkCosineTensor

  -- backprop manages our forward and backward passes
  let (o1, (c1', g1)) = backprop2 (lenetLayer steps pad 0.5) c input
  shape o1 >>= print
  -- shape g1 >>= print
  let ixs = longAsStatic (newIxDyn 0)
  (Ix.shape ixs) >>= print
  print "FOOOOOO"
  -- CEIL  Tensor '[2, 3, 5]
  -- FLOOR Tensor '[2, 3, 7]
  o1'                      <- spatialMaxPooling_updateOutput o1 ixs
            (Kernel2d :: Kernel2d 4 4) (Step2d :: Step2d 2 2)
            (Padding2d :: Padding2d 1 1) (sing :: SBool 'True)
  print "FOOOOOO"
  shape o1' >>= print
  (Ix.shape ixs) >>= print
  print o1'
  -- Utils.printFullConv2d "Unbatched convolution layer after backprop" c1'


--spatialMaxPooling_updateOutput
--  :: (SpatialDilationFloorCheckC iW iH oW oH kW kH pW pH dW dH 1 1, ceilMode ~ 'False)
--  => KnownNat inPlane
--  => Tensor      '[inPlane, iW, iH]       -- ^ input
--  -> IndexTensor '[inPlane, iW, iH]       -- ^ indices
--  -> Kernel2d kW kH                       -- ^ kernel size
--  -> Step2d dW dH                         -- ^ step size
--  -> Padding2d pW pH                      -- ^ padding size
--  -> Proxy ceilMode                       -- ^ ceil mode
--  -> IO (Tensor '[inPlane, oW, oH])      -- ^ output
--spatialMaxPooling_updateOutput = _spatialMaxPooling_updateOutput
--
--spatialMaxPooling_updateGradInput
--  :: (SpatialDilationFloorCheckC iW iH oW oH kW kH pW pH dW dH 1 1, ceilMode ~ 'False)
--  => Tensor      '[inPlane, iW, iH]         -- ^ input
--  -> Tensor      '[inPlane, oW, oH]         -- ^ gradOutput
--  -> IndexTensor '[inPlane, iW, iH]         -- ^ indices
--  -> Kernel2d kW kH                         -- ^ kernel size
--  -> Step2d dW dH                           -- ^ step size
--  -> Padding2d pW pH                        -- ^ padding size
--  -> Proxy ceilMode                         -- ^ ceil mode
--  -> IO (Tensor '[inPlane, iW, iH])         -- ^ gradInput
--spatialMaxPooling_updateGradInput = _spatialMaxPooling_updateGradInput


