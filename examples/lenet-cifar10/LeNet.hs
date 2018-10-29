{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE RankNTypes #-}
{- LANGUAGE Strict #-}
{-# LANGUAGE NoImplicitPrelude #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeSynonymInstances #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE CPP #-}
{- OPTIONS_GHC -fplugin GHC.TypeLits.Normalise #-}
{-# OPTIONS_GHC -fno-cse #-}
module LeNet
  ( LeNet
  -- , lenetBatch
  , lenetUpdate
  , lenetUpdate_
  , myupdate
  -- , Vision._conv1
  -- , Vision._conv2
  -- , y2cat
  -- , crossentropy

  -- Network-in-Network (bipass linear layer)
  ,   NIN
  , mkNIN
  , ninForwardBatch
  , ninBatch
  , ninBatchBP
  , ninUpdate

  -- * test
  , FC3Arch
  , mkFC3
  , ff3Batch

  -- * restart
  , maxPooling2dBatch'
  , conv2dMMBatch'
  ) where

import Prelude hiding (print, putStrLn)
import qualified Prelude as P
import Control.Exception.Safe (throwString)
import Control.Monad
import Data.IORef
import Data.Maybe
import Data.Function ((&))
import Data.Singletons.Prelude (Fst, Snd)
import Data.Singletons.Prelude.List (Product)
import Foreign hiding (new)
import Lens.Micro ((^.))
import System.IO (stdout, hFlush)
import System.IO.Unsafe
import GHC.TypeLits
import Control.Concurrent
import Data.Singletons
import Data.Singletons.Prelude.Bool
import Torch.Double hiding (logSoftMaxBatch, conv2dMMBatch)

import Criterion (criterion, crossEntropyIO)
import Torch.Double.NN.Linear (Linear(..))

import Torch.Data.Loaders.Cifar10

import qualified Torch.Double.NN           as NN
import qualified Torch.Double.NN.Linear    as Linear
import qualified Torch.Double.NN.Conv2d    as Conv2d
import qualified Torch.Long                as Long
import qualified Torch.Long.Dynamic        as LDynamic
import qualified Torch.Double              as Torch
import qualified Torch.Double.Storage      as Storage
import qualified Torch.Long.Storage        as LStorage
import qualified Torch.Models.Vision.LeNet as Vision

import qualified Torch.Double.Dynamic as Dynamic
import qualified Torch.Double.Dynamic.NN as Dynamic
import qualified Torch.Double.Dynamic.NN.Criterion as Dynamic
import qualified Torch.Double.Dynamic.NN.Activation as Dynamic

updateIORefWith :: IORef (Maybe (Tensor d)) -> Tensor d -> IO ()
updateIORefWith ref nxt =
  readIORef ref >>= \case
    Nothing  -> writeIORef ref (Just $ copy nxt)

    Just old ->
      finalizeForeignPtr oldfp >> writeIORef ref Nothing
     where
      oldfp = ctensor (asDynamic old)


-- FIXME: maybe this isn't working???
replaceIORefWith :: IORef (Tensor d) -> Tensor d -> IO ()
replaceIORefWith ref nxt = do
  old <- readIORef ref
  finalizeForeignPtr (ctensor (asDynamic old))
  writeIORef ref nxt

print :: Show a => a -> IO ()
print =
  -- P.print
  const (pure ())

putStrLn :: String -> IO ()
putStrLn =
  -- P.putStrLn
  const (pure ())

bsz = (dim :: Dim 4)
bs = (fromIntegral $ dimVal bsz) :: Int
type LeNet = Vision.LeNet 3 5

newLeNet :: Generator -> IO LeNet
newLeNet = Vision.newLeNet @3 @5

convin1Ref :: IORef (Maybe (Tensor '[4, 3, 32, 32]))
convin1Ref = unsafePerformIO $ newIORef Nothing
{-# NOINLINE convin1Ref #-}

convout1Ref :: IORef (Tensor '[4, 6])
convout1Ref = unsafePerformIO $ pure empty >>= newIORef
{-# NOINLINE convout1Ref #-}

columnsbuff1outRef :: IORef Dynamic
columnsbuff1outRef = unsafePerformIO $ pure Dynamic.empty >>= newIORef
{-# NOINLINE columnsbuff1outRef #-}

onesbuff1outRef :: IORef Dynamic
onesbuff1outRef = unsafePerformIO $ pure Dynamic.empty >>= newIORef
{-# NOINLINE onesbuff1outRef #-}


fcout1Ref :: IORef (Tensor '[4, 120])
fcout1Ref = unsafePerformIO $ newIORef (constant 0)
{-# NOINLINE fcout1Ref #-}

fcout2Ref :: IORef (Tensor '[4, 84])
fcout2Ref = unsafePerformIO $ newIORef (constant 0)
{-# NOINLINE fcout2Ref #-}


cnlloutRef :: IORef (Tensor '[1])
cnlloutRef = unsafePerformIO $ newIORef (constant 0)
{-# NOINLINE cnlloutRef #-}

cnllwRef :: IORef (Tensor '[1])
cnllwRef = unsafePerformIO $ newIORef (constant 1)
{-# NOINLINE cnllwRef #-}

cnllgRef :: IORef (Tensor '[4, 10])
cnllgRef = unsafePerformIO $ newIORef (constant 1)
{-# NOINLINE cnllgRef #-}

logsmgRef :: IORef ((Tensor '[4, 10]))
logsmgRef = unsafePerformIO $ newIORef (constant 0)
{-# NOINLINE logsmgRef #-}

counter :: IORef (Integer)
counter = unsafePerformIO $ newIORef 0
{-# NOINLINE counter #-}

printOn :: Show a => (Integer -> Bool) -> a -> IO ()
printOn cond x = do
  c <- readIORef counter
  if cond c
  then print x
  else pure ()

halt :: Integer -> String -> IO ()
halt mx msg = do
  modifyIORef counter (+1)
  c <- readIORef counter
  if c >= mx
  then throwString msg
  else pure ()


-- data NIN = NIN !(Conv2d 3 16 '(3, 3)) !(Conv2d 16 10 '(3, 3))
data NIN = NIN !(Conv2d 3 10 '(3, 3)) !(Conv2d 16 10 '(3, 3))
  deriving (Show)

mkNIN :: IO NIN
mkNIN = do
  g <- newRNG
  manualSeed g 1
  let Just rg = ord2Tuple (-1, 1)
  w0 <- uniform g rg
  b0 <- uniform g rg
  w1 <- uniform g rg
  b1 <- uniform g rg
  pure $ NIN (Conv2d (w0, b0)) (Conv2d (w1, b1))


ninForwardBatch net inp = fst <$> ninBatch undefined net inp

ninBatchBP
  :: HsReal
  -> NIN
  ->    (IndexTensor '[4])          -- ^ ys
  ->    (Tensor '[4, 3, 32, 32])    -- ^ xs
  -> IO (Tensor '[1], NIN)          -- ^ output and gradient
ninBatchBP = criterion crossEntropyIO ninBatch

ninBatch
  :: HsReal
  -> NIN
  ->    (Tensor '[4, 3, 32, 32])  -- ^ input
  -> IO (Tensor '[4, 10], Tensor '[4, 10] -> IO (NIN, Tensor '[4, 3, 32, 32]))
ninBatch lr (NIN conv1 conv2) i = do
  (conv1out :: Tensor '[4, 10, 30, 30], unconv1out)
    <- conv2dMMBatch'
      ((Step2d    @'(1,1)))
      ((Padding2d @'(0,0)))
      lr conv1 i

--  (mpout1 :: Tensor '[4, 16, 15, 15], getmpgrad)
--    <- maxPooling2dBatch'
--      ((Kernel2d  @'(2,2)))
--      ((Step2d    @'(2,2)))
--      ((Padding2d @'(0,0)))
--      ((sing      :: SBool 'True))
--      (conv1out)
--
--  (conv2out :: Tensor '[4, 10, 13, 13], unconv2out)
--    <- conv2dMMBatch'
--      ((Step2d    @'(1,1)))
--      ((Padding2d @'(0,0)))
--      lr conv2 mpout1
--  (gapout, getgapgrad) <- gapPool2dBatchIO conv2out

  (gapout, getgapgrad) <- gapPool2dBatchIO conv1out
  (smout, smgrads)       <- softMaxBatch gapout

  pure (smout, \gout -> do
    smgout <- smgrads gout
    -- (conv2g, gin1) <- (getgapgrad >=> unconv2out) smgout
    -- (conv1g, gin2) <- (getmpgrad >=> unconv1out) gin1
    -- pure (NIN conv1g conv2g, gin2))

    (conv1g, gin2) <- (getgapgrad >=> unconv1out) smgout
    pure (NIN conv1g conv2, gin2))

ninUpdate :: NIN -> (Positive HsReal, NIN) -> IO NIN
ninUpdate net@(NIN c1 c2) (plr, g@(NIN g1 g2)) = pure $ NIN (c1 + (g1 ^* lr)) (c2 + (g2 ^* lr))
 where
  lr = positiveValue plr


logsmoutRef :: IORef ((Tensor '[4, 10]))
logsmoutRef = unsafePerformIO $ newIORef (constant 0)
{-# NOINLINE logsmoutRef #-}


nullIx :: IORef Long.Dynamic
nullIx = unsafePerformIO $ do
  -- newForeignPtr nullPtr
  newIORef undefined
{-# NOINLINE nullIx #-}
-- ========================================================================= --
--
---- reluCONV2outRef :: IORef (Tensor '[4, 16, 2, 2])
--reluCONV2outRef = unsafePerformIO $ newIORef (constant 0)
--{-# NOINLINE reluCONV2outRef #-}
--
---- reluCONV2ginRef :: IORef (Tensor '[4, 16, 2, 2])
--reluCONV2ginRef = unsafePerformIO $ newIORef (constant 0)
--{-# NOINLINE reluCONV2ginRef #-}

-- ========================================================================= --

lenetUpdate_ :: LeNet -> (HsReal, LeNet) -> IO ()
lenetUpdate_ net (lr, g) = Vision.update_ net lr g

lenetUpdate :: LeNet -> (HsReal, LeNet) -> LeNet
lenetUpdate net (lr, g) = Vision.update net lr g

-- | update a LeNet network
myupdate :: LeNet -> (Positive HsReal, LeNet) -> IO LeNet
myupdate net (plr, grad) = do
  when verbose $ P.print $ Conv2d.weights (grad ^. Vision.conv1)
  when verbose $ P.print $ Conv2d.bias    (grad ^. Vision.conv1)
  when verbose $ P.print $ Conv2d.weights (grad ^. Vision.conv2)
  when verbose $ P.print $ Conv2d.bias    (grad ^. Vision.conv2)
  when verbose $ P.print $ Linear.weights (grad ^. Vision.fc1)
  when verbose $ P.print $ Linear.bias    (grad ^. Vision.fc1)
  when verbose $ P.print $ Linear.weights (grad ^. Vision.fc2)
  when verbose $ P.print $ Linear.bias    (grad ^. Vision.fc2)
  when verbose $ P.print $ Linear.weights (grad ^. Vision.fc3)
  when verbose $ P.print $ Linear.bias    (grad ^. Vision.fc3)
  -- throwString "x0"

  pure $ Vision.LeNet
    (Conv2d (conv1w', conv1b'))
    (Conv2d (conv2w', conv2b'))
    (Linear (fc1w', fc1b'))
    (Linear (fc2w', fc2b'))
    (Linear (fc3w', fc3b'))
 where
  verbose = False

  lr = positiveValue plr

  conv1w' = Conv2d.weights (net ^. Vision.conv1) - (Conv2d.weights (grad ^. Vision.conv1) ^* lr)
  conv1b' = Conv2d.bias    (net ^. Vision.conv1) - (Conv2d.bias    (grad ^. Vision.conv1) ^* lr)

  conv2w' = Conv2d.weights (net ^. Vision.conv2) - Conv2d.weights (grad ^. Vision.conv2) ^* lr
  conv2b' = Conv2d.bias    (net ^. Vision.conv2) - Conv2d.bias    (grad ^. Vision.conv2) ^* lr

  fc1w'   = Linear.weights (net ^. Vision.fc1)   - (Linear.weights (grad ^. Vision.fc1)   ^* lr)
  fc1b'   = Linear.bias    (net ^. Vision.fc1)   - (Linear.bias    (grad ^. Vision.fc1)   ^* lr)

  fc2w'   = Linear.weights (net ^. Vision.fc2)   - (Linear.weights (grad ^. Vision.fc2)   ^* lr)
  fc2b'   = Linear.bias    (net ^. Vision.fc2)   - (Linear.bias    (grad ^. Vision.fc2)   ^* lr)

  fc3w'   = Linear.weights (net ^. Vision.fc3)   - (Linear.weights (grad ^. Vision.fc3)   ^* lr)
  fc3b'   = Linear.bias    (net ^. Vision.fc3)   - (Linear.bias    (grad ^. Vision.fc3)   ^* lr)


classNLL
  :: IndexTensor '[4]

  -> (Tensor '[4, 10]                                       --  \___ these constitue a closed cartesian category and
  -> IO (Tensor '[1], Tensor '[1] -> IO (Tensor '[4, 10]))) --  /    can be abstracted away into an autodiff lib.
classNLL target inp = do
  (out, total_weight) <- (,) <$> readIORef cnlloutRef <*> readIORef cnllwRef
  -- let total_weight = constant 1  -- https://github.com/torch/nn/commit/3585e827eb65d071272a4aa4fab567b0b1eeee54#diff-1aa6a505cf16ad0e59498ada8432afb5
  onesLike_ total_weight total_weight

  updateOutput_ inp target szAvg Nothing ix reduce (out, total_weight)

  pure (out, \gout -> do
    gin <- readIORef cnllgRef
    zero_ gin
    -- updateGradInput_ gout out inp target gout szAvg Nothing total_weight ix reduce gin >> pure gin)
    updateGradInput_ inp target gout szAvg Nothing total_weight ix reduce gin
    pure gin)
  where
    ix = (-100)
    reduce = True
    szAvg = True

    updateOutput_
      :: Tensor '[sz, ps]            -- THTensor *input,
      -> IndexTensor '[sz]           -- THIndexTensor *target,
      -> Bool                        -- bool sizeAverage,
      -> Maybe (Tensor '[sz, ps])    -- THTensor *weights,
      -> Integer                     -- int64_t ignore_index,
      -> Bool                        -- bool reduce
      -> (Tensor '[1], Tensor '[1])  -- THTensor *input, total_weight
      -> IO ()
    updateOutput_ inp tar szAvg mws ix reduce (out, total_weight) = do
      Dynamic._ClassNLLCriterion_updateOutput (asDynamic inp) (Long.longAsDynamic tar) (asDynamic out)
        szAvg (asDynamic <$> mws) (asDynamic total_weight) ix reduce

    updateGradInput_
      :: Tensor '[sz, ps]          -- THTensor *input,
      -> IndexTensor '[sz]         -- THIndexTensor *target,
      -> Tensor '[1]               -- THTensor *gradOutput,
      -> Bool                      -- bool sizeAverage,
      -> Maybe (Tensor '[sz, ps])  -- THTensor *weights,
      -> Tensor '[1]               -- THTensor *total_weight,
      -> Integer                   -- int64_t ignore_index,
      -> Bool                      -- bool reduce

      -> Tensor '[sz, ps]          -- gradient to update inplace
      -> IO ()
    updateGradInput_ inp tar gout szAvg mws total_weight ix reduce gin =
      Dynamic._ClassNLLCriterion_updateGradInput (asDynamic inp) (Long.longAsDynamic tar) (asDynamic gout) (asDynamic gin)
        szAvg (asDynamic <$> mws) (asDynamic total_weight) ix reduce



type FC3Arch = (Linear (32*32*3) (32*32*3*2), Linear (32*32*3*2) (32*32), Linear (32*32) 10)
-- type FC3Arch = (Linear (32*32*3) 10)

mkFC3 :: IO FC3Arch
mkFC3 = do
  g <- newRNG
  manualSeed g 1
  let Just rg = ord2Tuple (-1, 1)
  w0 <- uniform g rg
  w1 <- uniform g rg
  w2 <- uniform g rg
  pure
    ( Linear (w0, constant 1)
    , Linear (w1, constant 1)
    , Linear (w2, constant 1)
    )


ff3Batch
  :: All KnownNat '[i,h0,h1] -- '[i] --
  => All KnownDim '[i,h0,h1] -- '[i] --
  => HsReal
  -> (Linear i h0, Linear h0 h1, Linear h1 10)
  -- -> (Linear i 10)
  ->    (Tensor '[4, i])  -- ^ input
  -> IO (Tensor '[4, 10], Tensor '[4, 10] -> IO ((Linear i h0, Linear h0 h1, Linear h1 10), Tensor '[4, i]))
  -- -> IO (Tensor '[4, 10], Tensor '[4, 10] -> IO ((Linear i 10), Tensor '[4, i]))
-- ff3Batch training lr (l1) i = do
ff3Batch lr (l1, l2, l3) i = do

  -- start fully connected network
#ifdef NONLINEAR
  (fc1out, fc1getgrad) <- linearBatch' 1 (constant 0) (constant 0) (Linear (constant 0, constant 0)) l1 i
  ( r1out, unrelu1)    <- relu''' fc1out
  (fc2out, fc2getgrad) <- linearBatch' 1 (constant 0) (constant 0) (Linear (constant 0, constant 0)) l2 r1out
  ( r2out, unrelu2)    <- relu''' fc2out
  (fc3out, fc3getgrad) <- linearBatch' 1 (constant 0) (constant 0) (Linear (constant 0, constant 0)) l3 r2out
  (fin, smgrads)       <- softMaxBatch fc3out
#else
  (fc1out, fc1getgrad) <- linearBatch' 1 (constant 0) (constant 0) (Linear (constant 0, constant 0)) l1 i
  let r1out = fc1out
  (fc2out, fc2getgrad) <- linearBatch' 1 (constant 0) (constant 0) (Linear (constant 0, constant 0)) l2 r1out
  let r2out = fc2out
  (fc3out, fc3getgrad) <- linearBatch' 1 (constant 0) (constant 0) (Linear (constant 0, constant 0)) l3 r2out
  (fin, smgrads)       <- softMaxBatch fc3out
#endif

  pure (fin, \gout -> do
    smg <- smgrads gout
    -- P.print gout

    (fc3g, fc3gin) <- fc3getgrad gout
    -- P.print (Linear.weights fc3g)
#ifdef NONLINEAR
    (fc2g, fc2gin) <- fc2getgrad =<< unrelu2 fc3gin
    (fc1g, fc1gin) <- fc1getgrad =<< unrelu1 fc2gin
#else
    (fc2g, fc2gin) <- fc2getgrad fc3gin
    (fc1g, fc1gin) <- fc1getgrad fc2gin
#endif
    pure ((fc1g, fc2g, fc3g), fc1gin))

    -- smg <- smgrads gout
    -- P.print gout
    -- (fc1g, fc1gin) <- fc1getgrad gout
    -- pure ((fc1g), fc1gin))





-- mp1ixRef = unsafePerformIO $ newIORef (Long.constant 0)
-- {-# NOINLINE mp1ixRef #-}
-- mp1outRef = unsafePerformIO $ newIORef (constant 0)
-- {-# NOINLINE mp1outRef #-}
-- mp1ginRef = unsafePerformIO $ newIORef (constant 0)
-- {-# NOINLINE mp1ginRef #-}

-- conv1ginbuffRef     = unsafePerformIO $ pure new >>= newIORef -- (Tensor '[b,f,h,w])     -- grad input buffer
-- {-# NOINLINE conv1ginbuffRef #-}
-- conv1columnsbuffRef = unsafePerformIO $ pure new >>= newIORef -- (Tensor '[])            -- columns buffer
-- {-# NOINLINE conv1columnsbuffRef #-}
-- conv1onesbuffRef    = unsafePerformIO $ pure new >>= newIORef -- (Tensor '[])            -- ones buffer
-- {-# NOINLINE conv1onesbuffRef #-}
-- conv1outRef         = unsafePerformIO $ pure new >>= newIORef -- (Tensor '[b, o,oH,oW])  -- output
-- {-# NOINLINE conv1outRef #-}
-- conv1iRef           = unsafePerformIO $ pure new >>= newIORef -- (Tensor '[b,f,h,w])     -- input
-- {-# NOINLINE conv1iRef #-}
-- conv1ginRef         = unsafePerformIO $ pure new >>= newIORef -- (Tensor '[b,f,h,w])     -- gradient input
-- {-# NOINLINE conv1ginRef #-}
-- conv1gparamsRef     = unsafePerformIO $ pure (Conv2d (new, new)) >>= newIORef  -- (Conv1d f o '(kH, kW))  -- gradient params
-- {-# NOINLINE conv1gparamsRef #-}


reluCONV1outRef :: Dimensions d => IORef (Tensor d)
reluCONV1outRef = unsafePerformIO $ newIORef (constant 0)
{-# NOINLINE reluCONV1outRef #-}

reluCONV1ginRef :: Dimensions d => IORef (Tensor d)
reluCONV1ginRef = unsafePerformIO $ newIORef (constant 0)
{-# NOINLINE reluCONV1ginRef #-}

---- conv2ginbuffRef     = unsafePerformIO $ pure new >>= newIORef -- (Tensor '[b,f,h,w])     -- grad input buffer
---- {-# NOINLINE conv2ginbuffRef #-}
---- conv2columnsbuffRef = unsafePerformIO $ pure new >>= newIORef -- (Tensor '[])            -- columns buffer
---- {-# NOINLINE conv2columnsbuffRef #-}
---- conv2onesbuffRef    = unsafePerformIO $ pure new >>= newIORef -- (Tensor '[])            -- ones buffer
---- {-# NOINLINE conv2onesbuffRef #-}
--conv2outRef         = unsafePerformIO $ pure new >>= newIORef -- (Tensor '[b, o,oH,oW])  -- output
--{-# NOINLINE conv2outRef #-}
--conv2iRef           = unsafePerformIO $ pure new >>= newIORef -- (Tensor '[b,f,h,w])     -- input
--{-# NOINLINE conv2iRef #-}
--conv2ginRef         = unsafePerformIO $ pure new >>= newIORef -- (Tensor '[b,f,h,w])     -- gradient input
--{-# NOINLINE conv2ginRef #-}
--conv2gparamsRef     = unsafePerformIO $ pure (Conv2d (new, new)) >>= newIORef  -- (Conv2d f o '(kH, kW))  -- gradient params
--{-# NOINLINE conv2gparamsRef #-}




nullFP :: IO (ForeignPtr a)
nullFP = newForeignPtr nullFunPtr nullPtr

-- | Backprop convolution function with batching
conv2dMMBatch'
  :: forall f h w kH kW dH dW pH pW oW oH s o b
  .  SpatialConvolutionC f h w kH kW dH dW pH pW oH oW
  => All KnownDim '[f,o,b,kW*kH*f,oH*oW]
  => Step2d '(dH,dW)                -- ^ step of the convolution in width and height dimensions.
  -> Padding2d '(pH,pW)             -- ^ zero padding to the input plane for width and height.
  -> Double                      -- ^ learning rate
  -> (Conv2d f o '(kH,kW))   -- ^ conv2d state
  -> (Tensor '[b,f,h,w])    -- ^ input: f stands for "features" or "input plane")
  -> IO (Tensor '[b, o,oH,oW], (Tensor '[b,o,oH,oW] -> IO (Conv2d f o '(kH,kW), Tensor '[b,f,h,w])))
conv2dMMBatch' = conv2dMMIO
  (constant 0 :: Tensor '[b,f,h,w])
  (constant 0 :: Tensor '[b,kW*kH*f,oH*oW])
  (constant 0 :: Tensor '[b,kW*kH*f,oH*oW])
  (constant 0)
  (constant 0)
  (Conv2d (constant 0, constant 0))

conv2dMMIO
  :: forall din dout fgin f o kH kW dH dW pH pW inBuff
  .  All Dimensions '[din,dout,fgin, inBuff]
  => All KnownDim '[f,o,kH,kW,dH,dW,pH,pW]

  -- buffers
  => (Tensor fgin)            -- ^ grad input buffer
  -> (Tensor inBuff)            -- ^ columns buffer
  -> (Tensor inBuff)            -- ^ ones buffer

  -- cacheables
  -> (Tensor dout)            -- output
  -> (Tensor din)             -- gradient input
  -> (Conv2d f o '(kH, kW))   -- gradient params

  -> Step2d '(dH,dW)                -- ^ step of the convolution in width and height dimensions.
  -> Padding2d '(pH,pW)             -- ^ zero padding to the input plane for width and height.
  -> Double                      -- ^ learning rate

  -> (Conv2d f o '(kH,kW))   -- ^ conv2d state
  -> (Tensor din)    -- ^ input: f stands for "features" or "input plane")
  -> IO (Tensor dout, (Tensor dout -> IO (Conv2d f o '(kH,kW), Tensor din)))
conv2dMMIO
  ginbuffer finput fgradInput out gin gparams
  step pad _ conv inp = do

  zero_ out
  updateOutput_ finput fgradInput step pad conv inp out

  pure (copy out,
    \gout -> do
      let putStrLn' s = putStrLn $ "=AD= [conv2d] " ++ s
      zero_ gin
      zero_ (Conv2d.weights gparams)
      zero_ (Conv2d.bias gparams)

      -- let finput = constant 0 :: Tensor '[5,150,100]
      -- let fgradInput = constant 0 :: Tensor '[kH, kW]

      putStrLn' "updateGradInput_"
      updateGradInput_ inp gout gin conv finput fgradInput step pad

      putStrLn' "accGradParameters_"
      accGradParameters_ inp gout gparams finput fgradInput step pad
      print (Conv2d.weights gparams)
      putStrLn "ad passes!"

      pure (gparams, gin))
 where
  updateOutput_ finput fgradInput step pad conv inp out =
    Dynamic._spatialConvolutionMM_updateOutput
      (asDynamic inp)                      -- ^ input
      (asDynamic out)                      -- ^ output
      (asDynamic (Conv2d.weights conv))    -- ^ 3D weight tensor (connTable:size(1) x kH x kW)
      (asDynamic (Conv2d.bias conv))       -- ^ 1D bias tensor (nOutputPlane)

      (asDynamic finput)                  -- ^ BUFFER: temporary columns -- also called "finput"
      (asDynamic fgradInput)              -- ^ BUFFER: buffer of ones for bias accumulation  -- also called "fgradInput"

      (Conv2d.kernel2d conv)               -- ^ (kW, kH) kernel height and width
      (param2d step)                       -- ^ (dW, dH) step of the convolution in width and height dimensions. C-default is 1 for both.
      (param2d pad)                        -- ^ (pW, pH) zero padding to the input plane for width and height. (kW-1)/2 is often used. C-default is 0 for both.

  updateGradInput_ inp gout gin conv colsbuffer onesbuffer step pad = do
    -- _spatialConvolutionMM_updateGradInput inp gout gin w columns ones (kW, kH) (dW, dH) (pW, pH) = do
    Dynamic._spatialConvolutionMM_updateGradInput
      (asDynamic inp)                      -- ^ input
      (asDynamic gout)                     -- ^ gradOutput
      (asDynamic gin)                      -- ^ gradInput
      (asDynamic (Conv2d.weights conv))    -- ^ weight
      (asDynamic colsbuffer)               -- ^ columns
      (asDynamic onesbuffer)               -- ^ ones
      (Conv2d.kernel2d conv)               -- ^ (kW, kH) kernel height and width
      (param2d step)                       -- ^ (dW, dH) step of the convolution in width and height dimensions
      (param2d pad)                        -- ^ (pW, pH) zero padding to the input plane for width and height. (kW-1)/2 is often used.


  accGradParameters_ inp gout gconv columnsbuff onesbuff step pad = do
    Dynamic._spatialConvolutionMM_accGradParameters
      (asDynamic inp)    -- ^ input
      (asDynamic gout)    -- ^ gradOutput
      (asDynamic (Conv2d.weights gconv))    -- ^ gradWeight
      (asDynamic (Conv2d.bias gconv))    -- ^ gradBias
      (asDynamic columnsbuff)    -- ^ finput/columns <<- required. This can be NULL in C if gradWeight is NULL.
      (asDynamic onesbuff)   -- ^ ones
      (Conv2d.kernel2d conv) -- ^ (kW, kH) kernel height and width
      (param2d step)         -- ^ (dW, dH) step of the convolution in width and height dimensions
      (param2d pad)          -- ^ (pW, pH) zero padding to the input plane for width and height. (kW-1)/2 is often used.
      1



-------------------------------------------------------------------------------

mp2ixRef  :: IORef (IndexTensor '[4,16, 5, 5])
mp2ixRef = unsafePerformIO $ newIORef (Long.constant 0)
{-# NOINLINE mp2ixRef #-}

mp2outRef :: IORef (     Tensor '[4,16, 5, 5])
mp2outRef = unsafePerformIO $ newIORef (constant 0)
{-# NOINLINE mp2outRef #-}

mp2ginRef :: IORef (     Tensor '[4,16,10,10])
mp2ginRef = unsafePerformIO $ newIORef (constant 0)
{-# NOINLINE mp2ginRef #-}


-- | internal function of 'maxPooling2d' and 'maxPooling2dBatch'. Should not be used.
maxPooling2dIO
  :: forall d d' kH kW dH dW pH pW ceilMode
  .  All KnownDim '[kH,kW,pH,pW,dH,dW]
  => All Dimensions '[d',d]

  -- optional buffers
  => (IndexTensor d')
  -> (Tensor d')
  -> (Tensor d)

  -- Parameters
  -> Kernel2d '(kH, kW)         -- ^ kernel size
  -> Step2d '(dH, dW)           -- ^ step size. Note: default in C is the kernel size.
  -> Padding2d '(pH, pW)        -- ^ padding size
  -> SBool ceilMode         -- ^ ceil mode

  -- function arguments
  -> Tensor d
  -> IO (Tensor d', Tensor d' -> IO (Tensor d))
maxPooling2dIO ix out gin ker step pad ceil inp = do
  zero_ out
  Long.zero_ ix

  updateOutput_ inp ker step pad ceil (ix, out)
  pure (out, \gout -> do
    zero_ gin
    updateGradInput_ inp gout ix ker step pad ceil gin
    pure gin)

 where
  updateOutput_
    :: Tensor d              -- ^ input
    -> Kernel2d '(kH, kW)        -- ^ kernel size
    -> Step2d '(dH, dW)          -- ^ step size
    -> Padding2d '(pH, pW)       -- ^ padding size
    -> SBool ceilMode                         -- ^ ceil mode
    -> (IndexTensor d', Tensor d')           -- ^ output
    -> IO ()
  updateOutput_ inp ker step pad sceil (ix, out) = do
    Dynamic._spatialMaxPooling_updateOutput (asDynamic inp) (asDynamic out) (longAsDynamic ix)
      (param2d ker) (param2d step) (param2d pad) (fromSing sceil)

  updateGradInput_
    :: Tensor d              -- ^ input
    -> Tensor d'             -- ^ gradOutput
    -> IndexTensor d'        -- ^ indices
    -> Kernel2d '(kH, kW)        -- ^ kernel size
    -> Step2d '(dH, dW)          -- ^ step size
    -> Padding2d '(pH, pW)       -- ^ padding size
    -> SBool ceilMode        -- ^ ceil mode
    -> Tensor d              -- ^ gradInput
    -> IO ()
  updateGradInput_ inp gout ix ker step pad sceil gin =
    Dynamic._spatialMaxPooling_updateGradInput
      (asDynamic inp) (asDynamic gout) (asDynamic gin) (longAsDynamic ix)
      (param2d ker) (param2d step) (param2d pad) (fromSing sceil)

-- | backprop-aware @maxPooling2d@ function.
maxPooling2d'
  :: forall iH iW kH kW dH dW pH pW oW oH ceilMode inPlane
  .  (SpatialDilationC iH iW kH kW dH dW pH pW oW oH 1 1 ceilMode)
  => KnownDim inPlane

  -- optional buffers
  -- => (IndexTensor '[inPlane, oH, oW])
  -- -> (Tensor '[inPlane, oH, oW])
  -- -> (Tensor '[inPlane, iH, iW])

  -- Parameters
  => Kernel2d '(kH, kW)       -- ^ kernel size
  -> Step2d '(dH, dW)       -- ^ step size
  -> Padding2d '(pH, pW)       -- ^ padding size
  -> SBool ceilMode        -- ^ ceil mode

  -> (Tensor '[inPlane, iH, iW])
  -> IO (Tensor '[inPlane, oH, oW], Tensor '[inPlane, oH, oW] -> IO (Tensor '[inPlane, iH, iW]))
maxPooling2d' = maxPooling2dIO (Long.new :: IndexTensor '[inPlane, oH, oW]) (new :: Tensor '[inPlane, oH, oW]) (new :: Tensor '[inPlane, iH, iW])

-- | backprop-aware @maxPooling2d@ function with a batch dimension.
maxPooling2dBatch'
  :: forall iH iW kH kW dH dW pH pW oW oH ceilMode b inPlane
  .  (SpatialDilationC iH iW kH kW dH dW pH pW oW oH 1 1 ceilMode)
  => KnownDim inPlane
  => KnownDim b

  -- optional buffers
  -- => IORef (IndexTensor '[b, inPlane, oH, oW])
  -- -> IORef (Tensor '[b, inPlane, oH, oW])
  -- -> IORef (Tensor '[b, inPlane, iH, iW])

  -- Parameters
  => Kernel2d '(kH, kW)        -- ^ kernel size
  -> Step2d '(dH, dW)          -- ^ step size
  -> Padding2d '(pH, pW)       -- ^ padding size
  -> SBool ceilMode        -- ^ ceil mode

  -> (Tensor '[b, inPlane, iH, iW])
  -> IO (Tensor '[b, inPlane, oH, oW], Tensor '[b, inPlane, oH, oW] -> IO (Tensor '[b, inPlane, iH, iW]))
-- maxPooling2dBatch' = _maxPooling2d'
maxPooling2dBatch' = maxPooling2dIO (Long.new :: IndexTensor '[b, inPlane, oH, oW]) (new :: Tensor '[b, inPlane, oH, oW]) (new :: Tensor '[b, inPlane, iH, iW])


-- ========================================================================= --


-- | A backpropable 'flatten' operation with a batch dimension
flattenBPBatch_
  :: forall d bs . (All KnownDim '[Product d, bs], All Dimensions '[bs:+d, d])
  => Product (bs:+d) ~ Product '[bs, Product d]
  => Tensor (bs:+d)
  -> IO (Tensor '[bs, Product d], Tensor '[bs, Product d] -> IO (Tensor (bs:+d)))
flattenBPBatch_ i = do
  o <- pure $ resizeAs i
  pure (o, \gout -> pure $ resizeAs gout)


-- ========================================================================= --

-- relu1outRef :: IORef (Tensor '[4, 120])
-- relu1outRef = unsafePerformIO $ newIORef (constant 0)
-- {-# NOINLINE relu1outRef #-}
--
-- relu1ginRef :: IORef (Tensor '[4, 120])
-- relu1ginRef = unsafePerformIO $ newIORef (constant 0)
-- {-# NOINLINE relu1ginRef #-}
--
--
-- reluBP__
--   :: forall d . Dimensions d
--   => Tensor d -> (IORef (Tensor d), IORef (Tensor d)) -> Bool -> IO (Tensor d, Bool -> Tensor d -> IO (Tensor d))
-- reluBP__ inp (outref, ginref) inplace = do
--   out <- readIORef outref
--   Dynamic._threshold_updateOutput (asDynamic inp) (asDynamic out) 0 0 inplace
--   pure (out, \ginplace gout -> do
--     -- throwString "xxxx"
--     replaceIORefWith ginref (constant 0 :: Tensor d)
--     -- zero_ ginref
--
--     gin <- readIORef ginref
--
--     -- zero_ gin
--     -- print "GOUT>>>>>>>>>>>>>"
--     -- print (asDynamic gout)
--     -- print "<<<<<<<<<<<<<<<<<"
--     -- print "GIN>>>>>>>>>>>>>>"
--     -- print (asDynamic gin)
--     -- print "<<<<<<<<<<<<<<<<<"
--
--     Dynamic._threshold_updateGradInput
--         (asDynamic inp) (asDynamic gout) (asDynamic gin) 0 0 False
--
--     -- print "GOUT>>>>>>>>>>>>>"
--     -- print (asDynamic gout)
--     -- print "<<<<<<<<<<<<<<<<<"
--     -- print "GIN>>>>>>>>>>>>>>"
--     -- print (asDynamic gin)
--     -- print "<<<<<<<<<<<<<<<<<"
--
--     -- zero_ gin
--     -- print "GOUT>>>>>>>>>>>>>"
--     -- print (asDynamic gout)
--     -- print "<<<<<<<<<<<<<<<<<"
--     -- print "GIN>>>>>>>>>>>>>>"
--     -- print (asDynamic gin)
--     -- print "<<<<<<<<<<<<<<<<<"
--
--     -- throwString "x"
--     pure gin)
--
--
-- -- reluBP'
-- --   :: forall d . Dimensions d
-- --   => Tensor d
-- --   -> ((Tensor d), (Tensor d)) -> IO (Tensor d, Tensor d -> IO (Tensor d))
-- -- reluBP' inp (out, gin) = do
-- --   Dynamic._threshold_updateOutput (asDynamic inp) (asDynamic out) 0 0 False
-- --   pure (out, \gout -> do
-- --     Dynamic._threshold_updateGradInput
-- --         (asDynamic inp) (asDynamic gout) (asDynamic gout) 0 0 False
-- --     pure gout)
-- --
-- -- reluBP
-- --   :: forall d . Dimensions d
-- --   => Tensor d
-- --   -> (IORef (Tensor d), IORef (Tensor d)) -> IO (Tensor d, Tensor d -> IO (Tensor d))
-- -- reluBP inp (outref, ginref) = do
-- --   out <- (readIORef outref)
-- --   gin <- readIORef ginref
-- --   reluBP' inp (out, gin)

-- | ReLU activation function
relu''' :: Dimensions d => Tensor d -> IO (Tensor d, Tensor d -> IO (Tensor d))
relu''' = thresholdIO 0 0


-- | A mutating 'flatten' operation with batch
flattenBatchForward_
  :: (All KnownDim '[Product d, bs], Dimensions d)
  => Product (bs:+d) ~ Product '[bs, Product d]
  => (Tensor (bs:+d))
  -> IO (Tensor '[bs, Product d])
flattenBatchForward_ = _resizeDim

-- ========================================================================= --

-- | 'linear' with a batch dimension
linearBatch
  :: forall i o b
   . All KnownDim '[b,i,o]
  => HsReal
  -> IORef (Tensor '[b, o])       -- output buffer. currently mutable.
  -> IORef (Tensor '[b, i])       -- gradin buffer. currently mutable.
  -> IORef (Linear i o)           -- gradparam buffer. currently mutable.

  -> (Linear i o)
  -> (Tensor '[b, i])
  -> IO (Tensor '[b, o], Tensor '[b, o] -> IO ((Linear i o),  (Tensor '[b, i])))     --- by "simple autodifferentiation", I am seeing that this is a fork
linearBatch lr outbufferRef gradinRef gradparamRef l i = do
  out    <- readIORef outbufferRef
  gin    <- readIORef gradinRef
  gparam <- readIORef gradparamRef
  linearBatch' lr out gin gparam l i


-- | 'linear' with a batch dimension
linearBatch'
  :: forall i o b
   . All KnownDim '[b,i,o]
  => HsReal

  -> (Tensor '[b, o])       -- output buffer. currently mutable.
  -> (Tensor '[b, i])       -- gradin buffer. currently mutable.
  -> (Linear i o)           -- gradparam buffer. currently mutable.

  -> (Linear i o)
  -> (Tensor '[b, i])
  -> IO (Tensor '[b, o], Tensor '[b, o] -> IO ((Linear i o),  (Tensor '[b, i])))     --- by "simple autodifferentiation", I am seeing that this is a fork
linearBatch' lr outbuffer gradinbuf gradparambuf l i = do
{-
  zero_ outbuffer
  updateOutput_ l i outbuffer
  pure (outbuffer, \gout -> do
    zero_ gradinbuf
    updateGradInput_ i gout (Linear.weights l) gradinbuf

    zero_ (Linear.weights gradparambuf)
    zero_ (Linear.bias gradparambuf)
    accGradParameters_ i gout l gradparambuf

    pure (gradparambuf, gradinbuf))
  where
    updateOutput_ :: Linear i o -> Tensor '[b, i] -> Tensor '[b, o] -> IO ()
    updateOutput_ (Linear (w,b)) inp out = do
      addmm_ 0 out 1 inp w
      addr_  1 out 1 (constant 1) b

    updateGradInput_ :: Tensor '[b, i] -> Tensor '[b, o] -> Tensor '[i,o] -> Tensor '[b, i] -> IO ()
    updateGradInput_ i gout w gin = do
      addmm_ 0 gin 1 gout (transpose2d w)

    accGradParameters_ :: Tensor '[b,i] -> Tensor '[b, o] -> Linear i o -> Linear i o -> IO ()
    accGradParameters_ i gout (Linear (w, b)) (Linear (gw, gb)) = do
      -- print (shape gw, shape gb, shape gout)
      addmm_ 1 gw lr (transpose2d i)    gout
      addmv_ 1 gb lr (transpose2d gout) (constant 1)
-}
  pure (updateOutput i l, \gout -> do
    let g@(Linear (gw, gb)) = accGradParameters i gout l
    let gin = updateGradInput i gout (Linear.weights l)
    -- P.print gb
    pure (g, gin))
   where
    updateOutput :: Tensor '[b, i] -> Linear i o -> Tensor '[b, o]
    updateOutput i (Linear (w,b)) =
      let
        o = addmm 0 (constant 0) 1 i w
      in
        addr 1 o 1 (constant 1) b

    updateGradInput :: Tensor '[b, i] -> Tensor '[b, o] -> Tensor '[i,o] -> Tensor '[b, i]
    updateGradInput i gout w = addmm 0 (constant 0) 1 gout (transpose2d w)

    accGradParameters :: Tensor '[b,i] -> Tensor '[b,o] -> Linear i o -> Linear i o
    accGradParameters i gout (Linear (w, b)) = Linear (gw, gb) -- addr 1 (constant 0) lr i gout, cadd (constant 0) lr gout)
      where
        gw :: Tensor '[i, o]
        gw = addmm 1 (constant 0) lr (transpose2d i) gout

        gb :: Tensor '[o]
        gb = addmv 1 (constant 0) lr tgout (constant 1)

        tgout :: Tensor '[o,b]
        tgout = transpose2d gout


-- ========================================================================= --

-- | run a threshold function againts two BVar variables
logSoftMaxBatch
  :: Tensor '[4, 10]    -- ^ input
  -> IO (Tensor '[4, 10], Tensor '[4, 10] -> IO (Tensor '[4, 10]))   -- ^ output and gradient
logSoftMaxBatch = __logSoftMaxBatch False

-- | run a threshold function againts two BVar variables
__logSoftMaxBatch
  :: Bool
  -> Tensor '[4, 10]    -- ^ input
  -> IO (Tensor '[4, 10], Tensor '[4, 10] -> IO (Tensor '[4, 10]))   -- ^ output and gradient
__logSoftMaxBatch islossfn inp = do
  -- replaceIORefWith logsmoutRef (constant 0)
  -- out <- readIORef logsmoutRef
  let out = constant 0

  when islossfn $ do
    print "start updateOutput loss function"
    print out

  -- Dynamic._logSoftMax_updateOutput (asDynamic inp) (asDynamic out) i -- (if islossfn then 0 else 1)
  updateOutput_ inp i out

  when islossfn $ do
    print "stop updateOutput loss function"
    print out

  -- print "outty"
  -- print inp
  -- print "outty"

  -- print out

  pure (out, \gout -> do
    -- putStrLn ""
    -- putStrLn ",--------------------------------------------------Q.Q-------------------------------------------------."
    -- putStrLn "|                                                                                                      |"
    -- print gout

    let gin = constant 0
    -- replaceIORefWith logsmgRef (constant 0)
    -- -- updateIORefWith logsmgRef (constant 0)
    -- gin <- readIORef logsmgRef

    -- THIS FUNCTION LINKS THE OUTPUT TO THE GRADINPUT !?!?
--     zerosLike_ gin gin
--     zeros_ gin

    -- print gout
    -- print gin

    -- putStrLn "|                                                                                                      |"
    -- putStrLn "`------------------------------------------------------------------------------------------------------'"
    -- putStrLn ""
    -- throwString "Q.Q"
    -- print gout
    -- halt 2 ""

    updateGradInput_ inp gout out i gin
    pure gin
    )

 where
  i = (dim :: Dim 1)

  updateOutput_ :: Tensor '[4, 10] -> Dim 1 -> Tensor '[4, 10] -> IO ()
  updateOutput_ inp i out =
    Dynamic._logSoftMax_updateOutput (asDynamic inp) (asDynamic out) (fromIntegral $ dimVal i)

  updateGradInput_
    :: Tensor '[4, 10]  -- input
    -> Tensor '[4, 10]  -- gradOutput
    -> Tensor '[4, 10]  -- output
    -> Dim 1            -- dimension

    -> Tensor '[4, 10]  -- gradInput
    -> IO ()
  updateGradInput_ inp gout out i gin =
    Dynamic._logSoftMax_updateGradInput
      (asDynamic inp)             -- input
      (asDynamic gout)            -- gradOutput
      (asDynamic gin)             -- gradInput
      (asDynamic out)             -- output
      (fromIntegral $ dimVal i)   -- dimension



-- ========================================================================= --

-- | run a threshold function againts two BVar variables
softMaxBatch
  :: Tensor '[4, 10]    -- ^ input
  -> IO (Tensor '[4, 10], Tensor '[4, 10] -> IO (Tensor '[4, 10]))   -- ^ output and gradient
softMaxBatch = __softMaxBatch False

-- | run a threshold function againts two BVar variables
__softMaxBatch
  :: Bool
  -> Tensor '[4, 10]    -- ^ input
  -> IO (Tensor '[4, 10], Tensor '[4, 10] -> IO (Tensor '[4, 10]))   -- ^ output and gradient
__softMaxBatch islossfn inp = do
  -- replaceIORefWith logsmoutRef (constant 0)
  -- out <- readIORef logsmoutRef
  let out = constant 0

  when islossfn $ do
    print "start updateOutput loss function"
    print out

  -- Dynamic._logSoftMax_updateOutput (asDynamic inp) (asDynamic out) i -- (if islossfn then 0 else 1)
  updateOutput_ inp i out

  when islossfn $ do
    print "stop updateOutput loss function"
    print out
  pure (out, \gout -> do
    -- replaceIORefWith logsmgRef (constant 0)
    -- gin <- readIORef logsmgRef
    let gin = constant 0
    updateGradInput_ inp gout out i gin
    pure gin
    )

 where
  i = (dim :: Dim 1)

  updateOutput_ :: Tensor '[4, 10] -> Dim 1 -> Tensor '[4, 10] -> IO ()
  updateOutput_ inp i out =
    Dynamic._softMax_updateOutput (asDynamic inp) (asDynamic out) (fromIntegral $ dimVal i)

  updateGradInput_
    :: Tensor '[4, 10]  -- input
    -> Tensor '[4, 10]  -- gradOutput
    -> Tensor '[4, 10]  -- output
    -> Dim 1            -- dimension

    -> Tensor '[4, 10]  -- gradInput
    -> IO ()
  updateGradInput_ inp gout out i gin =
    Dynamic._softMax_updateGradInput
      (asDynamic inp)             -- input
      (asDynamic gout)            -- gradOutput
      (asDynamic gin)             -- gradInput
      (asDynamic out)             -- output
      (fromIntegral $ dimVal i)   -- dimension



