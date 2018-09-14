{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE TupleSections #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeSynonymInstances #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE TypeApplications #-}
{-# OPTIONS_GHC -fno-cse #-}
module LeNet
  ( newLeNet
  , bs, bsz, LeNet
  , lenetBatchForward
  , lenetBatchBP
  , lenetUpdate
  , Vision._conv1
  , Vision._conv2
  , y2cat
  ) where

import Control.Exception.Safe (throwString)
import Control.Monad
import Data.IORef
import Data.Function ((&))
import Data.Singletons.Prelude.List (Product)
import Foreign hiding (new)
import Lens.Micro.Platform ((^.))
import System.IO.Unsafe
import Data.Singletons
import Data.Singletons.Prelude.Bool
import Torch.Double hiding (logSoftMaxBatch)
import Torch.Double.NN hiding (logSoftMaxBatch)
import Torch.Double.NN.Linear hiding (linearBatch)

import Torch.Data.Loaders.Cifar10
import qualified Torch.Long as Long
import qualified Torch.Long.Dynamic as LDyn
import qualified Torch.Double.NN.Conv2d as Conv2d
import qualified Torch.Double as Torch
import qualified Torch.Double.Storage as CPUS
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


bsz = (dim :: Dim 4)
bs = (fromIntegral $ dimVal bsz) :: Int
type LeNet = Vision.LeNet 3 5

newLeNet :: IO LeNet
newLeNet = Vision.newLeNet @3 @5

convin1Ref :: IORef (Maybe (Tensor '[4, 3, 32, 32]))
convin1Ref = unsafePerformIO $ newIORef Nothing
{-# NOINLINE convin1Ref #-}

convout1Ref :: IORef (Tensor '[4, 6])
convout1Ref = unsafePerformIO $ empty >>= newIORef
{-# NOINLINE convout1Ref #-}

columnsbuff1outRef :: IORef Dynamic
columnsbuff1outRef = unsafePerformIO $ Dynamic.empty >>= newIORef
{-# NOINLINE columnsbuff1outRef #-}

onesbuff1outRef :: IORef Dynamic
onesbuff1outRef = unsafePerformIO $ Dynamic.empty >>= newIORef
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

logsmgRef :: IORef (Tensor '[4, 10])
logsmgRef = unsafePerformIO $ empty >>= newIORef
{-# NOINLINE logsmgRef #-}

logsmoutRef :: IORef (Tensor '[4, 10])
logsmoutRef = unsafePerformIO $ empty >>= newIORef
{-# NOINLINE logsmoutRef #-}


nullIx :: IORef Long.Dynamic
nullIx = unsafePerformIO $ do
  -- newForeignPtr nullPtr
  newIORef undefined
{-# NOINLINE nullIx #-}

-- ========================================================================= --

lenetUpdate :: LeNet -> (HsReal, LeNet) -> IO ()
lenetUpdate net (lr, g) = Vision.update net lr g


lenetBatchForward
  :: LeNet
  ->    (Tensor '[4, 3, 32, 32])  -- ^ input
  -> IO (Tensor '[4, 10])         -- ^ output
lenetBatchForward net inp = lenetBatch False net inp


lenetBatchBP
  :: LeNet                          -- ^ architecture
  ->    (IndexTensor '[4])          -- ^ ys
  ->    (Tensor '[4, 3, 32, 32])    -- ^ xs
  -> IO (Tensor '[1], LeNet)    -- ^ output and gradient
lenetBatchBP arch ys xs = do
  out <- lenetBatch True arch xs
  (loss, getCEgrad) <- crossentropy ys out
  cegrad <- getCEgrad loss
  print (fromEnum <$> y2cat out)
  print (ys)
  print loss
  print cegrad
  throwString "that's all, folks!"
  pure (loss, gnet)

crossentropy
  :: IndexTensor '[4]            -- THIndexTensor *target,
  -> Tensor '[4, 10]            -- THTensor *input,
  -> IO (Tensor '[1], Tensor '[1] -> IO (Tensor '[4, 10]))               -- THTensor *output, gradient
crossentropy ys inp = do
  (lout, getLSMGrad) <- logSoftMaxBatch inp
  (nllout, getNLLGrad) <- classNLL ys lout
  pure (nllout, getNLLGrad >=> getLSMGrad)


y2cat :: Tensor '[4, 10] -> [Category]
y2cat ys = map (toEnum . fromIntegral . (\i -> Long.get2d rez i 0)) [0..3]
  where
    rez :: LongTensor '[4, 1]
    (_, Just rez) = Torch.max ys (dim :: Dim 1) keep

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



lenetBatch
  :: Bool                         -- ^ if you should perform backprop as well
  -> LeNet
  ->    (Tensor '[4, 3, 32, 32])  -- ^ input
  -> IO (Tensor '[4, 10])  -- ^ output and gradient
lenetBatch training arch i = do
    -- LAYER 1
    when training $ convin1Ref `updateIORefWith` i

    let conv1 = arch ^. Vision.conv1

    convout1     <- readIORef convout1Ref
    columnsbuff1 <- readIORef columnsbuff1outRef
    onesbuff1    <- readIORef onesbuff1outRef

    Dynamic._spatialConvolutionMM_updateOutput (asDynamic i) (asDynamic convout1) (asDynamic (Conv2d.weights conv1)) (asDynamic (Conv2d.bias conv1))
      columnsbuff1 onesbuff1
      (kernel2d conv1)
      (param2d (Step2d    :: Step2d '(1,1)))
      (param2d (Padding2d :: Padding2d '(0,0)))

    -- relu forward
    (out, unrelu) <- reluBP__ convout1 (reluCONV1outRef, reluCONV1ginRef) False

    let maxpoolInput1 = out
    -- maxpoolOutput1 <- empty
    ix1 <- LDyn.empty

    Dynamic._spatialMaxPooling_updateOutput (asDynamic convout1) (asDynamic convout1) ix1
      (param2d  (Kernel2d  :: Kernel2d '(2,2)))
      (param2d  (Step2d    :: Step2d '(2,2)))
      (param2d  (Padding2d :: Padding2d '(0,0)))
      True -- (fromSing (sing      :: SBool 'True))

    -- print ix
    pure (asStatic (asDynamic convout1) :: Tensor '[4, 6, 5, 5])

  >>= \i -> do
    -- LAYER 1
    let conv2 = arch ^. Vision.conv2
    convout2     <- readIORef convout1Ref
    columnsbuff2 <- readIORef columnsbuff1outRef
    onesbuff2    <- readIORef onesbuff1outRef

    conv2ginbuffRef          -- IORef (Tensor '[b,f,h,w])     -- grad input buffer
    conv2columnsbuffRef     -- IORef (Tensor '[])            -- columns buffer
    conv2onesbuffRef        -- IORef (Tensor '[])            -- ones buffer
    conv2outRef             -- IORef (Tensor '[b, o,oH,oW])  -- output
    conv2iRef              -- IORef (Tensor '[b,f,h,w])     -- input
    conv2ginRef          -- IORef (Tensor '[b,f,h,w])     -- gradient input
    conv2gparamsRef    -- IORef (Conv2d f o '(kH, kW))  -- gradient params
    (convout2, _) <-
      conv2dMMBatch
        lr conv2 i

    (reluout2, unrelu) <- reluBP__ convout2 (reluCONV2outRef, reluCONV2ginRef) False

    (mpout2, getmpgrad) <- maxPooling2dBatch'
      mp2ixRef
      mp2outRef
      mp2ginRef
      ((Kernel2d  :: Kernel2d '(2,2)))
      ((Step2d    :: Step2d '(2,2)))
      ((Padding2d :: Padding2d '(0,0)))
      ((sing      :: SBool 'True))
      reluout2

    pure mpout2
  >>= \i -> fst <$> flattenBPBatch_ i
  >>= \(inp :: Tensor '[4, 400])-> do
    out <- fst <$> fc1BP (arch ^. Vision.fc1) inp
    pure out

  >>= \inp -> do
    out <- fst <$> fc2BP (arch ^. Vision.fc2) inp
    pure out

  >>= \inp -> do
    fst <$> fc3BP (arch ^. Vision.fc3) inp

-------------------------------------------------------------------------------
    let conv2 = arch ^. Vision.conv2
    convout2     <- readIORef convout1Ref
    columnsbuff2 <- readIORef columnsbuff1outRef
    onesbuff2    <- readIORef onesbuff1outRef

    Dynamic._spatialConvolutionMM_updateOutput (asDynamic i) (asDynamic convout2) (asDynamic (Conv2d.weights conv2)) (asDynamic (Conv2d.bias conv2))
      columnsbuff2 onesbuff2
      (kernel2d conv2)
      (param2d (Step2d    :: Step2d '(1,1)))
      (param2d (Padding2d :: Padding2d '(0,0)))



-- | Backprop convolution function with batching
conv2dMMBatch
  :: forall f h w kH kW dH dW pH pW oW oH s o b
  .  SpatialConvolutionC f h w kH kW dH dW pH pW oH oW
  => All KnownDim '[f,o,b]
  => IORef (Tensor '[b,f,h,w])            -- ^ grad input buffer

  -- buffers
  -> IORef (Tensor '[])            -- ^ columns buffer
  -> IORef (Tensor '[])            -- ^ ones buffer

  -- cacheables
  -> IORef (Tensor '[b, o,oH,oW])            -- output
  -> IORef (Tensor '[b,f,h,w])             -- input
  -> IORef (Tensor '[b,f,h,w])             -- gradient input
  -> IORef (Conv2d f o '(kH, kW))   -- gradient params

  -> Step2d '(dH,dW)                -- ^ step of the convolution in width and height dimensions.
  -> Padding2d '(pH,pW)             -- ^ zero padding to the input plane for width and height.
  -> Double                      -- ^ learning rate
  -> (Conv2d f o '(kH,kW))   -- ^ conv2d state
  -> (Tensor '[b,f,h,w])    -- ^ input: f stands for "features" or "input plane")
  -> IO (Tensor '[b, o,oH,oW], (Tensor '[b,o,oH,oW] -> IO (Conv2d f o '(kH,kW), Tensor '[b,f,h,w])))
conv2dMMBatch = conv2dMM__

conv2dMM__
  :: forall din dout fgin f o kH kW dH dW pH pW
  .  All Dimensions '[din,dout,fgin]
  => All KnownDim '[f,o,kH,kW,dH,dW,pH,pW]

  -- buffers
  => IORef (Tensor fgin)            -- ^ grad input buffer
  -> IORef (Tensor '[])            -- ^ columns buffer
  -> IORef (Tensor '[])            -- ^ ones buffer

  -- cacheables
  -> IORef (Tensor dout)            -- output
  -> IORef (Tensor din)             -- input
  -> IORef (Tensor din)             -- gradient input
  -> IORef (Conv2d f o '(kH, kW))   -- gradient params

  -> Step2d '(dH,dW)                -- ^ step of the convolution in width and height dimensions.
  -> Padding2d '(pH,pW)             -- ^ zero padding to the input plane for width and height.
  -> Double                      -- ^ learning rate

  -> (Conv2d f o '(kH,kW))   -- ^ conv2d state
  -> (Tensor din)    -- ^ input: f stands for "features" or "input plane")
  -> IO (Tensor dout, (Tensor dout -> IO (Conv2d f o '(kH,kW), Tensor din)))
conv2dMM__
  ginbufferRef columnsbuffref onesref outref inref ginref gparamsref
  step pad lr conv inp = do
  onesbuff <- readIORef onesref
  columnsbuff <- readIORef columnsbuffref

  out <- readIORef outref
  updateOutput_ columnsbuff onesbuff step pad conv inp out
  pure (out,
    \gout -> do
      ginbuffer <- readIORef ginbufferRef
      gin <- readIORef ginref
      gparams <- readIORef gparamsref
      accGradParameters_ inp gout gparams columnsbuff onesbuff step pad

      updateGradInput_ inp gout gin conv columnsbuff onesbuff step pad

      pure (gparams, gin))
 where
  updateOutput_ colbuff onesbuff step pad conv inp out =
    Dynamic._spatialConvolutionMM_updateOutput
      (asDynamic inp)                      -- ^ input
      (asDynamic out)                      -- ^ output
      (asDynamic (Conv2d.weights conv))    -- ^ 3D weight tensor (connTable:size(1) x kH x kW)
      (asDynamic (Conv2d.bias conv))       -- ^ 1D bias tensor (nOutputPlane)
      (asDynamic colbuff)                  -- ^ BUFFER: temporary columns
      (asDynamic onesbuff)                 -- ^ BUFFER: buffer of ones for bias accumulation
      (Conv2d.kernel2d conv)               -- ^ (kW, kH) kernel height and width
      (param2d step)                       -- ^ (dW, dH) step of the convolution in width and height dimensions. C-default is 1 for both.
      (param2d pad)                        -- ^ (pW, pH) zero padding to the input plane for width and height. (kW-1)/2 is often used. C-default is 0 for both.

  updateGradInput_ inp gout gin conv colsbuffer onesbuffer step pad = do
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
      lr



-------------------------------------------------------------------------------

mp2ixRef  :: IORef (IndexTensor '[4, 6, 1, 1])
mp2ixRef = unsafePerformIO $ newIORef (Long.constant 0)
{-# NOINLINE mp2ixRef #-}

mp2outRef :: IORef (     Tensor '[4, 6, 1, 1])
mp2outRef = unsafePerformIO $ newIORef (constant 0)
{-# NOINLINE mp2outRef #-}

mp2ginRef :: IORef (     Tensor '[4, 6, 2, 2])
mp2ginRef = unsafePerformIO $ newIORef (constant 0)
{-# NOINLINE mp2ginRef #-}


-- | internal function of 'maxPooling2d' and 'maxPooling2dBatch'. Should not be used.
_maxPooling2d'
  :: forall d d' kH kW dH dW pH pW ceilMode
  .  All KnownDim '[kH,kW,pH,pW,dH,dW]
  => All Dimensions '[d',d]

  -- optional buffers
  => IORef (IndexTensor d')
  -> IORef (Tensor d')
  -> IORef (Tensor d)

  -- Parameters
  -> Kernel2d '(kH, kW)         -- ^ kernel size
  -> Step2d '(dH, dW)           -- ^ step size. Note: default in C is the kernel size.
  -> Padding2d '(pH, pW)        -- ^ padding size
  -> SBool ceilMode         -- ^ ceil mode

  -- function arguments
  -> Tensor d
  -> IO (Tensor d', Tensor d' -> IO (Tensor d))
_maxPooling2d' ixref outref ginref ker step pad ceil inp = do
  ix <- readIORef ixref
  out <- readIORef outref

  updateOutput_ inp ker step pad ceil (ix, out)
  pure (out, \gout -> do
    gin <- readIORef ginref
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
  :: (SpatialDilationC iH iW kH kW dH dW pH pW oW oH 1 1 ceilMode)
  => KnownDim inPlane

  -- optional buffers
  => IORef (IndexTensor '[inPlane, oH, oW])
  -> IORef (Tensor '[inPlane, oH, oW])
  -> IORef (Tensor '[inPlane, iH, iW])

  -- Parameters
  -> Kernel2d '(kH, kW)       -- ^ kernel size
  -> Step2d '(dH, dW)       -- ^ step size
  -> Padding2d '(pH, pW)       -- ^ padding size
  -> SBool ceilMode        -- ^ ceil mode

  -> (Tensor '[inPlane, iH, iW])
  -> IO (Tensor '[inPlane, oH, oW], Tensor '[inPlane, oH, oW] -> IO (Tensor '[inPlane, iH, iW]))
maxPooling2d' = _maxPooling2d'

-- | backprop-aware @maxPooling2d@ function with a batch dimension.
maxPooling2dBatch'
  :: (SpatialDilationC iH iW kH kW dH dW pH pW oW oH 1 1 ceilMode)
  => KnownDim inPlane
  => KnownDim b

  -- optional buffers
  => IORef (IndexTensor '[b, inPlane, oH, oW])
  -> IORef (Tensor '[b, inPlane, oH, oW])
  -> IORef (Tensor '[b, inPlane, iH, iW])

  -- Parameters
  -> Kernel2d '(kH, kW)        -- ^ kernel size
  -> Step2d '(dH, dW)          -- ^ step size
  -> Padding2d '(pH, pW)       -- ^ padding size
  -> SBool ceilMode        -- ^ ceil mode

  -> (Tensor '[b, inPlane, iH, iW])
  -> IO (Tensor '[b, inPlane, oH, oW], Tensor '[b, inPlane, oH, oW] -> IO (Tensor '[b, inPlane, iH, iW]))
maxPooling2dBatch' = _maxPooling2d'


-- ========================================================================= --


-- | A backpropable 'flatten' operation with a batch dimension
flattenBPBatch_
  :: forall d bs . (All KnownDim '[Product d, bs], All Dimensions '[bs:+d, d])
  => Product (bs:+d) ~ Product '[bs, Product d]
  => Tensor (bs:+d)
  -> IO (Tensor '[bs, Product d], Tensor '[bs, Product d] -> IO (Tensor (bs:+d)))
flattenBPBatch_ i = resizeAs_ i >>= \o -> pure (o, resizeAs_)


-- ========================================================================= --

reluCONV1outRef :: IORef (Tensor '[4, 6])
reluCONV1outRef = unsafePerformIO $ newIORef (constant 0)
{-# NOINLINE reluCONV1outRef #-}

reluCONV1ginRef :: IORef (Tensor '[4, 6])
reluCONV1ginRef = unsafePerformIO $ newIORef (constant 0)
{-# NOINLINE reluCONV1ginRef #-}


reluCONV2outRef :: IORef (Tensor '[4, 6])
reluCONV2outRef = unsafePerformIO $ newIORef (constant 0)
{-# NOINLINE reluCONV2outRef #-}

reluCONV2ginRef :: IORef (Tensor '[4, 6])
reluCONV2ginRef = unsafePerformIO $ newIORef (constant 0)
{-# NOINLINE reluCONV2ginRef #-}


-- ========================================================================= --

fc1outRef :: IORef (Tensor '[4, 120])
fc1outRef = unsafePerformIO $ newIORef (constant 0)
{-# NOINLINE fc1outRef #-}

fc1ginRef :: IORef (Tensor '[4, 400])
fc1ginRef = unsafePerformIO $ newIORef (constant 0)
{-# NOINLINE fc1ginRef #-}


fc1gradparamRef :: IORef (Linear 400 120)
fc1gradparamRef = unsafePerformIO $ newIORef (Linear (constant 0, constant 0))
{-# NOINLINE fc1gradparamRef #-}

fc1BP :: Linear 400 120 -> Tensor '[4, 400] -> IO (Tensor '[4, 120], Tensor '[4, 120] -> IO (Linear 400 120, Tensor '[4, 400]))
fc1BP fc1 inp = do
  (out, getgrads) <- linearBatch 1 fc1outRef fc1ginRef fc1gradparamRef fc1 inp
  (fin, unrelu) <- reluBP__ out (relu1outRef, relu1ginRef) False
  pure (fin, \gout -> do
    g <- unrelu False gout
    getgrads g)


relu1outRef :: IORef (Tensor '[4, 120])
relu1outRef = unsafePerformIO $ newIORef (constant 0)
{-# NOINLINE relu1outRef #-}

relu1ginRef :: IORef (Tensor '[4, 120])
relu1ginRef = unsafePerformIO $ newIORef (constant 0)
{-# NOINLINE relu1ginRef #-}


-- ========================================================================= --

fc2outRef :: IORef (Tensor '[4, 84])
fc2outRef = unsafePerformIO $ newIORef (constant 0)
{-# NOINLINE fc2outRef #-}

fc2ginRef :: IORef (Tensor '[4, 120])
fc2ginRef = unsafePerformIO $ newIORef (constant 0)
{-# NOINLINE fc2ginRef #-}

fc2gradparamRef :: IORef (Linear 120 84)
fc2gradparamRef = unsafePerformIO $ newIORef (Linear (constant 0, constant 0))
{-# NOINLINE fc2gradparamRef #-}

fc2BP :: Linear 120 84 -> Tensor '[4, 120] -> IO (Tensor '[4, 84], Tensor '[4, 84] -> IO (Linear 120 84, Tensor '[4, 120]))
fc2BP fc2 inp = do
  (out, getgrads) <- linearBatch 1 fc2outRef fc2ginRef fc2gradparamRef fc2 inp
  (fin, unrelu) <- reluBP__ out (relu2outRef, relu2ginRef) False
  pure (fin, \gout -> do
    g <- unrelu False gout
    getgrads g)


relu2outRef :: IORef (Tensor '[4, 84])
relu2outRef = unsafePerformIO $ newIORef (constant 0)
{-# NOINLINE relu2outRef #-}

relu2ginRef :: IORef (Tensor '[4, 84])
relu2ginRef = unsafePerformIO $ newIORef (constant 0)
{-# NOINLINE relu2ginRef #-}



-- ========================================================================= --

fc3outRef :: IORef (Tensor '[4, 10])
fc3outRef = unsafePerformIO $ newIORef (constant 0)
{-# NOINLINE fc3outRef #-}

fc3ginRef :: IORef (Tensor '[4, 84])
fc3ginRef = unsafePerformIO $ newIORef (constant 0)
{-# NOINLINE fc3ginRef #-}

fc3gradparamRef :: IORef (Linear 84 10)
fc3gradparamRef = unsafePerformIO $ newIORef (Linear (constant 0, constant 0))
{-# NOINLINE fc3gradparamRef #-}

fc3BP :: Linear 84 10 -> Tensor '[4, 84] -> IO (Tensor '[4, 10], Tensor '[4, 10] -> IO (Linear 84 10, Tensor '[4, 84]))
fc3BP fc3 inp = do
  (fc3out, getfc3Grads) <- linearBatch 1 fc3outRef fc3ginRef fc3gradparamRef fc3 inp
  (fin, getlogsmGrads) <- logSoftMaxBatch fc3out
  pure (fin, getlogsmGrads >=> getfc3Grads)



-- | A mutating 'flatten' operation with batch
flattenBatchForward_
  :: (All KnownDim '[Product d, bs], Dimensions d)
  => Product (bs:+d) ~ Product '[bs, Product d]
  => (Tensor (bs:+d))
  -> IO (Tensor '[bs, Product d])
flattenBatchForward_ = _resizeDim

reluBP__ :: Tensor d -> (IORef (Tensor d), IORef (Tensor d)) -> Bool -> IO (Tensor d, Bool -> Tensor d -> IO (Tensor d))
reluBP__ inp (outref, ginref) inplace = do
  out <- readIORef outref
  Dynamic._threshold_updateOutput (asDynamic inp) (asDynamic out) 0 0 inplace
  pure (out, \ginplace gout -> do
    gin <- readIORef ginref
    Dynamic._threshold_updateGradInput
        (asDynamic inp) (asDynamic gout) (asDynamic gin) 0 0 ginplace
    pure gin)


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
  out <- readIORef outbufferRef
  zero_ out
  updateOutput_ l i out
  pure (out, \gout -> do

    gin <- readIORef gradinRef
    zero_ gin
    updateGradInput_ i gout (weights l) gin

    -- I am seeing that this is a fork : Notice there is no inter dependency here other than gout. `gout -> (linear -> dlinear, input s -> dinputs)`
    gparam <- readIORef gradparamRef
    zero_ (weights gparam)
    zero_ (bias gparam)
    accGradParameters_ i gout l gparam

    pure (gparam, gin))
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
      addmm_ 1 gw lr (transpose2d i) gout
      addmv_ 1 gb lr (transpose2d gout) (constant 1)

-- ========================================================================= --

-- | run a threshold function againts two BVar variables
logSoftMaxBatch
  :: Tensor '[4, 10]    -- ^ input
  -> IO (Tensor '[4, 10], Tensor '[4, 10] -> IO (Tensor '[4, 10]))   -- ^ output and gradient
logSoftMaxBatch inp = do
  out <- readIORef logsmoutRef
  updateOutput_ inp i out
  pure (out, \gout -> do
    gin <- readIORef logsmgRef
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



