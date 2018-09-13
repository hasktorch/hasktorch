{-# LANGUAGE FlexibleInstances #-}
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
import Foreign
import Lens.Micro.Platform ((^.))
import System.IO.Unsafe
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
  gnet <- getgrads
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


getgrads = undefined




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


    Dynamic._spatialConvolutionMM_updateOutput (asDynamic i) (asDynamic convout2) (asDynamic (Conv2d.weights conv2)) (asDynamic (Conv2d.bias conv2))
      columnsbuff2 onesbuff2
      (kernel2d conv2)
      (param2d (Step2d    :: Step2d '(1,1)))
      (param2d (Padding2d :: Padding2d '(0,0)))

    -- relu forward
    (out, unrelu) <- reluBP__ convout2 (reluCONV2outRef, reluCONV2ginRef) False

    let maxpoolInput2 = out
    -- maxpoolOutput2 <- empty
    ix2 <- LDyn.empty

    Dynamic._spatialMaxPooling_updateOutput (asDynamic convout2) (asDynamic convout2) ix2
      (param2d  (Kernel2d  :: Kernel2d '(2,2)))
      (param2d  (Step2d    :: Step2d '(2,2)))
      (param2d  (Padding2d :: Padding2d '(0,0)))
      True -- (fromSing (sing      :: SBool 'True))

    -- print ix
    -- pure (maxpoolOutput2 :: Tensor '[4, 16, 5, 5])
    pure (asStatic (asDynamic convout2) :: Tensor '[4, 16, 5, 5])
  >>= \i -> (flattenBatchForward_ i :: IO (Tensor '[4, 16*5*5]))
  >>= \inp -> do
    out <- fst <$> fc1BP (arch ^. Vision.fc1) inp
    pure out

  >>= \inp -> do
    out <- fst <$> fc2BP (arch ^. Vision.fc2) inp
    pure out

  >>= \inp -> do
    fst <$> fc3BP (arch ^. Vision.fc3) inp

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



