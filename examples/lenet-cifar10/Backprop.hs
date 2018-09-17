{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeSynonymInstances #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE TypeApplications #-}
{-# OPTIONS_GHC -fno-cse #-}
module Backprop
  ( newLeNet
  , bs, bsz, LeNet
  , lenetBatchForward
  , lenetBatchBP
  , lenetUpdate
  , Vision._conv1
  , Vision._conv2
  ) where

import Control.Exception.Safe (throwString)
import Control.Monad
import Data.IORef
import Data.Function ((&))
import Data.Singletons.Prelude.List (Product)
import Foreign
import Lens.Micro.Platform
import System.IO.Unsafe
import Torch.Double
import Torch.Double.NN
import Torch.Double.NN.Linear
import qualified Torch.Long as Long
import qualified Torch.Long.Dynamic as LDyn
import qualified Torch.Double.NN.Conv2d as Conv2d
import qualified Torch.Double as Torch
import qualified Torch.Double.Storage as CPUS
import qualified Torch.Models.Vision.LeNet as Vision

import qualified Torch.Double.Dynamic as Dynamic
import qualified Torch.Double.Dynamic.NN as Dynamic
import qualified Torch.Double.Dynamic.NN.Activation as Dynamic


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

fcout3Ref :: IORef (Tensor '[4, 10])
fcout3Ref = unsafePerformIO $ newIORef (constant 0)
{-# NOINLINE fcout3Ref #-}


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
  -- :: (Tensor '[4, 10] -> IO HsReal) -- ^ criterion              <<< hardcode CE
  :: LeNet                          -- ^ architecture
  ->    (IndexTensor '[4])          -- ^ ys
  ->    (Tensor '[4, 3, 32, 32])    -- ^ xs
  -> IO (Tensor '[1], LeNet)    -- ^ output and gradient
lenetBatchBP arch ys xs = do
  out <- lenetBatch True arch xs
  loss <- crossentropy ys out
  gnet <- getgrads
  pure (loss, gnet)

crossentropy
  :: IndexTensor '[4]            -- THIndexTensor *target,
  -> Tensor '[4, 10]            -- THTensor *input,
  -> IO (Tensor '[1])               -- THTensor *output,
crossentropy ys inp
  = undefined
  -- = logSoftMaxN (dim :: Dim 1) inp
  -- & classNLLCriterion ys

getgrads = undefined

-- -- By default, the losses are averaged over observations for each minibatch. However, if the argument sizeAverage is set to false, the losses are instead summed for each minibatch.
-- -- FIXME: add batch dimension
-- classNLLCriterion'
--   :: forall s i sz ps
--   . (Reifies s W, All KnownDim '[sz, ps])
--   => Integer                    -- int64_t ignore_index,
--   -> Bool                       -- bool sizeAverage,
--   -> Bool                       -- bool reduce
--   -> IndexTensor '[sz]          -- THIndexTensor *target. _not_ a one-hot encoded vector.
--   -- -> Maybe Dynamic           -- THTensor *weights,
--   -> BVar s (Tensor '[sz, ps])  -- THTensor *input,
--   -> BVar s (Tensor '[1])       -- THTensor *output,
-- classNLLCriterion' ix szAvg reduce target = liftOp1 . op1 $ \inp ->
--   let
--     (out, total_weight) = updateOutput inp target szAvg Nothing ix reduce
--   in
--     (out, \gout -> updateGradInput inp target gout szAvg Nothing total_weight ix reduce)
--
--     {-# NOINLINE updateOutput #-}
--     updateOutput
--       :: Tensor '[sz, ps]            -- THTensor *input,
--       -> IndexTensor '[sz]           -- THIndexTensor *target,
--       -> Bool                        -- bool sizeAverage,
--       -> Maybe (Tensor '[sz, ps])    -- THTensor *weights,
--       -> Integer                     -- int64_t ignore_index,
--       -> Bool                        -- bool reduce
--       -> (Tensor '[1], Tensor '[1])
--     updateOutput inp tar szAvg mws ix reduce = unsafePerformIO $ do
--       out <- new
--       let total_weight = constant 1  -- https://github.com/torch/nn/commit/3585e827eb65d071272a4aa4fab567b0b1eeee54#diff-1aa6a505cf16ad0e59498ada8432afb5
--       Dynamic._ClassNLLCriterion_updateOutput (asDynamic inp) (Ix.longAsDynamic tar) (asDynamic out)
--         szAvg (asDynamic <$> mws) (asDynamic total_weight) ix reduce
--       pure (out, total_weight)
--
--     {-# NOINLINE updateGradInput #-}
--     updateGradInput
--       :: Tensor '[sz, ps]          -- THTensor *input,
--       -> IndexTensor '[sz]         -- THIndexTensor *target,
--       -> Tensor '[1]               -- THTensor *gradOutput,
--       -> Bool                      -- bool sizeAverage,
--       -> Maybe (Tensor '[sz, ps])  -- THTensor *weights,
--       -> Tensor '[1]               -- THTensor *total_weight,
--       -> Integer                   -- int64_t ignore_index,
--       -> Bool                      -- bool reduce
--       -> Tensor '[sz, ps]
--     updateGradInput inp tar gout szAvg mws total_weight ix reduce = unsafePerformIO . withEmpty $ \gin ->
--       Dynamic._ClassNLLCriterion_updateGradInput (asDynamic inp) (Ix.longAsDynamic tar) (asDynamic gout) (asDynamic gin)
--         szAvg (asDynamic <$> mws) (asDynamic total_weight) ix reduce
--


-- crossEntropy
--   :: (All KnownDim '[b, p])
--   => IndexTensor '[b]            -- THIndexTensor *target,
--   -> (Tensor '[b, p])            -- THTensor *input,
--   -> (Tensor '[1])               -- THTensor *output,
-- crossEntropy ys inp
--   = logSoftMaxN (dim :: Dim 1) inp
--   & classNLLCriterion ys


updateIORefWith :: IORef (Maybe (Tensor d)) -> Tensor d -> IO ()
updateIORefWith ref nxt =
  readIORef ref >>= \case
    Nothing  -> writeIORef ref (Just $ copy nxt)

    Just old ->
      finalizeForeignPtr oldfp >> writeIORef ref Nothing
     where
      oldfp = ctensor (asDynamic old)



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
    reluForward_ convout1

    let maxpoolInput1 = convout1
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
    reluForward_ convout2

    let maxpoolInput2 = convout2
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
    out <- readIORef fcout1Ref
    zero_ out
    -- print "hello1"
    linearForwardBatch_ (arch ^. Vision.fc1) inp out
    reluForward_ out
    pure out

  >>= \inp -> do
    out <- readIORef fcout2Ref
    zero_ out
    -- print "hello2"
    linearForwardBatch_ (arch ^. Vision.fc2) inp out
    reluForward_ out
    pure out

  >>= \inp -> do
    out <- readIORef fcout3Ref
    zero_ out
    -- print "hello3"
    linearForwardBatch_ (arch ^. Vision.fc3) inp out
    -- fin <- empty
    Dynamic._logSoftMax_updateOutput (asDynamic out) (asDynamic out) 1 -- (fromIntegral $ dimVal 1)
    -- throwString "hello!"
    pure out



-- | A mutating 'flatten' operation with batch
flattenBatchForward_
  :: (All KnownDim '[Product d, bs], Dimensions d)
  => Product (bs:+d) ~ Product '[bs, Product d]
  => (Tensor (bs:+d))
  -> IO (Tensor '[bs, Product d])
flattenBatchForward_ = _resizeDim

-- lenetLayerBatchForward1
--   :: forall inp h w ker ow oh out mow moh step pad batch
--
--   -- backprop constraint to hold the wengert tape
--
--   -- leave input, output and square kernel size variable so that we
--   -- can reuse the layer...
--   .  All KnownDim '[4,inp,out,ker]
--
--   -- FIXME: derive these from the signature (maybe assign them as args)
--   => pad ~ 0   --  default padding size
--   => step ~ 1  --  default step size for Conv2d
--
--   -- ...this means we need the constraints for conv2dMM and maxPooling2d
--   -- Note that oh and ow are then used as input to the maxPooling2d constraint.
--   => SpatialConvolutionC inp h  w ker ker 1 1 0 0  oh  ow
--   => SpatialDilationC       oh ow   2   2 2 2 0 0 mow moh 1 1 'True
--
--   =>    (Conv2d inp out '(ker,ker))       -- ^ convolutional layer
--   ->    (Tensor '[4, inp,   h,   w])  -- ^ input
--   -> IO (Tensor '[4, out, moh, mow])  -- ^ output
-- lenetLayerBatchForward1 conv inp = do
--   conv2out <- empty
--   columnsbuff <- Dynamic.empty
--   onesbuff <- Dynamic.empty
--   Dynamic._spatialConvolutionMM_updateOutput (asDynamic inp) (asDynamic conv2out) (asDynamic (Conv2d.weights conv)) (asDynamic (Conv2d.bias conv))
--     columnsbuff onesbuff
--     (kernel2d conv)
--     (param2d (Step2d    :: Step2d '(1,1)))
--     (param2d (Padding2d :: Padding2d '(0,0)))
--
--   -- relu forward
--   let reluInput = conv2out
--   reluForward_ reluInput
--
--   let maxpoolInput = reluInput
--   maxpoolOutput <- empty
--   ix <- LDyn.empty
--
--   Dynamic._spatialMaxPooling_updateOutput (asDynamic maxpoolInput) (asDynamic maxpoolOutput) ix
--     (param2d  (Kernel2d  :: Kernel2d '(2,2)))
--     (param2d  (Step2d    :: Step2d '(2,2)))
--     (param2d  (Padding2d :: Padding2d '(0,0)))
--     True -- (fromSing (sing      :: SBool 'True))
--
--   -- print ix
--   pure maxpoolOutput
--

reluForward_ :: Tensor d -> IO ()
reluForward_ reluInput = do
  Dynamic._threshold_updateOutput (asDynamic reluInput) (asDynamic reluInput) 0 0 (True {- inplace -})
  -- print (asDynamic reluInput)
  -- print (asDynamic reluOutput)


-- | 'linear' with a batch dimension
linearForwardBatch_
  :: forall s i o b
  .  All KnownDim '[b,i,o]
  => (Linear i o)
  -> (Tensor '[b, i])
  -> (Tensor '[b, o]) -- should be initialized at zeros
  -> IO ()
linearForwardBatch_ (Linear (w,b)) inp out = do
  addmm_ 0 out 1 inp w
  addr_ 1 out 1 (constant 1) b












-- -- | run a threshold function againts two BVar variables
-- my_logSoftMaxN
--   :: forall s i d
--   .  i < Length d ~ 'True
--   => Dimensions d
--   => Dim i                -- ^ dimension to logSoftMax over
--   -> (Tensor d -> IO (Tensor d, (Tensor d -> IO Tensor d)))        -- ^ output
-- my_logSoftMaxN i = liftOp1 . op1 $ \inp ->
--   let out = updateOutput inp i
--   in (updateOutput inp i, \gout -> updateGradInput inp gout out i)
--  where
--   idim = fromIntegral (dimVal i)
--
--   {-# NOINLINE updateOutput #-}
--   updateOutput :: Tensor d -> Dim i -> Tensor d
--   updateOutput inp i = unsafePerformIO . withEmpty $ \out ->
--     Dynamic._logSoftMax_updateOutput (asDynamic inp) (asDynamic out) idim
--
--   {-# NOINLINE updateGradInput #-}
--   updateGradInput
--     :: Tensor d  -- input
--     -> Tensor d  -- gradOutput
--     -> Tensor d  -- output
--     -> Dim i     -- dimension
--     -> Tensor d  -- gradInput
--   updateGradInput inp gout out i = unsafePerformIO . withEmpty $ \gin ->
--     Dynamic._logSoftMax_updateGradInput
--       (asDynamic inp)             -- input
--       (asDynamic gout)            -- gradOutput
--       (asDynamic gin)             -- gradInput
--       (asDynamic out)             -- output
--       (fromIntegral $ dimVal i)   -- dimension











































