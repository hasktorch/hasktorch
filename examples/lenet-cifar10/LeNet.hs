{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE NoImplicitPrelude #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeSynonymInstances #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE TypeApplications #-}
{-# OPTIONS_GHC -fno-cse #-}
module LeNet
  ( LeNet
  , lenetBatchForward
  , lenetBatchBP
  , lenetUpdate
  , lenetUpdate_
  , Vision._conv1
  , Vision._conv2
  , y2cat
  ) where

import Prelude hiding (print, putStrLn)
import qualified Prelude as P
import Control.Exception.Safe (throwString)
import Control.Monad
import Data.IORef
import Data.Maybe
import Data.Function ((&))
import Data.Singletons.Prelude.List (Product)
import Foreign hiding (new)
import Lens.Micro ((^.))
import System.IO.Unsafe
import Data.Singletons
import Data.Singletons.Prelude.Bool
import Torch.Double hiding (logSoftMaxBatch, conv2dMMBatch)

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
print = const (pure ()) --} P.print

putStrLn :: String -> IO ()
putStrLn = const (pure ()) --} P.putStrLn

bsz = (dim :: Dim 4)
bs = (fromIntegral $ dimVal bsz) :: Int
type LeNet = Vision.LeNet 3 5

newLeNet :: IO LeNet
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



logsmoutRef :: IORef ((Tensor '[4, 10]))
logsmoutRef = unsafePerformIO $ newIORef (constant 0)
{-# NOINLINE logsmoutRef #-}


nullIx :: IORef Long.Dynamic
nullIx = unsafePerformIO $ do
  -- newForeignPtr nullPtr
  newIORef undefined
{-# NOINLINE nullIx #-}
-- ========================================================================= --

-- reluCONV2outRef :: IORef (Tensor '[4, 16, 2, 2])
reluCONV2outRef = unsafePerformIO $ newIORef (constant 0)
{-# NOINLINE reluCONV2outRef #-}

-- reluCONV2ginRef :: IORef (Tensor '[4, 16, 2, 2])
reluCONV2ginRef = unsafePerformIO $ newIORef (constant 0)
{-# NOINLINE reluCONV2ginRef #-}

-- ========================================================================= --

lenetUpdate_ :: LeNet -> (HsReal, LeNet) -> IO ()
lenetUpdate_ net (lr, g) = Vision.update_ net lr g

lenetUpdate :: LeNet -> (HsReal, LeNet) -> LeNet
lenetUpdate net (lr, g) = Vision.update net lr g

lenetBatchForward
  :: LeNet
  ->    (Tensor '[4, 3, 32, 32])  -- ^ input
  -> IO (Tensor '[4, 10])         -- ^ output
lenetBatchForward net inp = fst <$> lenetBatch False undefined net inp


lenetBatchBP
  :: LeNet                          -- ^ architecture
  ->    (IndexTensor '[4])          -- ^ ys
  ->    (Tensor '[4, 3, 32, 32])    -- ^ xs
  -> IO (Tensor '[1], LeNet)    -- ^ output and gradient
lenetBatchBP arch ys xs = do
  -- let n = (constant 1 :: Tensor '[2, 3, 5])
  -- print n
  -- newStrideOf n >>= LStorage.tensordata >>= print
  -- newSizeOf n >>= LStorage.tensordata >>= print
  (out, getlenetgrad) <- lenetBatch True 1 arch xs
  -- throwString "working? next check that all values are < 2.73~"
  -- putStrLn "\n======================================"
  -- print out
  -- putStrLn   "======================================"

  -- halt 2 "all values are < 2.73~?"
  (loss, getCEgrad) <- crossentropy ys out
  -- putStrLn "\n--------------------------------------"
  -- print loss
  -- putStrLn   "--------------------------------------"
  cegrad <- getCEgrad loss
  (gnet, _) <- getlenetgrad cegrad
  -- print (Conv2d.weights $ gnet ^. Vision.conv1)
  -- print (Conv2d.weights $ gnet ^. Vision.conv2)
  -- print (Linear.weights $ gnet ^. Vision.fc1)
  -- print (Linear.weights $ gnet ^. Vision.fc2)
  -- print (Linear.weights $ gnet ^. Vision.fc3)
  -- print "!!!"
  -- halt 2 "!!!"
  pure (loss, gnet)

crossentropy
  :: IndexTensor '[4]            -- THIndexTensor *target,
  -> Tensor '[4, 10]            -- THTensor *input,
  -> IO (Tensor '[1], Tensor '[1] -> IO (Tensor '[4, 10]))               -- THTensor *output, gradient
crossentropy ys inp = do
  -- putStrLn "!!!!!!!!!!!!!!!!!!!!!!!! crossentropy !!!!!!!!!!!!!!!!!!!!!!!!"
  -- -- print inp
  -- putStrLn "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
  (lout, getLSMGrad) <- logSoftMaxBatch inp
  (nllout, getNLLGrad) <- classNLL ys lout
  pure (nllout, getNLLGrad >=> getLSMGrad)


y2cat :: Tensor '[4, 10] -> IO [Category]
y2cat ys = do
  putStrLn "starting to build tensor"
  print ys
  let rez = fromJust . snd $ Torch.max2d0 ys keep
  print rez
  mapM ((\i -> pure . toEnum . fromIntegral . fromJust $ Long.get2d rez i 0)) [0..3]
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



lenetBatch
  :: Bool                         -- ^ if you should perform backprop as well
  -> HsReal
  -> LeNet
  ->    (Tensor '[4, 3, 32, 32])  -- ^ input
  -> IO (Tensor '[4, 10], Tensor '[4, 10] -> IO (LeNet, Tensor '[4, 3, 32, 32]))
lenetBatch training lr arch i = do
    -- ========================================================================= --
    -- LAYER 1

    -- when training $ convin1Ref `updateIORefWith` i
    when tverbose $ do
      print "------------FORWARD-------------"
      print "getting convout1"

    (convout1 :: Tensor '[4, 6, 28, 28], unconvout1) <-
      conv2dMMBatch
        conv1ginbuffRef         -- IORef (Tensor '[b,f,h,w])     -- grad input buffer
        conv1columnsbuffRef     -- IORef (Tensor '[])            -- columns buffer
        conv1onesbuffRef        -- IORef (Tensor '[])            -- ones buffer
        conv1outRef             -- IORef (Tensor '[b, o,oH,oW])  -- output
        conv1iRef               -- IORef (Tensor '[b,f,h,w])     -- input
        conv1ginRef             -- IORef (Tensor '[b,f,h,w])     -- gradient input
        conv1gparamsRef         -- IORef (Conv2d f o '(kH, kW))  -- gradient params
        ((Step2d    :: Step2d '(1,1)))
        ((Padding2d :: Padding2d '(0,0)))
        lr
        (arch ^. Vision.conv1)
        (i::Tensor '[4, 3, 32, 32])

    when tverbose $ print convout1
    -- halt 2 "ORISET"

    (reluout1 :: Tensor '[4, 6, 28, 28], unrelu1) <- reluBP__ convout1 (reluCONV1outRef, reluCONV1ginRef) False
    (mpout1 :: Tensor '[4, 6, 14, 14], getmpgrad1) <- maxPooling2dBatch'
      mp1ixRef
      mp1outRef
      mp1ginRef
      ((Kernel2d  :: Kernel2d '(2,2)))
      ((Step2d    :: Step2d '(2,2)))
      ((Padding2d :: Padding2d '(0,0)))
      ((sing      :: SBool 'True))
      reluout1

    -- ========================================================================= --
    -- LAYER 2
    when tverbose $ print "getting convout2"
    (convout2::Tensor '[4, 16, 10, 10], unconvout2) <- conv2dMMBatch
        conv2ginbuffRef         -- IORef (Tensor '[b,f,h,w])     -- grad input buffer
        conv2columnsbuffRef     -- IORef (Tensor '[])            -- columns buffer
        conv2onesbuffRef        -- IORef (Tensor '[])            -- ones buffer
        conv2outRef             -- IORef (Tensor '[b, o,oH,oW])  -- output
        conv2iRef               -- IORef (Tensor '[b,f,h,w])     -- input
        conv2ginRef             -- IORef (Tensor '[b,f,h,w])     -- gradient input
        conv2gparamsRef         -- IORef (Conv2d f o '(kH, kW))  -- gradient params
        ((Step2d    :: Step2d '(1,1)))
        ((Padding2d :: Padding2d '(0,0)))
        lr
        (arch ^. Vision.conv2)
        mpout1

    (reluout2 :: Tensor '[4, 16, 10, 10], unrelu2) <- reluBP__ convout2 (reluCONV2outRef, reluCONV2ginRef) False

    (mpout2 :: Tensor '[4,16,5,5], getmpgrad2) <- maxPooling2dBatch'
      mp2ixRef
      mp2outRef
      mp2ginRef
      ((Kernel2d  :: Kernel2d '(2,2)))
      ((Step2d    :: Step2d '(2,2)))
      ((Padding2d :: Padding2d '(0,0)))
      ((sing      :: SBool 'True))
      reluout2

    -------------------------------------------------------------------------------
    -- print mpout2

    when tverbose $ print "getting flatten"
    (ftout  :: Tensor '[4, 400], unflatten) <- flattenBPBatch_ mpout2
    when tverbose $ print "getting fc1"
    (fc1out :: Tensor '[4, 120], fc1getgrad) <- fc1BP (arch ^. Vision.fc1) ftout
    when tverbose $ print "getting fc2"
    (fc2out :: Tensor '[4,  84], fc2getgrad) <- fc2BP (arch ^. Vision.fc2) fc1out
    when tverbose $ print "getting fc3"
    (fc3out :: Tensor '[4,  10], fc3getgrad) <- fc3BP (arch ^. Vision.fc3) fc2out


    when tverbose $ print "done!"
    when tverbose $ print "----------END------------"

    pure (fc3out, \(gout::Tensor '[4, 10]) -> do
      when verbose $ do
        print "+++++++++++AD++++++++++++"
        print "getting fc3g"
        print gout
      (fc3g::Linear   84 10, fc3gin::Tensor '[4,  84]) <- fc3getgrad gout
      when verbose $ do
        -- print (Linear.weights fc3g)
        print "getting fc2g"

      (fc2g::Linear  120 84, fc2gin::Tensor '[4, 120]) <- fc2getgrad fc3gin
      when verbose $ do
        print "getting fc1g"
        pure (shape fc2gin) >>= print
      (fc1g::Linear 400 120, fc1gin::Tensor '[4, 400]) <- fc1getgrad fc2gin
      when verbose $ print "getting flatteng"

      inflatedg :: Tensor '[4,16, 5, 5] <- unflatten fc1gin
      when verbose $ print "getting conv2g"

      (conv2g::Conv2d 6 16 '(5,5), conv2gin :: Tensor '[4, 6, 14, 14]) <- unconvout2 =<< unrelu2 True =<< getmpgrad2 inflatedg
      when verbose $ print "getting conv1g"

      (conv1g::Conv2d 3  6 '(5,5), conv1gin :: Tensor '[4, 3, 32, 32]) <- unconvout1 =<< unrelu1 True =<< getmpgrad1 conv2gin

      when verbose $ do
        print "done!"
        print "+++++++++++END++++++++++++"

      pure (Vision.LeNet conv1g conv2g fc1g fc2g fc3g, conv1gin))

  where
    verbose = False
    tverbose = False
    -- tverbose = verbose && not training

mp1ixRef = unsafePerformIO $ newIORef (Long.constant 0)
{-# NOINLINE mp1ixRef #-}
mp1outRef = unsafePerformIO $ newIORef (constant 0)
{-# NOINLINE mp1outRef #-}
mp1ginRef = unsafePerformIO $ newIORef (constant 0)
{-# NOINLINE mp1ginRef #-}

conv1ginbuffRef     = unsafePerformIO $ pure new >>= newIORef -- (Tensor '[b,f,h,w])     -- grad input buffer
{-# NOINLINE conv1ginbuffRef #-}
conv1columnsbuffRef = unsafePerformIO $ pure new >>= newIORef -- (Tensor '[])            -- columns buffer
{-# NOINLINE conv1columnsbuffRef #-}
conv1onesbuffRef    = unsafePerformIO $ pure new >>= newIORef -- (Tensor '[])            -- ones buffer
{-# NOINLINE conv1onesbuffRef #-}
conv1outRef         = unsafePerformIO $ pure new >>= newIORef -- (Tensor '[b, o,oH,oW])  -- output
{-# NOINLINE conv1outRef #-}
conv1iRef           = unsafePerformIO $ pure new >>= newIORef -- (Tensor '[b,f,h,w])     -- input
{-# NOINLINE conv1iRef #-}
conv1ginRef         = unsafePerformIO $ pure new >>= newIORef -- (Tensor '[b,f,h,w])     -- gradient input
{-# NOINLINE conv1ginRef #-}
conv1gparamsRef     = unsafePerformIO $ pure (Conv2d (new, new)) >>= newIORef  -- (Conv1d f o '(kH, kW))  -- gradient params
{-# NOINLINE conv1gparamsRef #-}


-- reluCONV1outRef :: IORef (Tensor '[4, 6])
reluCONV1outRef = unsafePerformIO $ newIORef (constant 0)
{-# NOINLINE reluCONV1outRef #-}
-- reluCONV1ginRef :: IORef (Tensor '[4, 6])
reluCONV1ginRef = unsafePerformIO $ newIORef (constant 0)
{-# NOINLINE reluCONV1ginRef #-}
conv2ginbuffRef     = unsafePerformIO $ pure new >>= newIORef -- (Tensor '[b,f,h,w])     -- grad input buffer
{-# NOINLINE conv2ginbuffRef #-}
conv2columnsbuffRef = unsafePerformIO $ pure new >>= newIORef -- (Tensor '[])            -- columns buffer
{-# NOINLINE conv2columnsbuffRef #-}
conv2onesbuffRef    = unsafePerformIO $ pure new >>= newIORef -- (Tensor '[])            -- ones buffer
{-# NOINLINE conv2onesbuffRef #-}
conv2outRef         = unsafePerformIO $ pure new >>= newIORef -- (Tensor '[b, o,oH,oW])  -- output
{-# NOINLINE conv2outRef #-}
conv2iRef           = unsafePerformIO $ pure new >>= newIORef -- (Tensor '[b,f,h,w])     -- input
{-# NOINLINE conv2iRef #-}
conv2ginRef         = unsafePerformIO $ pure new >>= newIORef -- (Tensor '[b,f,h,w])     -- gradient input
{-# NOINLINE conv2ginRef #-}
conv2gparamsRef     = unsafePerformIO $ pure (Conv2d (new, new)) >>= newIORef  -- (Conv2d f o '(kH, kW))  -- gradient params
{-# NOINLINE conv2gparamsRef #-}


-- | Backprop convolution function with batching
conv2dMMBatch
  :: forall f h w kH kW dH dW pH pW oW oH s o b
  .  SpatialConvolutionC f h w kH kW dH dW pH pW oH oW
  => All KnownDim '[f,o,b,kW*kH*f,oH*oW]
  => IORef (Tensor '[b,f,h,w])            -- ^ grad input buffer

  -- buffers
  -> IORef (Tensor '[b,kW*kH*f,oH*oW])            -- ^ columns buffer
  -> IORef (Tensor '[b,kW*kH*f,oH*oW])            -- ^ ones buffer

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


nullFP :: IO (ForeignPtr a)
nullFP = newForeignPtr nullFunPtr nullPtr

conv2dMM__
  :: forall din dout fgin f o kH kW dH dW pH pW inBuff
  .  All Dimensions '[din,dout,fgin, inBuff]
  => All KnownDim '[f,o,kH,kW,dH,dW,pH,pW]

  -- buffers
  => IORef (Tensor fgin)            -- ^ grad input buffer
  -> IORef (Tensor inBuff)            -- ^ columns buffer
  -> IORef (Tensor inBuff)            -- ^ ones buffer

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
  zero_ onesbuff

  -- THTensor_(resize2d)(finput, kW*kH*nInputPlane, outputHeight*outputWidth);
  zero_ columnsbuff

  out <- readIORef outref
  zero_ out

  -- print out
  -- print (Conv2d.weights out)
  -- buff <- (DoubleTensor . dynamic torchstate) <$> nullFP
  -- print inp
  updateOutput_ (constant 0 :: Tensor inBuff) (constant 0 :: Tensor inBuff) step pad conv inp out
  -- print out
  -- halt 0 "x"


  pure (out,
    \gout -> do
      ginbuffer <- readIORef ginbufferRef
      zero_ ginbuffer
      gin <- readIORef ginref
      zero_ gin
      gparams <- readIORef gparamsref
      zero_ (Conv2d.weights gparams)
      zero_ (Conv2d.bias gparams)
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

      (asDynamic colbuff)                  -- ^ BUFFER: temporary columns -- also called "finput"
      (asDynamic onesbuff)                 -- ^ BUFFER: buffer of ones for bias accumulation  -- also called "fgradInput"

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
flattenBPBatch_ i = do
  o <- pure $ resizeAs i
  pure (o, \gout -> pure $ resizeAs gout)


-- ========================================================================= --

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
  pure (fin, \gout -> unrelu False gout >>= getgrads)


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
  pure (fin, \gout -> do

    -- print "gout"
    -- print gout
    x <- getlogsmGrads gout
    -- print "x"
    -- print x
    y <- getfc3Grads x
    -- print "y"
    -- print y
    pure y)
    -- (getlogsmGrads >=> getfc3Grads) gout)

reluBP__
  :: forall d . Dimensions d
  => Tensor d -> (IORef (Tensor d), IORef (Tensor d)) -> Bool -> IO (Tensor d, Bool -> Tensor d -> IO (Tensor d))
reluBP__ inp (outref, ginref) inplace = do
  out <- readIORef outref
  Dynamic._threshold_updateOutput (asDynamic inp) (asDynamic out) 0 0 inplace
  pure (out, \ginplace gout -> do
    -- throwString "xxxx"
    replaceIORefWith ginref (constant 0 :: Tensor d)
    -- zero_ ginref

    gin <- readIORef ginref

    -- zero_ gin
    -- print "GOUT>>>>>>>>>>>>>"
    -- print (asDynamic gout)
    -- print "<<<<<<<<<<<<<<<<<"
    -- print "GIN>>>>>>>>>>>>>>"
    -- print (asDynamic gin)
    -- print "<<<<<<<<<<<<<<<<<"

    Dynamic._threshold_updateGradInput
        (asDynamic inp) (asDynamic gout) (asDynamic gin) 0 0 False

    -- print "GOUT>>>>>>>>>>>>>"
    -- print (asDynamic gout)
    -- print "<<<<<<<<<<<<<<<<<"
    -- print "GIN>>>>>>>>>>>>>>"
    -- print (asDynamic gin)
    -- print "<<<<<<<<<<<<<<<<<"

    -- zero_ gin
    -- print "GOUT>>>>>>>>>>>>>"
    -- print (asDynamic gout)
    -- print "<<<<<<<<<<<<<<<<<"
    -- print "GIN>>>>>>>>>>>>>>"
    -- print (asDynamic gin)
    -- print "<<<<<<<<<<<<<<<<<"

    -- throwString "x"
    pure gin)


reluBP
  :: forall d . Dimensions d
  => Tensor d
  -> (IORef (Tensor d), IORef (Tensor d)) -> IO (Tensor d, Tensor d -> IO (Tensor d))
reluBP inp (outref, ginref) = do
  out <- readIORef outref
  Dynamic._threshold_updateOutput (asDynamic inp) (asDynamic out) 0 0 False
  pure (out, \gout -> do
    Dynamic._threshold_updateGradInput
        (asDynamic inp) (asDynamic gout) (asDynamic gout) 0 0 False
    pure gout)


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
  out <- readIORef outbufferRef
  zero_ out
  updateOutput_ l i out

  pure (out, \gout -> do
    putStrLn "\nglinear"
    when (dimVal (dim :: Dim o) == 10) $ do
      putStrLn "Starting linear reverse\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
      putStrLn "getting gin"
      -- print gout

    gin <- readIORef gradinRef
    -- print ("gin shape:", shape gin)
    zero_ gin
    -- print gout
    -- print "updating gin"
    -- print gout
    updateGradInput_ i gout (Linear.weights l) gin
    when (dimVal (dim :: Dim o) == 10) $ do
      pure ()
      -- print gin
    -- print gout
    -- halt 6 "stop!!!"

    -- I am seeing that this is a fork : Notice there is no inter dependency here other than gout. `gout -> (linear -> dlinear, input s -> dinputs)`
    -- print "getting gparam"
    gparam <- readIORef gradparamRef
    zero_ (Linear.weights gparam)
    zero_ (Linear.bias gparam)
    -- print "updating gparam"
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
      -- print (shape gw, shape gb, shape gout)
      addmm_ 1 gw lr (transpose2d i)    gout
      addmv_ 1 gb lr (transpose2d gout) (constant 1)

    -- | Performs a matrix-matrix multiplication between @mat1@ (2D Tensor) and @mat2@ (2D Tensor).
    --
    -- Values @v1@ and @v2@ are scalars that multiply @M@ and @mat1 * mat2@ respectively.
    -- They are optional in C and we may be able to add this to the API in the future.
    --
    -- In other words,
    --
    -- @
    --   res = (v1 * M) + (v2 * mat1 * mat2)
    -- @
    --
    -- If @mat1@ is a @n × m@ matrix, @mat2@ a @m × p@ matrix, @M@ must be a @n × p@ matrix.
    --
    -- -- | Inline version of 'addmm', mutating @M@ inplace.
    -- addmm_
    --   :: HsReal    -- ^ v1
    --   -> Dynamic   -- ^ M
    --   -> HsReal    -- ^ v2
    --   -> Dynamic   -- ^ mat1
    --   -> Dynamic   -- ^ mat2
    --   -> IO ()

    -- accGradParameters :: Tensor '[b,i] -> Tensor '[b,o] -> Linear i o -> Linear i o
    -- accGradParameters i gout (Linear (w, b)) = Linear (gw, gb) -- addr 1 (constant 0) lr i gout, cadd (constant 0) lr gout)
    --   where
    --     gw :: Tensor '[i, o]
    --     gw = addmm 1 (constant 0) lr (transpose2d i) gout

    --     gb :: Tensor '[o]
    --     gb = addmv 1 (constant 0) lr tgout (constant 1)

    --     tgout :: Tensor '[o,b]
    --     tgout = transpose2d gout


-- ========================================================================= --

-- | run a threshold function againts two BVar variables
logSoftMaxBatch
  :: Tensor '[4, 10]    -- ^ input
  -> IO (Tensor '[4, 10], Tensor '[4, 10] -> IO (Tensor '[4, 10]))   -- ^ output and gradient
logSoftMaxBatch inp = do
  replaceIORefWith logsmoutRef (constant 0)
  out <- readIORef logsmoutRef

  updateOutput_ inp i out

  -- print "outty"
  -- print inp
  -- print "outty"

  -- print out

  pure (out, \gout -> do

    -- putStrLn ""
    -- putStrLn ",--------------------------------------------------Q.Q-------------------------------------------------."
    -- putStrLn "|                                                                                                      |"
    -- print gout

    replaceIORefWith logsmgRef (constant 0)
    -- updateIORefWith logsmgRef (constant 0)
    gin <- readIORef logsmgRef

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



