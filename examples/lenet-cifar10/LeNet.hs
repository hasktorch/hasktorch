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

  -- * restart
  , maxPooling2dBatchIO
  , conv2dBatchIO
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
import Torch.Double hiding (logSoftMaxBatch, conv2dBatch)

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
    <- conv2dBatchIO
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
--    <- conv2dBatch'
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


nullFP :: IO (ForeignPtr a)
nullFP = newForeignPtr nullFunPtr nullPtr



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



