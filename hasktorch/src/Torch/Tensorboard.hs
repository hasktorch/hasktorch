-- |
-- Module      : Torch.Internal.CRC32C
-- Copyright   : (c) 2017-2019 Andrei Barbu
-- original from https://github.com/abarbu/haskell-torch/blob/master/haskell-torch/src/Torch/Tensorboard.hs

{-# LANGUAGE AllowAmbiguousTypes, CPP, ConstraintKinds, DataKinds, FlexibleContexts, FlexibleInstances, GADTs, MultiParamTypeClasses #-}
{-# LANGUAGE OverloadedLabels, OverloadedStrings, PartialTypeSignatures, PolyKinds, QuasiQuotes, RankNTypes, ScopedTypeVariables     #-}
{-# LANGUAGE TypeApplications, TypeFamilyDependencies, TypeInType, TypeOperators, UndecidableInstances                               #-}
{-# options_ghc -fplugin GHC.TypeLits.KnownNat.Solver #-}
{-# options_ghc -Wno-partial-type-signatures -fconstraint-solver-iterations=10 #-}

{-| Talk 
to Tensorboard. You will need to run it manually with 'tensorboard --logdir <yourdir>'.
It will occasionally update on its own.
We have fairly comprehensive Tensorboard support and integration with showing images, plots, and graphs.
-}

module Torch.Tensorboard where
import           Control.Monad
import           Control.Monad.Logger
import           Data.Binary.Put
import           Data.Bits
import qualified Data.ByteString              as B
import qualified Data.ByteString              as BS
import qualified Data.ByteString.Internal     as BS
import qualified Data.ByteString.Lazy         as BL
import           Data.Coerce
import           Data.IORef
import qualified Data.Map                     as M
import           Data.ProtoLens
import           Data.ProtoLens.Labels        ()
import           Data.Singletons
import           Data.Singletons.TypeLits
import           Data.Text                    (Text)
import qualified Data.Text                    as T
import qualified Data.Text.Encoding           as T
import           Data.Time.Clock.POSIX
import           Data.Time.Format
import           Data.Time.LocalTime
import           Data.Vector.Storable         (Vector)
import qualified Data.Vector.Storable         as V
import           Foreign.C.Types
import           Foreign.ForeignPtr
import           Foreign.Ptr
import           GHC.Float
import           GHC.Int
import           GHC.Word
import qualified Graphics.Matplotlib          as M
import           Lens.Micro
import           Network.HostName
import qualified Statistics.Quantile          as S
import           System.Directory
import           System.FilePath
import           System.IO

import qualified Tensorboard.Proto.Attributes as PA
import qualified Tensorboard.Proto.Event      as PE
import qualified Tensorboard.Proto.Graph      as PG
import qualified Tensorboard.Proto.Summary    as PS
import qualified Tensorboard.Proto.Tensor     as PT

-- import qualified Torch.C.Tensor               as C
-- import qualified Torch.C.Types                as C
-- import qualified Torch.C.Variable             as C
-- import           Torch.Images
import           Torch.Internal.CRC32C
-- import           Torch.Tensor                 as Tensor
-- import           Torch.Types                  as Tensor
-- import           Torch.Visualization

import qualified Torch.Typed
import qualified Torch.Device                  as D
import qualified Torch.Functional              as F

-- | `SummaryWriter` keeps track of some basic information so we can talk to
-- Tensorboard. You'll be passing this around everywhere.
data SummaryWriter = SummaryWriter { runName      :: Text
                                   , runDirectory :: Text
                                   , runFilename  :: Text
                                   , lastStep     :: IORef Int
                                   }

-- | Make a new summary writer for a given directory and run name. We will also
-- add some information to the run time like the current host and local timestamp.
summaryWriter :: Text -> Text -> IO SummaryWriter
summaryWriter directory runname = do
  now <- getCurrentTime
  tz <- getCurrentTimeZone
  let time = formatTime defaultTimeLocale "%Y-%m-%d_%H:%M:%S" (utcToZonedTime tz now)
  let timeUnix = formatTime defaultTimeLocale "%s" (utcToZonedTime tz now)
  host <- T.pack <$> getHostName
  let name = runname <> "_" <> host <> "_" <> T.pack time
  let location = T.unpack directory </> T.unpack name
  createDirectoryIfMissing True location
  step <- newIORef 0
  pure $ SummaryWriter { runName = name
                       , runDirectory = T.pack location
                       , runFilename = "events.hasktorch." <> T.pack timeUnix <> ".tfevents." <> host
                       , lastStep = step
                       }

-- | Advance the writer to the next step. This helps you keep track of the
-- iteration number without needing a manual counter.
nextStep :: SummaryWriter -> IO SummaryWriter
nextStep r = do
  modifyIORef' (lastStep r) (+ 1)
  pure r

-- | Explicitly set the step counter.
setStep :: SummaryWriter -> Int -> IO SummaryWriter
setStep sw s = do
  writeIORef (lastStep sw) s
  pure sw

-- | Add a scalar to Tensorboard with a given name and value.
addScalar :: Real a => SummaryWriter -> Text -> a -> IO ()
addScalar summary name value =
  writeEvent summary (EventSummary name $ SummaryValue $ fromRational $ toRational value)

-- -- | Add a histogram to Tensorboard, we do some internal work to make sure the
-- -- number of bins and their sizes make sense.
-- addHistogram ::
--   SummaryWriter -> Text -> Torch.Typed.Tensor '( 'D.CPU, 0) dtype shape -> IO ()
-- addHistogram summary name tensor = do
--   bins  <- histogramBinsAuto <$> V.map toDoubleC <$> toVector tensor
--   withSomeSing (fromIntegral bins) (\ n@(SNat :: SNat m) -> do
--     hist_ <- toVector =<< histc @m tensor 0 0
--     let hist = V.map toDoubleC hist_
--     min   <- toDouble <$> (fromScalar =<< Torch.Typed.min tensor)
--     max   <- toDouble <$> (fromScalar =<< Torch.Typed.max tensor)
--     let step = (max-min)/2
--     let binLimits = V.generate bins (\i -> step * fromIntegral i)
--     sum     <- fromScalar =<< Torch.Typed.sumAll tensor
--     sumSq <- fromScalar =<< Torch.Typed.sumAll =<< Torch.Typed.pow tensor =<< toScalar 2
--     writeEvent summary
--       (EventSummary name
--        (SummaryHistogram
--          { shMin = min
--          , shMax = max
--          , shNum = fromIntegral $ Torch.Typed.numel tensor
--          , shSum = toDouble sum
--          , shSumSq = toDouble sumSq
--          , shBucketLimit = binLimits
--          , shBucket = hist })))

-- -- | Add an RGB image tensor to the Tensorboard.
-- addImageRGB :: (KnownNat szw, KnownNat szh, SingI dtype)
--             => SummaryWriter -> Text -> Torch.Typed.Tensor '( 'D.CPU, 0) dtype '[3, szh, szw] -> IO ()
-- addImageRGB summary name image = do
--     (bytes, len) <- V.unsafeToForeignPtr0 <$> rgbTensorToVector image
--     (h,w,c,b) <- pure $ case V.toList $ size image of
--        [3,h,w] -> (h,w,3,bytes)
--        -- TODO Move this to compile time
--        _       -> error $ "Don't know how to display a tensor with these dimensions as an RGB image " ++ show (size image)
--     writeEvent summary
--       (EventSummary name
--        (SummaryImage
--         { siHeight     = fromIntegral h
--         , siWidth      = fromIntegral w
--         , siColorspace = c
--         , siBytes      = BS.PS b 0 len}))

-- -- | Add a greyscale image tensor to the Tensorboard.
-- addImageGrey :: (KnownNat szw, KnownNat szh, SingI dtype)
--              => SummaryWriter -> Text -> Torch.Typed.Tensor '( 'D.CPU, 0) dtype '[1, szh, szw] -> IO ()
-- addImageGrey summary name image = do
--     (bytes, len) <- V.unsafeToForeignPtr0 <$> greyTensorToVector image
--     (h,w,c,b) <- pure $ case V.toList $ size image of
--        [1,h,w] -> (h,w,1,bytes)
--        -- TODO Move this to compile time
--        _       -> error $ "Don't know how to display a tensor with these dimensions as an Greyscale image " ++ show (size image)
--     writeEvent summary
--       (EventSummary name
--        (SummaryImage
--         { siHeight     = fromIntegral h
--         , siWidth      = fromIntegral w
--         , siColorspace = c
--         , siBytes      = BS.PS b 0 len}))

-- -- | Add a tensor that contains a grid of RGB images, like a batch of RGB images
-- -- with the leading dimension being the number of images. Grids are padded with
-- -- the given Size, constructed with the @Size@ datatype. They are padded with a
-- -- given constant, usually 0.
-- addImageRGBGrid ::
--   forall (imagesPerRow :: Nat) (padding :: Nat) (rows :: Nat) (szh :: Nat) (szw :: Nat) (nr :: Nat) (dtype :: TensorType).
--     _
--   => SummaryWriter
--   -> Text
--   -> Size imagesPerRow
--   -> Padding padding
--   -> TensorTyToHs dtype
--   -> Torch.Typed.Tensor '( 'D.CPU, 0) dtype '[nr, 3, szh, szw]
--   -> IO ()
-- addImageRGBGrid summary name shape padding padValue images = do
--   grid <- makeRGBGrid shape padding padValue images
--   addImageRGB summary name grid

-- -- | Add a matplotlib plot to Tensorboard. This uses the Haskell matplotlib
-- -- bindings from the `matplotlib` package,
-- -- `https://github.com/abarbu/matplotlib-haskell`.
-- addPlot :: SummaryWriter -> Text -> M.Matplotlib -> IO ()
-- addPlot summary name mplot = do
--   (fname, handle) <- openTempFile "/tmp/" "plot.png"
--   hClose handle
--   M.file fname mplot
--   (h,w,_) <- fileImageProperties (T.pack fname)
--   withSomeSing (fromIntegral h)
--     (\ szh@(SNat :: SNat szh) ->
--         withSomeSing (fromIntegral w)
--           (\ szw@(SNat :: SNat szw) -> do
--               tensor <- typed @TByte <$> readRGBTensorFromFile @szh @szw (T.pack fname)
--               addImageRGB summary name tensor))
--   fileExists <- doesFileExist fname
--   when fileExists (removeFile fname)

-- -- | Add the trace of a computation to Tensorboard as a graph.
-- -- TODO Support naming nodes, more type information, more device information
-- addGraph :: SummaryWriter -> Text -> ForeignPtr C.CTracingState -> IO ()
-- addGraph summary name trace = do
--   (inputs, nodes, outputs, ret, block) <-
--     withForeignPtr trace C.tracing_state_graph
--   inputs' <- mapM (\i -> do
--                     b <- cbool <$> C.check_value_tensor i
--                     na <- T.pack <$> C.value_name i
--                     if b then do
--                       shape <- C.value_sizes i
--                       dtype <- C.value_scalar_type i
--                       pure (na, Just shape, Just dtype)
--                       else
--                       pure (na, Nothing, Nothing))
--             $ V.toList inputs
--   nodes' <- mapM (\(n,nr) -> do
--                    ins <- C.node_inputs n
--                    outs <- C.node_outputs n
--                    kind <- C.node_kind n
--                    b <- C.node_has_attribute n "value"
--                    attr <- if cbool b then do
--                             ak <- C.node_attribute_kind n "value"
--                             case ak of
--                               C.AttributeKindFloat -> do
--                                 f <- C.node_get_attribute_float n "value"
--                                 pure $ Just $ defMessage & #f .~ coerce f
--                               C.AttributeKindInt -> do
--                                 i <- C.node_get_attribute_int n "value"
--                                 pure $ Just $ defMessage & #i .~ fromIntegral i
--                               C.AttributeKindString -> do
--                                 s <- C.node_get_attribute_string n "value"
--                                 pure $ Just $ defMessage & #s .~ T.encodeUtf8 (T.pack s)
--                               C.AttributeKindTensor -> do
--                                 tensor <- C.node_get_attribute_tensor n "value"
--                                 dtype <- C.getType tensor
--                                 tensor' <- C.contiguous_mm tensor (fromIntegral $ fromEnum C.MemoryFormatContiguous)
--                                 ptr <- newForeignPtr_ =<< castPtr <$> C.data_ptr tensor'
--                                 len <- fromIntegral <$> Torch.Typed.numel tensor'
--                                 let get :: V.Storable a => [a]
--                                     get = V.toList $ V.unsafeFromForeignPtr0 (castForeignPtr ptr) len
--                                 pure $ Just $ defMessage & #tensor .~
--                                   case dtype of
--                                     C.ScalarTypeByte ->
--                                       defMessage & #dtype .~ PS.DT_UINT8
--                                                  & #intVal .~ map (fromIntegral :: Word8 -> Int32) get
--                                     C.ScalarTypeChar ->
--                                       defMessage & #dtype .~ PS.DT_INT8
--                                                  & #intVal .~ map (fromIntegral :: Int8 -> Int32) get
--                                     C.ScalarTypeShort ->
--                                       defMessage & #dtype .~ PS.DT_INT16
--                                                  & #intVal .~ map (fromIntegral :: Int16 -> Int32) get
--                                     C.ScalarTypeInt ->
--                                       defMessage & #dtype .~ PS.DT_INT32
--                                                  & #intVal .~ get
--                                     C.ScalarTypeLong ->
--                                       defMessage & #dtype .~ PS.DT_INT64
--                                                  & #int64Val .~ get
--                                     C.ScalarTypeHalf ->
--                                       defMessage & #dtype .~ PS.DT_HALF
--                                                  & #halfVal .~ get
--                                     C.ScalarTypeFloat ->
--                                       defMessage & #dtype .~ PS.DT_FLOAT
--                                                  & #floatVal .~ get
--                                     C.ScalarTypeDouble ->
--                                       defMessage & #dtype .~ PS.DT_DOUBLE
--                                                  & #doubleVal .~ get
--                                     C.ScalarTypeUndefined ->
--                                       defMessage & #dtype .~ PS.DT_INVALID
--                               _ -> pure Nothing
--                      else pure Nothing
--                    ins' <- mapM (\v -> do
--                                 n <- C.value_name v
--                                 t <- C.value_type v
--                                 ct <- convertCType t
--                                 pure $ TempGraphValue v n t ct)
--                           $ V.toList ins
--                    outs' <- mapM (\v -> do
--                                 n <- C.value_name v
--                                 t <- C.value_type v
--                                 ct <- convertCType t
--                                 pure $ TempGraphValue v n t ct)
--                            $ V.toList outs
--                    pure (n, TempGraphNode nr kind ins' outs' attr)) $ zip (V.toList nodes) [0..]
--   let nodesMap = M.fromList nodes'
--   let insToNodes  = M.fromListWith (++)
--                   $ concatMap (\(n, TempGraphNode nr kinds ins outs attr) ->
--                              map (\(TempGraphValue v _ _ _) -> (v,[n])) ins) nodes'
--   let outsToNodes = M.fromListWith (++)
--                   $ concatMap (\(n, TempGraphNode nr kinds ins outs attr) ->
--                              map (\(TempGraphValue v _ _ _) -> (v,[n])) outs) nodes'
--   let nodeName nr kind = T.replace "prim::" "" $ T.replace "aten::" "" $ T.pack kind <> "_"<> T.pack (show nr)
--   writeEvent summary
--     (EventGraph name
--       (map (\(name, size, dtype) ->
--                case (size, dtype) of
--                  (Just shape, Just t) ->
--                    defMessage & #name .~ "input_" <> name
--                               & #op .~ "Input"
--                               & #attr .~ M.fromList [("_output_shapes",
--                                                       defMessage & #list .~
--                                                        (defMessage & #shape .~
--                                                          [defMessage & #dim .~
--                                                           map (\x -> defMessage & #size .~ x) (V.toList shape)]))]
--                  (_, _) ->
--                    defMessage & #name .~ "input_" <> name
--                               & #op .~ "Input")
--          inputs'
--        ++
--        map (\(node,(TempGraphNode nr kind ins outs attr)) ->
--                defMessage & #name .~ nodeName nr kind
--                           & #op .~ T.pack kind
--                           & #device .~ T.pack ""
--                           & #input .~ map (\(TempGraphValue v n t ct) ->
--                                                 case M.lookup v outsToNodes of
--                                                   Nothing -> T.pack ("input_" ++ n)
--                                                   Just [outNode] -> case M.lookup outNode nodesMap of
--                                                                    Just (TempGraphNode nr kind _ _ _) -> nodeName nr kind
--                                                                    Nothing                            -> error "Can't find node") ins
--                           & #attr .~ M.fromList([("_output_shapes",
--                                                    defMessage
--                                                    & #list .~ (defMessage
--                                                                & #shape .~
--                                                                   (map (\(TempGraphValue v n t ct) ->
--                                                                            case ct of
--                                                                              TraceTypeSimple _ _ ->
--                                                                                defMessage & #dim .~ [defMessage & #size .~ 1]
--                                                                              TraceTypeTensor _ (Just shape) _ _ ->
--                                                                                defMessage & #dim .~ (map (\x -> defMessage & #size .~ x)
--                                                                                                          (V.toList shape))
--                                                                              _ -> defMessage)
--                                                                     outs)))]
--                                                 ++ (case attr of
--                                                      Nothing -> []
--                                                      Just a  -> [("value",a)])))
--            nodes'))

-- -- | A convenience function, add the summary statistics and histograms for a
-- -- Tensor.
-- addTensorSummary :: (Num (TensorTyToHs dtype), Num (TensorTyToHsC dtype),
--                      V.Storable (TensorTyToHs dtype), V.Storable (TensorTyToHsC dtype),
--                      SingI shape, SingI dtype) =>
--                    SummaryWriter -> Text -> Torch.Typed.Tensor '( 'D.CPU, 0) dtype shape -> IO ()
-- addTensorSummary summary name tensor = do
--   mean   <- fromScalar =<< Torch.Typed.mean tensor
--   median <- fromScalar =<< Torch.Typed.median tensor
--   var    <- fromScalar =<< (Torch.Typed.UnsafeMkTensor . F.var . Torch.Typed.toDynamic $ tensor)
--   min    <- fromScalar =<< Torch.Typed.min tensor
--   max    <- fromScalar =<< Torch.Typed.max tensor
--   addScalar    summary ("summaries/" <> name <> "/mean")   $ toDouble mean
--   addScalar    summary ("summaries/" <> name <> "/median") $ toDouble median
--   addScalar    summary ("summaries/" <> name <> "/stddev") $ Prelude.sqrt $ toDouble var
--   addScalar    summary ("summaries/" <> name <> "/min")    $ toDouble min
--   addScalar    summary ("summaries/" <> name <> "/max")    $ toDouble max
--   addHistogram summary ("summaries/" <> name <> "/histogram") tensor

-- * Basic heuristics

-- | The numpy auto histogram mode
histogramBinsAuto :: Vector Double -> Int
histogramBinsAuto x | V.length x < 1000 = histogramBinsSturges x
                    | otherwise = histogramBinsFD x

-- | The Freedman-Diaconis histogram bin estimator.
-- bins = 2 \frac{IQR}{n^{1/3}}
histogramBinsFD :: Vector Double -> Int
histogramBinsFD v =
  ceiling $ (V.maximum v - V.minimum v) /
               (2 * (S.weightedAvg 3 4 v - S.weightedAvg 1 4 v)
                  / (fromIntegral (V.length v) ** (1/3)))

-- | Sturges's histogram bin estimator.
histogramBinsSturges :: Vector Double -> Int
histogramBinsSturges v = Prelude.round $ logBase 2 (fromIntegral $ V.length v) + 1

-- * Internal

-- | This is internal, you never need to use it manually. It's what we fill out
-- with the various functions in this module and then serialize for Tensorboard.
data Summary = SummaryValue Double
             | SummaryHistogram { shMin         :: Double
                                , shMax         :: Double
                                , shNum         :: Double
                                , shSum         :: Double
                                , shSumSq       :: Double
                                , shBucketLimit :: Vector Double
                                , shBucket      :: Vector Double }
             | SummaryImage { siHeight     :: Int32
                            , siWidth      :: Int32
                            , siColorspace :: Int32
                            , siBytes      :: BS.ByteString }

-- | This is internal.
data Node = Node { nName   :: Text
                 , nOp     :: Text
                 , nInput  :: [Text]
                 , nDevice :: Text
                 }

-- | Internal. Used to capture the various kinds of entities that Tensorboard
-- can understand.
data Event = EventVersion
           | EventGraph Text [PG.NodeDef]
           | EventSummary Text Summary
           | EventMessage LogLevel Text
           | EventSessionLog
           | EventMetadata
           | EventMetaGraph

-- -- | Internal
-- data TempGraphValue = TempGraphValue (Ptr C.CJitValue) String (Ptr C.CType) TraceType
--   deriving (Show)
-- -- | Internal
-- data TempGraphNode  = TempGraphNode Int String [TempGraphValue] [TempGraphValue] (Maybe PA.AttrValue)
--   deriving (Show)

-- | Internal
eventToProto :: Int -> Event -> IO PE.Event
eventToProto step event = do
  (time :: Double) <- realToFrac <$> getPOSIXTime
  let d = case event of
            EventVersion -> defMessage & #fileVersion .~ "brain.Event:2"
            EventMessage lvl txt ->
              defMessage & #logMessage .~ (defMessage & #level .~ (case lvl of
                                                        LevelDebug         -> PE.LogMessage'DEBUG
                                                        LevelInfo          -> PE.LogMessage'INFO
                                                        LevelWarn          -> PE.LogMessage'WARN
                                                        LevelError         -> PE.LogMessage'ERROR
                                                        LevelOther "fatal" -> PE.LogMessage'FATAL
                                                        LevelOther _       -> PE.LogMessage'UNKNOWN)
                                        & #message .~ txt)
            EventSummary tag (SummaryValue val) ->
              defMessage & #summary .~
              (defMessage & #value .~
                [defMessage & #tag .~ tag
                -- TODO Metadata!
                     & #simpleValue .~ double2Float val])
            EventSummary tag sh@SummaryHistogram {} ->
              defMessage & #summary .~
              (defMessage & #value .~
                [defMessage & #tag .~ tag
                -- TODO Metadata!
                     & #histo .~
                     (defMessage
                          & #min .~ shMin sh
                          & #max .~ shMax sh
                          & #num .~ shNum sh
                          & #sum   .~ shSum sh
                          & #sumSquares .~ shSumSq sh
                          & #bucketLimit .~ V.toList (shBucketLimit sh)
                          & #bucket .~ V.toList (shBucket sh))])
            EventSummary tag sh@SummaryImage{} ->
              defMessage & #summary .~
              (defMessage & #value .~
                [defMessage & #tag .~ tag
                -- TODO Metadata!
                     & #image .~
                     (defMessage
                          & #height             .~ siHeight sh
                          & #width              .~ siWidth sh
                          & #colorspace         .~ siColorspace sh
                          & #encodedImageString .~ siBytes sh)])
            EventGraph tag nodes ->
              defMessage & #graphDef .~
                encodeMessage ((defMessage :: PG.GraphDef)
                                & #versions .~ (defMessage & #producer .~ 1
                                                           & #minConsumer .~ 1
                                                           & #badConsumers .~ [])
                                & #version .~ 3
                                & #node .~ nodes)
  pure $ d & #wallTime .~ time
           & #step .~ fromIntegral step

-- | Internal
encodeEvent :: PE.Event -> B.ByteString
encodeEvent ev = BL.toStrict $ runPut $ do
  putByteString len
  putWord32le (maskedCRC32 len)
  putByteString msg
  putWord32le (maskedCRC32 msg)
  where msg = encodeMessage ev
        len = BL.toStrict $ runPut $ putWord64le (fromIntegral $ B.length msg)

-- | Internal
writeEvent :: SummaryWriter -> Event -> IO ()
writeEvent r e = do
  step <- readIORef $ lastStep r
  let fname = T.unpack (runDirectory r) </> (T.unpack $ runFilename r)
  b <- doesFileExist fname
  unless b $ do
    bs <- encodeEvent <$> eventToProto step EventVersion
    B.writeFile fname bs
  bs <- encodeEvent <$> eventToProto step e
  B.appendFile fname bs

-- | Internal
maskedCRC32 :: BS.ByteString -> Word32
maskedCRC32 bs = (crc32c bs `rotateR` 15) + 0xa282ead8
