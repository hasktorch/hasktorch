-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Indef.Dynamic.NN
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
--
-- NN backpack API which is _not recommended to be used directly_. This package
-- is just a wrapper around the backpack signatures of Torch's C-based NN
-- library.
--
-- Instead, use 'Torch.Indef.Static.NN' which includes an AD abstraction,
-- simple forward- and backward- functions, and staticly-typed dimension-level
-- checking.
--
-- This library will, over time, iterate into type-safe code -- but for the
-- moment we hoist the safety into the dependent types, which is faster to
-- iterate with and is semantically clearer for development (the errors take a
-- bit of getting used to).
-------------------------------------------------------------------------------
module Torch.Indef.Dynamic.NN
  ( module X
  , _abs_updateOutput
  , _abs_updateGradInput
  , _gatedLinear_updateOutput
  , _gatedLinear_updateGradInput
  , _hardTanh_updateOutput
  , _hardTanh_updateGradInput
  , _im2Col_updateOutput
  , _im2Col_updateGradInput
  , _col2Im_updateOutput
  , _col2Im_updateGradInput
  , _gRUFused_updateOutput
  , _gRUFused_updateGradInput
  , _lSTMFused_updateOutput
  , _lSTMFused_updateGradInput
  , _logSigmoid_updateOutput
  , _logSigmoid_updateGradInput
  , _logSoftMax_updateOutput
  , _logSoftMax_updateGradInput
  , _sigmoid_updateOutput
  , _sigmoid_updateGradInput
  , _softMax_updateOutput
  , _softMax_updateGradInput
  , _softPlus_updateOutput
  , _softPlus_updateGradInput
  , _softShrink_updateOutput
  , _softShrink_updateGradInput
  , _sparseLinear_updateOutput
  , _sparseLinear_accGradParameters
  , _sparseLinear_zeroGradParameters
  , _sparseLinear_updateParameters
  , _sparseLinear_legacyUpdateOutput
  , _sparseLinear_legacyAccGradParameters
  , _sqrt_updateOutput
  , _sqrt_updateGradInput
  , _square_updateOutput
  , _square_updateGradInput
  , _tanh_updateOutput
  , _tanh_updateGradInput
  , _temporalConvolution_updateOutput
  , _temporalConvolution_updateGradInput
  , _temporalConvolution_accGradParameters
  , _temporalRowConvolution_updateOutput
  , _temporalRowConvolution_updateGradInput
  , _temporalRowConvolution_accGradParameters
  , _temporalUpSamplingNearest_updateOutput
  , _temporalUpSamplingNearest_updateGradInput
  , _temporalUpSamplingLinear_updateOutput
  , _temporalUpSamplingLinear_updateGradInput
  , _batchNormalization_updateOutput
  , _batchNormalization_backward
  , spatialConvolutionMM_updateOutput
  , _spatialConvolutionMM_updateGradInput
  , _spatialConvolutionMM_accGradParameters
  , _spatialConvolutionLocal_updateOutput
  , _spatialConvolutionLocal_updateGradInput
  , _spatialConvolutionLocal_accGradParameters
  , _spatialFullConvolution_updateOutput
  , _spatialFullConvolution_updateGradInput
  , _spatialFullConvolution_accGradParameters
  , _spatialDilatedConvolution_updateOutput
  , _spatialDilatedConvolution_updateGradInput
  , _spatialDilatedConvolution_accGradParameters
  , _spatialFullDilatedConvolution_updateOutput
  , _spatialFullDilatedConvolution_updateGradInput
  , _spatialFullDilatedConvolution_accGradParameters
  , _spatialSubSampling_updateOutput
  , _spatialSubSampling_updateGradInput
  , _spatialSubSampling_accGradParameters
  , _spatialUpSamplingNearest_updateOutput
  , _spatialUpSamplingNearest_updateGradInput
  , _spatialUpSamplingBilinear_updateOutput
  , _spatialUpSamplingBilinear_updateGradInput
  , _spatialGridSamplerBilinear_updateOutput
  , _spatialGridSamplerBilinear_updateGradInput
  , _volumetricGridSamplerBilinear_updateOutput
  , _volumetricGridSamplerBilinear_updateGradInput
  , _volumetricConvolution_updateOutput
  , _volumetricConvolution_updateGradInput
  , _volumetricConvolution_accGradParameters
  , _volumetricFullConvolution_updateOutput
  , _volumetricFullConvolution_updateGradInput
  , _volumetricFullConvolution_accGradParameters
  , _volumetricDilatedConvolution_updateOutput
  , _volumetricDilatedConvolution_updateGradInput
  , _volumetricDilatedConvolution_accGradParameters
  , _volumetricFullDilatedConvolution_updateOutput
  , _volumetricFullDilatedConvolution_updateGradInput
  , _volumetricFullDilatedConvolution_accGradParameters
  , _spatialReflectionPadding_updateOutput
  , _spatialReflectionPadding_updateGradInput
  , _spatialReplicationPadding_updateOutput
  , _spatialReplicationPadding_updateGradInput
  , _volumetricReplicationPadding_updateOutput
  , _volumetricReplicationPadding_updateGradInput
  , _volumetricUpSamplingNearest_updateOutput
  , _volumetricUpSamplingNearest_updateGradInput
  , _volumetricUpSamplingTrilinear_updateOutput
  , _volumetricUpSamplingTrilinear_updateGradInput
  , _temporalReflectionPadding_updateOutput
  , _temporalReflectionPadding_updateGradInput
  , _temporalReplicationPadding_updateOutput
  , _temporalReplicationPadding_updateGradInput
  ) where


import Foreign.C.Types
import Torch.Sig.Types.NN
import Torch.Indef.Dynamic.Tensor -- (empty, new)
import Torch.Indef.Dynamic.NN.Activation as X
import Torch.Indef.Dynamic.NN.Pooling as X
import qualified Torch.Sig.NN as Sig

import Torch.Indef.Types

-- -- | indexLinear forward pass
-- --
-- -- FIXME: reintroduce these
-- _indexLinear_updateOutput
--   :: IndexDynamic
--   -> Integer
--   -> Dynamic
--   -> IndexDynamic
--   -> IndexDynamic
--   -> Dynamic
--   -> Dynamic
--   -> Dynamic
--   -> Dynamic
--   -> Int
--   -> IO ()
-- _indexLinear_updateOutput i0 l t0 i1 i2 t1 t2 t3 t4 i = undefined
  -- Sig.c_IndexLinear_updateOutput

-- c_IndexLinear_accGradParameters :: IndexDynamic -> CLLong -> Dynamic -> IndexDynamic -> IndexDynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> CDouble -> CDouble -> IO ()
-- c_IndexLinear_accUpdateGradParameters :: IndexDynamic -> CLLong -> Dynamic -> IndexDynamic -> IndexDynamic -> Dynamic -> Dynamic -> Dynamic -> CDouble -> CDouble -> IO ()
-- c_IndexLinear_updateParameters :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> IndexDynamic -> IndexDynamic -> CLLong -> CDouble -> CDouble -> IO ()

-- c_LookupTable_accGradParameters :: Ptr CIndexTensor -> Ptr CDoubleTensor -> Ptr CDoubleTensor -> Ptr CIndexTensor -> Ptr CIndexTensor -> Ptr CIndexTensor -> CBool -> CInt -> CDouble -> IO ()
-- c_LookupTable_renorm :: Ptr CIndexTensor -> Ptr CDoubleTensor -> CDouble -> CDouble -> IO ()

-- | abs forward
_abs_updateOutput :: Dynamic -> Dynamic -> IO ()
_abs_updateOutput t0 t1 = with2DynamicState t0 t1 $ \s' t0' t1' ->
  Sig.c_Abs_updateOutput s' t0' t1'

-- | abs backward
_abs_updateGradInput :: Dynamic -> Dynamic -> Dynamic -> IO ()
_abs_updateGradInput t0 t1 t2 = with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
  Sig.c_Abs_updateGradInput s' t0' t1' t2'

_gatedLinear_updateOutput     :: Dynamic -> Dynamic -> Int -> IO ()
-- | gatedLinear backward-update (updates the layer and bias tensors)
_gatedLinear_updateGradInput  :: Dynamic -> Dynamic -> Dynamic -> Int -> IO ()
-- | hardTanh forward pass (updates the output tensor)
_hardTanh_updateOutput        :: Dynamic -> Dynamic -> Double -> Double -> Bool -> IO ()
-- | hardTanh backward-update (updates the layer and bias tensors)
_hardTanh_updateGradInput     :: Dynamic -> Dynamic -> Dynamic -> Double -> Double -> Bool -> IO ()
-- | im2Col forward pass (updates the output tensor)
_im2Col_updateOutput          :: Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
-- | im2Col backward-update (updates the layer and bias tensors)
_im2Col_updateGradInput       :: Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()

-- | col2Im forward pass (updates the output tensor)
_col2Im_updateOutput
  :: Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
_col2Im_updateOutput t0 t1 a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 =
  with2DynamicState t0 t1 $ \s' t0' t1' ->
    Sig.c_Col2Im_updateOutput s' t0' t1'
      (fromIntegral a0)
      (fromIntegral a1)
      (fromIntegral a2)
      (fromIntegral a3)
      (fromIntegral a4)
      (fromIntegral a5)
      (fromIntegral a6)
      (fromIntegral a7)
      (fromIntegral a8)
      (fromIntegral a9)


-- | col2Im backward-update (updates the layer and bias tensors)
_col2Im_updateGradInput :: Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
_col2Im_updateGradInput t0 t1 a0 a1 a2 a3 a4 a5 a6 a7 =
  with2DynamicState t0 t1 $ \s' t0' t1' ->
    Sig.c_Col2Im_updateGradInput s' t0' t1'
      (fromIntegral a0)
      (fromIntegral a1)
      (fromIntegral a2)
      (fromIntegral a3)
      (fromIntegral a4)
      (fromIntegral a5)
      (fromIntegral a6)
      (fromIntegral a7)


-- | gRUFused forward pass (updates the output tensor)
_gRUFused_updateOutput        :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> IO ()
-- | gRUFused backward-update (updates the layer and bias tensors)
_gRUFused_updateGradInput     :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> IO ()
-- | lSTMFused forward pass (updates the output tensor)
_lSTMFused_updateOutput       :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> IO ()
-- | lSTMFused backward-update (updates the layer and bias tensors)
_lSTMFused_updateGradInput    :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> IO ()
-- | logSigmoid forward pass (updates the output tensor)
_logSigmoid_updateOutput      :: Dynamic -> Dynamic -> Dynamic -> IO ()
-- | logSigmoid backward-update (updates the layer and bias tensors)
_logSigmoid_updateGradInput   :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> IO ()
-- | logSoftMax forward pass (updates the output tensor)
_logSoftMax_updateOutput      :: Dynamic -> Dynamic -> Integer -> IO ()
-- | logSoftMax backward-update (updates the layer and bias tensors)
_logSoftMax_updateGradInput   :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Integer -> IO ()

_sigmoid_updateOutput     :: Dynamic -> Dynamic -> IO ()
-- | sigmoid backward-update (updates the layer and bias tensors)
_sigmoid_updateGradInput  :: Dynamic -> Dynamic -> Dynamic -> IO ()

-- | softMax forward pass (updates the output tensor)
_softMax_updateOutput       :: Dynamic -> Dynamic -> Integer -> IO ()
-- | softMax backward-update (updates the layer and bias tensors)
_softMax_updateGradInput    :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Integer -> IO ()
-- | softPlus forward pass (updates the output tensor)
_softPlus_updateOutput      :: Dynamic -> Dynamic -> Double -> Double -> IO ()
-- | softPlus backward-update (updates the layer and bias tensors)
_softPlus_updateGradInput   :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Double -> Double -> IO ()
-- | softShrink forward pass (updates the output tensor)
_softShrink_updateOutput    :: Dynamic -> Dynamic -> Double -> IO ()
-- | softShrink backward-update (updates the layer and bias tensors)
_softShrink_updateGradInput :: Dynamic -> Dynamic -> Dynamic -> Double -> IO ()
-- | sparseLinear forward pass (updates the output tensor)
_sparseLinear_updateOutput             :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> IO ()
-- | sparseLinear backward-update (updates the layer and bias tensors). Called 'accGradParameters' in C to indicate accumulating the gradient parameters.
_sparseLinear_accGradParameters        :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Double -> Double -> IO ()
-- | sparseLinear zeroGradParameters
_sparseLinear_zeroGradParameters       :: Dynamic -> Dynamic -> Dynamic -> IO ()
-- | sparseLinear updateParameters
_sparseLinear_updateParameters         :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Double -> IO ()
-- | sparseLinear legacyUpdateOutput
_sparseLinear_legacyUpdateOutput       :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> IO ()
-- | sparseLinear legacyAccGradParameters
_sparseLinear_legacyAccGradParameters  :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Double -> Double -> IO ()
-- | sqrt forward pass (updates the output tensor)
_sqrt_updateOutput         :: Dynamic -> Dynamic -> Double -> IO ()
-- | sqrt backward-update (updates the layer and bias tensors)
_sqrt_updateGradInput      :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> IO ()
-- | square forward pass (updates the output tensor)
_square_updateOutput       :: Dynamic -> Dynamic -> IO ()
-- | square backward-update (updates the layer and bias tensors)
_square_updateGradInput    :: Dynamic -> Dynamic -> Dynamic -> IO ()
-- | tanh forward pass (updates the output tensor)
_tanh_updateOutput         :: Dynamic -> Dynamic -> IO ()
-- | tanh backward-update (updates the layer and bias tensors)
_tanh_updateGradInput      :: Dynamic -> Dynamic -> Dynamic -> IO ()

-- | temporalConvolution forward pass (updates the output tensor)
_temporalConvolution_updateOutput
  :: Dynamic     -- ^ input
  -> Dynamic     -- ^ output -- this is the mutated return value
  -> Dynamic     -- ^ 2d weight tensor
  -> Dynamic     -- ^ 1d bias tensor
  -> Int         -- ^ kernel width
  -> Int         -- ^ step size
  -> Int         -- ^ feature size
  -> Int         -- ^ output size
  -> IO ()
_temporalConvolution_updateOutput t0 t1 t2 t3 i0 i1 i2 i3 = 
  with2DynamicState t0 t1 $ \s' t0' t1' ->
    with2DynamicState t2 t3 $ \_  t2' t3' ->
      Sig.c_TemporalConvolution_updateOutput s' t0' t1' t2' t3'
        (fromIntegral i0) (fromIntegral i1)
        (fromIntegral i2) (fromIntegral i3)

-- | temporalConvolution backward-update (updates the layer and bias tensors)
_temporalConvolution_updateGradInput
  :: Dynamic     -- ^ input
  -> Dynamic     -- ^ grad output
  -> Dynamic     -- ^ grad input -- this is the mutated return value
  -> Dynamic     -- ^ weights
  -> Int         -- ^ kernel width
  -> Int         -- ^ step size
  -> IO ()
_temporalConvolution_updateGradInput t0 t1 t2 t3 i0 i1 =
  with2DynamicState t0 t1 $ \s' t0' t1' ->
    with2DynamicState t2 t3 $ \_  t2' t3' ->
      Sig.c_TemporalConvolution_updateGradInput s' t0' t1' t2' t3'
        (fromIntegral i0) (fromIntegral i1)


-- | temporalConvolution backward-update (updates the layer and bias tensors). Called 'accGradParameters' in C to indicate accumulating the gradient parameters.
_temporalConvolution_accGradParameters
  :: Dynamic   -- ^ input
  -> Dynamic   -- ^ grad output
  -> Dynamic   -- ^ grad weight -- this is a mutated argument and torch will accumulate this gradient
  -> Dynamic   -- ^ grad bias -- this is a mutated argument and torch will accumulate this gradient
  -> Int       -- ^ kernel width
  -> Int       -- ^ step size
  -> Double    -- ^ scale
  -> IO ()
_temporalConvolution_accGradParameters t0 t1 t2 t3 i0 i1 scale =
  with2DynamicState t0 t1 $ \s' t0' t1' ->
    with2DynamicState t2 t3 $ \_  t2' t3' ->
      Sig.c_TemporalConvolution_accGradParameters s' t0' t1' t2' t3'
        (fromIntegral i0) (fromIntegral i1)
        (realToFrac scale)

-- | temporalRowConvolution forward pass (updates the output tensor)
_temporalRowConvolution_updateOutput                :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Int -> Int -> Int -> Bool -> IO ()
-- | temporalRowConvolution backward-update (updates the layer and bias tensors)
_temporalRowConvolution_updateGradInput             :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Int -> Int -> Int -> Bool -> IO ()
-- | temporalRowConvolution backward-update (updates the layer and bias tensors). Called 'accGradParameters' in C to indicate accumulating the gradient parameters.
_temporalRowConvolution_accGradParameters           :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Int -> Int -> Int -> Bool -> Double -> IO ()
-- | temporalUpSamplingNearest forward pass (updates the output tensor)
_temporalUpSamplingNearest_updateOutput             :: Dynamic -> Dynamic -> Int -> IO ()
-- | temporalUpSamplingNearest backward-update (updates the layer and bias tensors)
_temporalUpSamplingNearest_updateGradInput          :: Dynamic -> Dynamic -> Dynamic -> Int -> IO ()
-- | temporalUpSamplingLinear forward pass (updates the output tensor)
_temporalUpSamplingLinear_updateOutput              :: Dynamic -> Dynamic -> Int -> IO ()
-- | temporalUpSamplingLinear backward-update (updates the layer and bias tensors)
_temporalUpSamplingLinear_updateGradInput           :: Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> IO ()
-- | batchNormalization forward pass (updates the output tensor)
_batchNormalization_updateOutput                    :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Bool -> Double -> Double -> IO ()
-- | batchNormalization backward
_batchNormalization_backward                        :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Bool -> Double -> Double -> IO ()

-- | spatialConvolutionMM forward pass
spatialConvolutionMM_updateOutput
  :: Dynamic    -- ^ input
  -> Dynamic    -- ^ 3D weight tensor (connTable:size(1) x kH x kW) 
  -> Dynamic    -- ^ 1D bias tensor (nOutputPlane)
  -> (Int, Int) -- ^ (kW, kH) kernel height and width
  -> (Int, Int) -- ^ (dW, dH) step of the convolution in width and height dimensions. C-default is 1 for both.
  -> (Int, Int) -- ^ (pW, pH) zero padding to the input plane for width and height. (kW-1)/2 is often used. C-default is 0 for both.
  -> IO Dynamic -- ^ output
spatialConvolutionMM_updateOutput inp weight bias (kW, kH) (dW, dH) (pW, pH) = do
  -- these are temporary placeholders and do not require dimensions as internal torch code will resize and fill them. See:
  -- https://github.com/zdevito/ATen/blob/682cb389db5a318539ff03f031bf896a43a71b13/aten/src/THCUNN/generic/SpatialConvolutionMM.cu#L141
  --
  -- TODO: someone needs to verify that this is all above-board and we aren't missing out on some optimization tricks.
  columns <- empty   -- temporary columns
  ones    <- empty   -- buffer of ones for bias accumulation

  -- This one as well:
  out     <- empty   -- output
  with3DynamicState inp out weight $ \s' inp' out' weight' ->
   with3DynamicState bias columns ones $ \_ bias' columns' ones' ->
    Sig.c_SpatialConvolutionMM_updateOutput s' inp' out' weight' bias' columns' ones'
      (fromIntegral kW) (fromIntegral kH)
      (fromIntegral dW) (fromIntegral dH)
      (fromIntegral pW) (fromIntegral pH)
  pure out

-- | spatialConvolutionMM backward-update (updates the layer and bias tensors)
_spatialConvolutionMM_updateGradInput
  :: Dynamic    -- ^ input
  -> Dynamic    -- ^ gradOutput
  -> Dynamic    -- ^ gradInput
  -> Dynamic    -- ^ weight
  -> Dynamic    -- ^ columns
  -> Dynamic    -- ^ ones
  -> (Int, Int) -- ^ (kW, kH) kernel height and width
  -> (Int, Int) -- ^ (dW, dH) step of the convolution in width and height dimensions
  -> (Int, Int) -- ^ (pW, pH) zero padding to the input plane for width and height. (kW-1)/2 is often used.
  -> IO ()
_spatialConvolutionMM_updateGradInput inp gout gin w columns ones (kW, kH) (dW, dH) (pW, pH) = do
  -- columns and ones are reshaped (and I'm not even sure ones is used):
  -- https://github.com/zdevito/ATen/blob/682cb389db5a318539ff03f031bf896a43a71b13/aten/src/THCUNN/generic/SpatialConvolutionMM.cu#L294
  -- shape inp >>= print
  -- (gin, columns, ones) <- (,,) <$> new (dim :: Dim '[2,2]) <*> newWithTensor inp <*> new (dim :: Dim '[1,1])
  with3DynamicState inp gout gin $ \s' inp' gout' gin' ->
   with3DynamicState w columns ones $ \_ w' columns' ones' ->
    Sig.c_SpatialConvolutionMM_updateGradInput s' inp' gout' gin' w' columns' ones'
      (fromIntegral kW) (fromIntegral kH)
      (fromIntegral dW) (fromIntegral dH)
      (fromIntegral pW) (fromIntegral pH)
  -- pure gin
  -- asStatic <$> Dynamic.spatialConvolutionMM_updateGradInput
  --   (asDynamic inp) (asDynamic gout)
  --   (asDynamic (weights conv))
  --   (kernel2d conv)
  --   (param2d step)
  --   (param2d pad)


-- | spatialConvolutionMM backward-update (updates the layer and bias tensors). Called 'accGradParameters' in C to indicate accumulating the gradient parameters.
_spatialConvolutionMM_accGradParameters
  :: Dynamic    -- ^ input
  -> Dynamic    -- ^ gradOutput
  -> Dynamic    -- ^ gradWeight
  -> Dynamic    -- ^ gradBias
  -> Dynamic    -- ^ finput/columns <<- required. This can be NULL in C if gradWeight is NULL.
  -> Dynamic    -- ^ ones
  -> (Int, Int) -- ^ (kW, kH) kernel height and width
  -> (Int, Int) -- ^ (dW, dH) step of the convolution in width and height dimensions
  -> (Int, Int) -- ^ (pW, pH) zero padding to the input plane for width and height. (kW-1)/2 is often used.
  -> Double
  -> IO ()
_spatialConvolutionMM_accGradParameters inp gout gweight gbias finput fgradInput (kW, kH) (dW, dH) (pW, pH) scale = do
  with3DynamicState inp gout gweight $ \s' inp' gout' gweight' ->
   with3DynamicState gbias finput fgradInput $ \_ gbias' finput' fgradInput' ->
    Sig.c_SpatialConvolutionMM_accGradParameters s' inp' gout' gweight' gbias' finput' fgradInput'
      (fromIntegral kW) (fromIntegral kH)
      (fromIntegral dW) (fromIntegral dH)
      (fromIntegral pW) (fromIntegral pH)
      (realToFrac scale)


-- | spatialConvolutionLocal forward pass (updates the output tensor)
_spatialConvolutionLocal_updateOutput               :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> Int -> Int -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()
-- | spatialConvolutionLocal backward-update (updates the layer and bias tensors)
_spatialConvolutionLocal_updateGradInput            :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> Int -> Int -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()
-- | spatialConvolutionLocal backward-update (updates the layer and bias tensors). Called 'accGradParameters' in C to indicate accumulating the gradient parameters.
_spatialConvolutionLocal_accGradParameters          :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> Int -> Int -> CLLong -> CLLong -> CLLong -> CLLong -> Double -> IO ()
-- | spatialFullConvolution forward pass (updates the output tensor)
_spatialFullConvolution_updateOutput                :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
-- | spatialFullConvolution backward-update (updates the layer and bias tensors)
_spatialFullConvolution_updateGradInput             :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
-- | spatialFullConvolution backward-update (updates the layer and bias tensors). Called 'accGradParameters' in C to indicate accumulating the gradient parameters.
_spatialFullConvolution_accGradParameters           :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Double -> IO ()
-- | spatialDilatedConvolution forward pass (updates the output tensor)
_spatialDilatedConvolution_updateOutput             :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
-- | spatialDilatedConvolution backward-update (updates the layer and bias tensors)
_spatialDilatedConvolution_updateGradInput          :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
-- | spatialDilatedConvolution backward-update (updates the layer and bias tensors). Called 'accGradParameters' in C to indicate accumulating the gradient parameters.
_spatialDilatedConvolution_accGradParameters        :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Double -> IO ()
-- | spatialFullDilatedConvolution forward pass (updates the output tensor)
_spatialFullDilatedConvolution_updateOutput         :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
-- | spatialFullDilatedConvolution backward-update (updates the layer and bias tensors)
_spatialFullDilatedConvolution_updateGradInput      :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
-- | spatialFullDilatedConvolution backward-update (updates the layer and bias tensors). Called 'accGradParameters' in C to indicate accumulating the gradient parameters.
_spatialFullDilatedConvolution_accGradParameters    :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Double -> IO ()
-- | spatialSubSampling forward pass (updates the output tensor)
_spatialSubSampling_updateOutput                    :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> IO ()
-- | spatialSubSampling backward-update (updates the layer and bias tensors)
_spatialSubSampling_updateGradInput                 :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> IO ()
-- | spatialSubSampling backward-update (updates the layer and bias tensors). Called 'accGradParameters' in C to indicate accumulating the gradient parameters.
_spatialSubSampling_accGradParameters               :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> Double -> IO ()
-- | spatialUpSamplingNearest forward pass (updates the output tensor)
_spatialUpSamplingNearest_updateOutput              :: Dynamic -> Dynamic -> Int -> IO ()
-- | spatialUpSamplingNearest backward-update (updates the layer and bias tensors)
_spatialUpSamplingNearest_updateGradInput           :: Dynamic -> Dynamic -> Dynamic -> Int -> IO ()
-- | spatialUpSamplingBilinear forward pass (updates the output tensor)
_spatialUpSamplingBilinear_updateOutput             :: Dynamic -> Dynamic -> Int -> Int -> IO ()
-- | spatialUpSamplingBilinear backward-update (updates the layer and bias tensors)
_spatialUpSamplingBilinear_updateGradInput          :: Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
-- | spatialGridSamplerBilinear forward pass (updates the output tensor)
_spatialGridSamplerBilinear_updateOutput            :: Dynamic -> Dynamic -> Dynamic -> Int -> IO ()
-- | spatialGridSamplerBilinear backward-update (updates the layer and bias tensors)
_spatialGridSamplerBilinear_updateGradInput         :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Int -> IO ()
-- | volumetricGridSamplerBilinear forward pass (updates the output tensor)
_volumetricGridSamplerBilinear_updateOutput         :: Dynamic -> Dynamic -> Dynamic -> Int -> IO ()
-- | volumetricGridSamplerBilinear backward-update (updates the layer and bias tensors)
_volumetricGridSamplerBilinear_updateGradInput      :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Int -> IO ()
-- | volumetricConvolution forward pass (updates the output tensor)
_volumetricConvolution_updateOutput                 :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
-- | volumetricConvolution backward-update (updates the layer and bias tensors)
_volumetricConvolution_updateGradInput              :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
-- | volumetricConvolution backward-update (updates the layer and bias tensors). Called 'accGradParameters' in C to indicate accumulating the gradient parameters.
_volumetricConvolution_accGradParameters            :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> Int -> Int -> Double -> IO ()
-- | volumetricFullConvolution forward pass (updates the output tensor)
_volumetricFullConvolution_updateOutput             :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
-- | volumetricFullConvolution backward-update (updates the layer and bias tensors)
_volumetricFullConvolution_updateGradInput          :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
-- | volumetricFullConvolution backward-update (updates the layer and bias tensors). Called 'accGradParameters' in C to indicate accumulating the gradient parameters.
_volumetricFullConvolution_accGradParameters        :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Double -> IO ()
-- | volumetricDilatedConvolution forward pass (updates the output tensor)
_volumetricDilatedConvolution_updateOutput          :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
-- | volumetricDilatedConvolution backward-update (updates the layer and bias tensors)
_volumetricDilatedConvolution_updateGradInput       :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
-- | volumetricDilatedConvolution backward-update (updates the layer and bias tensors). Called 'accGradParameters' in C to indicate accumulating the gradient parameters.
_volumetricDilatedConvolution_accGradParameters     :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Double -> IO ()
-- | volumetricFullDilatedConvolution forward pass (updates the output tensor)
_volumetricFullDilatedConvolution_updateOutput      :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
-- | volumetricFullDilatedConvolution backward-update (updates the layer and bias tensors)
_volumetricFullDilatedConvolution_updateGradInput   :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
-- | volumetricFullDilatedConvolution backward-update (updates the layer and bias tensors). Called 'accGradParameters' in C to indicate accumulating the gradient parameters.
_volumetricFullDilatedConvolution_accGradParameters :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Double -> IO ()
-- | spatialReflectionPadding forward pass (updates the output tensor)
_spatialReflectionPadding_updateOutput              :: Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> IO ()
-- | spatialReflectionPadding backward-update (updates the layer and bias tensors)
_spatialReflectionPadding_updateGradInput           :: Dynamic -> Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> IO ()
-- | spatialReplicationPadding forward pass (updates the output tensor)
_spatialReplicationPadding_updateOutput             :: Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> IO ()
-- | spatialReplicationPadding backward-update (updates the layer and bias tensors)
_spatialReplicationPadding_updateGradInput          :: Dynamic -> Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> IO ()
-- | volumetricReplicationPadding forward pass (updates the output tensor)
_volumetricReplicationPadding_updateOutput          :: Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
-- | volumetricReplicationPadding backward-update (updates the layer and bias tensors)
_volumetricReplicationPadding_updateGradInput       :: Dynamic -> Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
-- | volumetricUpSamplingNearest forward pass (updates the output tensor)
_volumetricUpSamplingNearest_updateOutput           :: Dynamic -> Dynamic -> Int -> IO ()
-- | volumetricUpSamplingNearest backward-update (updates the layer and bias tensors)
_volumetricUpSamplingNearest_updateGradInput        :: Dynamic -> Dynamic -> Dynamic -> Int -> IO ()
-- | volumetricUpSamplingTrilinear forward pass (updates the output tensor)
_volumetricUpSamplingTrilinear_updateOutput         :: Dynamic -> Dynamic -> Int -> Int -> Int -> IO ()
-- | volumetricUpSamplingTrilinear backward-update (updates the layer and bias tensors)
_volumetricUpSamplingTrilinear_updateGradInput      :: Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
-- | temporalReflectionPadding forward pass (updates the output tensor)
_temporalReflectionPadding_updateOutput             :: Dynamic -> Dynamic -> Int -> Int -> IO ()
-- | temporalReflectionPadding backward-update (updates the layer and bias tensors)
_temporalReflectionPadding_updateGradInput          :: Dynamic -> Dynamic -> Dynamic -> Int -> Int -> IO ()
-- | temporalReplicationPadding forward pass (updates the output tensor)
_temporalReplicationPadding_updateOutput            :: Dynamic -> Dynamic -> Int -> Int -> IO ()
-- | temporalReplicationPadding backward-update (updates the layer and bias tensors)
_temporalReplicationPadding_updateGradInput         :: Dynamic -> Dynamic -> Dynamic -> Int -> Int -> IO ()

_sqrt_updateGradInput = ten4 Sig.c_Sqrt_updateGradInput
_square_updateOutput = ten2 Sig.c_Square_updateOutput

_square_updateGradInput t0 t1 t2 = with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
  Sig.c_Square_updateGradInput s' t0' t1' t2'

_tanh_updateOutput = ten2 Sig.c_Tanh_updateOutput
_tanh_updateGradInput t0 t1 t2 = with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
  Sig.c_Tanh_updateGradInput s' t0' t1' t2'


_logSigmoid_updateOutput t0 t1 t2 = with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
  Sig.c_LogSigmoid_updateOutput s' t0' t1' t2'

_logSigmoid_updateGradInput = ten4 Sig.c_LogSigmoid_updateGradInput
_sigmoid_updateOutput = ten2 Sig.c_Sigmoid_updateOutput
_sigmoid_updateGradInput t0 t1 t2 = with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
  Sig.c_Sigmoid_updateGradInput s' t0' t1' t2'

_logSoftMax_updateOutput = ten2dim1 Sig.c_LogSoftMax_updateOutput
_im2Col_updateOutput = ten2int8 Sig.c_Im2Col_updateOutput
_im2Col_updateGradInput = ten2int10 Sig.c_Im2Col_updateGradInput
_gRUFused_updateGradInput = ten5 Sig.c_GRUFused_updateGradInput
_softMax_updateOutput = ten2dim1 Sig.c_SoftMax_updateOutput

_sqrt_updateOutput = ten2double1 Sig.c_Sqrt_updateOutput
_softShrink_updateOutput = ten2double1 Sig.c_SoftShrink_updateOutput
_softPlus_updateOutput = ten2double2 Sig.c_SoftPlus_updateOutput
_hardTanh_updateOutput = ten2double2bool1 Sig.c_HardTanh_updateOutput
_hardTanh_updateGradInput t0 t1 t2 d0 d1 b0 = with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
  Sig.c_HardTanh_updateGradInput s' t0' t1' t2' (realToFrac d0) (realToFrac d0) (toEnum $ fromEnum b0)

_softShrink_updateGradInput t0 t1 t2 d0 = with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
  Sig.c_SoftShrink_updateGradInput s' t0' t1' t2' (realToFrac d0)

_softPlus_updateGradInput = ten4double2 Sig.c_SoftPlus_updateGradInput
_logSoftMax_updateGradInput = ten4dim1 Sig.c_LogSoftMax_updateGradInput
_softMax_updateGradInput = ten4dim1 Sig.c_SoftMax_updateGradInput
_spatialSubSampling_updateOutput = ten4int4 Sig.c_SpatialSubSampling_updateOutput
_spatialSubSampling_updateGradInput = ten4int4 Sig.c_SpatialSubSampling_updateGradInput
_spatialSubSampling_accGradParameters = ten4int4double1 Sig.c_SpatialSubSampling_accGradParameters
_spatialGridSamplerBilinear_updateGradInput = ten5int1 Sig.c_SpatialGridSamplerBilinear_updateGradInput
_sparseLinear_updateParameters = ten5double1 Sig.c_SparseLinear_updateParameters
_volumetricGridSamplerBilinear_updateGradInput = ten5int1 Sig.c_VolumetricGridSamplerBilinear_updateGradInput
_spatialGridSamplerBilinear_updateOutput t0 t1 t2 i0 = with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
  Sig.c_SpatialGridSamplerBilinear_updateOutput s' t0' t1' t2' (fromIntegral i0)

_volumetricGridSamplerBilinear_updateOutput t0 t1 t2 i0 = with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
  Sig.c_VolumetricGridSamplerBilinear_updateOutput s' t0' t1' t2' (fromIntegral i0)

_spatialUpSamplingNearest_updateGradInput t0 t1 t2 i0 = with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
  Sig.c_SpatialUpSamplingNearest_updateGradInput s' t0' t1' t2' (fromIntegral i0)

_temporalUpSamplingNearest_updateGradInput t0 t1 t2 i0 = with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
  Sig.c_TemporalUpSamplingNearest_updateGradInput s' t0' t1' t2' (fromIntegral i0)

_gatedLinear_updateGradInput t0 t1 t2 i0 = with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
  Sig.c_GatedLinear_updateGradInput s' t0' t1' t2' (fromIntegral i0)

_temporalUpSamplingNearest_updateOutput = ten2int1 Sig.c_TemporalUpSamplingNearest_updateOutput
_temporalUpSamplingLinear_updateOutput = ten2int1 Sig.c_TemporalUpSamplingLinear_updateOutput
_gatedLinear_updateOutput = ten2int1 Sig.c_GatedLinear_updateOutput
_spatialUpSamplingNearest_updateOutput = ten2int1 Sig.c_SpatialUpSamplingNearest_updateOutput
_spatialUpSamplingBilinear_updateOutput = ten2int2 Sig.c_SpatialUpSamplingBilinear_updateOutput
_temporalUpSamplingLinear_updateGradInput = ten2int4 Sig.c_TemporalUpSamplingLinear_updateGradInput
_spatialUpSamplingBilinear_updateGradInput = ten2int6 Sig.c_SpatialUpSamplingBilinear_updateGradInput
_volumetricConvolution_updateGradInput = ten5int6 Sig.c_VolumetricConvolution_updateGradInput
_spatialDilatedConvolution_updateGradInput = ten5int8 Sig.c_SpatialDilatedConvolution_updateGradInput
_spatialFullConvolution_updateGradInput = ten5int8 Sig.c_SpatialFullConvolution_updateGradInput
_spatialFullDilatedConvolution_updateGradInput = ten5int10 Sig.c_SpatialFullDilatedConvolution_updateGradInput
_temporalRowConvolution_updateOutput = ten6int3bool1 Sig.c_TemporalRowConvolution_updateOutput
_temporalRowConvolution_updateGradInput = ten6int3bool1 Sig.c_TemporalRowConvolution_updateGradInput
_temporalRowConvolution_accGradParameters = ten6int3bool1double1 Sig.c_TemporalRowConvolution_accGradParameters
_sparseLinear_legacyAccGradParameters = ten6double2 Sig.c_SparseLinear_legacyAccGradParameters
_sparseLinear_accGradParameters = ten6double2 Sig.c_SparseLinear_accGradParameters
_spatialConvolutionLocal_updateOutput = ten6int6long4 Sig.c_SpatialConvolutionLocal_updateOutput
_spatialConvolutionLocal_updateGradInput = ten6int6long4 Sig.c_SpatialConvolutionLocal_updateGradInput
_spatialConvolutionLocal_accGradParameters = ten6int6long4double1 Sig.c_SpatialConvolutionLocal_accGradParameters
_volumetricConvolution_updateOutput = ten6int6 Sig.c_VolumetricConvolution_updateOutput
_spatialFullConvolution_updateOutput = ten6int8 Sig.c_SpatialFullConvolution_updateOutput
_spatialFullConvolution_accGradParameters = ten6int8double1 Sig.c_SpatialFullConvolution_accGradParameters
_spatialDilatedConvolution_updateOutput = ten6int8 Sig.c_SpatialDilatedConvolution_updateOutput
_spatialDilatedConvolution_accGradParameters = ten6int8double1 Sig.c_SpatialDilatedConvolution_accGradParameters
_spatialFullDilatedConvolution_updateOutput = ten6int10 Sig.c_SpatialFullDilatedConvolution_updateOutput
_spatialFullDilatedConvolution_accGradParameters = ten6int10double1 Sig.c_SpatialFullDilatedConvolution_accGradParameters
_gRUFused_updateOutput = ten7 Sig.c_GRUFused_updateOutput
_lSTMFused_updateOutput = ten7 Sig.c_LSTMFused_updateOutput
_lSTMFused_updateGradInput = ten7 Sig.c_LSTMFused_updateGradInput
_batchNormalization_updateOutput = ten8bool1double2 Sig.c_BatchNormalization_updateOutput
_batchNormalization_backward = ten10bool1double2 Sig.c_BatchNormalization_backward
_volumetricConvolution_accGradParameters = ten6int6double1 Sig.c_VolumetricConvolution_accGradParameters
_volumetricFullConvolution_updateOutput = ten6int12 Sig.c_VolumetricFullConvolution_updateOutput
_volumetricFullConvolution_updateGradInput = ten6int12 Sig.c_VolumetricFullConvolution_updateGradInput
_volumetricFullConvolution_accGradParameters = ten6int12double1 Sig.c_VolumetricFullConvolution_accGradParameters
_volumetricDilatedConvolution_updateOutput = ten6int12 Sig.c_VolumetricDilatedConvolution_updateOutput
_volumetricDilatedConvolution_updateGradInput = ten5int12 Sig.c_VolumetricDilatedConvolution_updateGradInput

_sparseLinear_updateOutput = ten4 Sig.c_SparseLinear_updateOutput
_sparseLinear_zeroGradParameters t0 t1 t2 = with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
  Sig.c_SparseLinear_zeroGradParameters s' t0' t1' t2'

_sparseLinear_legacyUpdateOutput = ten4 Sig.c_SparseLinear_legacyUpdateOutput
_volumetricDilatedConvolution_accGradParameters = ten6int12double1 Sig.c_VolumetricDilatedConvolution_accGradParameters
_volumetricFullDilatedConvolution_updateOutput = ten6int15 Sig.c_VolumetricFullDilatedConvolution_updateOutput
_volumetricFullDilatedConvolution_updateGradInput = ten6int15 Sig.c_VolumetricFullDilatedConvolution_updateGradInput
_volumetricFullDilatedConvolution_accGradParameters = ten6int15double1 Sig.c_VolumetricFullDilatedConvolution_accGradParameters


_spatialReflectionPadding_updateOutput     = ten2int4 Sig.c_SpatialReflectionPadding_updateOutput
_spatialReflectionPadding_updateGradInput  t0 t1 t2 i0 i1 i2 i3 = with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
  Sig.c_SpatialReflectionPadding_updateGradInput s' t0' t1' t2' (fromIntegral i0) (fromIntegral i1) (fromIntegral i2) (fromIntegral i3)

_spatialReplicationPadding_updateOutput    = ten2int4 Sig.c_SpatialReplicationPadding_updateOutput

_spatialReplicationPadding_updateGradInput t0 t1 t2 i0 i1 i2 i3 = with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
  Sig.c_SpatialReplicationPadding_updateGradInput s' t0' t1' t2' (fromIntegral i0) (fromIntegral i1) (fromIntegral i2) (fromIntegral i3)



_volumetricReplicationPadding_updateOutput =
  ten2int6 Sig.c_VolumetricReplicationPadding_updateOutput

_volumetricReplicationPadding_updateGradInput =
  ten3int6 Sig.c_VolumetricReplicationPadding_updateGradInput

_volumetricUpSamplingNearest_updateOutput =
  ten2int1 Sig.c_VolumetricUpSamplingNearest_updateOutput

_volumetricUpSamplingNearest_updateGradInput =
  ten3int1 Sig.c_VolumetricUpSamplingNearest_updateGradInput

_volumetricUpSamplingTrilinear_updateOutput =
  ten2int3 Sig.c_VolumetricUpSamplingTrilinear_updateOutput

_volumetricUpSamplingTrilinear_updateGradInput =
  ten2int8 Sig.c_VolumetricUpSamplingTrilinear_updateGradInput

_temporalReflectionPadding_updateOutput =
  ten2int2 Sig.c_TemporalReflectionPadding_updateOutput

_temporalReflectionPadding_updateGradInput =
  ten3int2 Sig.c_TemporalReflectionPadding_updateGradInput

_temporalReplicationPadding_updateOutput =
  ten2int2 Sig.c_TemporalReplicationPadding_updateOutput

_temporalReplicationPadding_updateGradInput =
  ten3int2 Sig.c_TemporalReplicationPadding_updateGradInput

-------------------------------------------------------------------------------
-- Deal love of god...
--
ten1 fn t0 d0 =
  withDynamicState t0 $ \s' t0' -> fn s' t0' (fromIntegral d0)

ten2dim1 fn t0 t1 d0 =
  with2DynamicState t0 t1 $ \s' t0' t1' ->
    fn s' t0' t1' (fromIntegral d0)

ten2 fn t0 t1 =
  with2DynamicState t0 t1 $ \s' t0' t1' ->
    fn s' t0' t1'

ten3 fn t0 t1 t2 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
    fn s' t0' t1' t2'

ten4 fn t0 t1 t2 t3 =
  with2DynamicState t0 t1 $ \s' t0' t1' ->
  with2DynamicState t2 t3 $ \_  t2' t3' ->
    fn s' t0' t1' t2' t3'

ten4dim1 fn t0 t1 t2 t3 d0 =
  with2DynamicState t0 t1 $ \s' t0' t1' ->
  with2DynamicState t2 t3 $ \_  t2' t3' ->
    fn s' t0' t1' t2' t3'
      (fromIntegral d0)

ten5 fn t0 t1 t2 t3 t4 =
  with2DynamicState t0 t1 $ \s' t0' t1' ->
  with3DynamicState t2 t3 t4 $ \_  t2' t3' t4' ->
    fn s' t0' t1' t2' t3' t4'

ten3bool2 fn t0 t1 t2 b0 b1 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
    fn s' t0' t1' t2' (toEnum $ fromEnum b0) (toEnum $ fromEnum b1)

ten4bool2 fn t0 t1 t2 t3 b0 b1 =
  with2DynamicState t0 t1 $ \s' t0' t1' ->
  with2DynamicState t2 t3 $ \_ t2' t3' ->
    fn s' t0' t1' t2' t3' (toEnum $ fromEnum b0) (toEnum $ fromEnum b1)

ten3bool1ten1bool1 fn t0 t1 t2 b0 t3 b1 =
  with2DynamicState t0 t1 $ \s' t0' t1' ->
  with2DynamicState t2 t3 $ \_ t2' t3' ->
    fn s' t0' t1' t2' (toEnum $ fromEnum b0) t3' (toEnum $ fromEnum b1)


ten3int1 fn t0 t1 t2 i0 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
    fn s' t0' t1' t2' (fromIntegral i0)

ten3int2 fn t0 t1 t2 i0 i1 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
    fn s' t0' t1' t2' (fromIntegral i0) (fromIntegral i1)

ten3int3 fn t0 t1 t2 i0 i1 i2 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
    fn s' t0' t1' t2' (fromIntegral i0) (fromIntegral i1) (fromIntegral i2)

ten3int4 fn t0 t1 t2 i0 i1 i2 i3 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
    fn s' t0' t1' t2' (fromIntegral i0) (fromIntegral i1) (fromIntegral i2) (fromIntegral i3)

ten4int2 fn t0 t1 t2 t3 i0 i1 =
  with2DynamicState t0 t1 $ \s' t0' t1' ->
  with2DynamicState t2 t3 $ \_  t2' t3' ->
    fn s' t0' t1' t2' t3'
      (fromIntegral i0) (fromIntegral i1)

ten4int4 fn t0 t1 t2 t3 i0 i1 i2 i3 =
  with2DynamicState t0 t1 $ \s' t0' t1' ->
  with2DynamicState t2 t3 $ \_  t2' t3' ->
    fn s' t0' t1' t2' t3'
      (fromIntegral i0) (fromIntegral i1)
      (fromIntegral i2) (fromIntegral i3)

ten3int6bool2 fn t0 t1 t2 i0 i1 i2 i3 i4 i5 b0 b1 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
    fn s' t0' t1' t2'
      (fromIntegral i0) (fromIntegral i1)
      (fromIntegral i2) (fromIntegral i3)
      (fromIntegral i4) (fromIntegral i5)
      (toEnum $ fromEnum b0)
      (toEnum $ fromEnum b1)

ten2int6bool2 fn t0 t1 i0 i1 i2 i3 i4 i5 b0 b1 =
  with2DynamicState t0 t1 $ \s' t0' t1' ->
    fn s' t0' t1'
      (fromIntegral i0) (fromIntegral i1)
      (fromIntegral i2) (fromIntegral i3)
      (fromIntegral i4) (fromIntegral i5)
      (toEnum $ fromEnum b0)
      (toEnum $ fromEnum b1)

ten2int9bool2 fn t0 t1 i0 i1 i2 i3 i4 i5 i6 i7 i8 b0 b1 =
  with2DynamicState t0 t1 $ \s' t0' t1' ->
    fn s' t0' t1'
      (fromIntegral i0) (fromIntegral i1)
      (fromIntegral i2) (fromIntegral i3)
      (fromIntegral i4) (fromIntegral i5)
      (fromIntegral i6) (fromIntegral i7)
      (fromIntegral i8)
      (toEnum $ fromEnum b0)
      (toEnum $ fromEnum b1)



ten3int9bool2 fn t0 t1 t2 i0 i1 i2 i3 i4 i5 i6 i7 i8 b0 b1 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
    fn s' t0' t1' t2'
      (fromIntegral i0) (fromIntegral i1)
      (fromIntegral i2) (fromIntegral i3)
      (fromIntegral i4) (fromIntegral i5)
      (fromIntegral i6) (fromIntegral i7)
      (fromIntegral i8)
      (toEnum $ fromEnum b0)
      (toEnum $ fromEnum b1)


ten3int6 fn t0 t1 t2 i0 i1 i2 i3 i4 i5 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
    fn s' t0' t1' t2'
      (fromIntegral i0) (fromIntegral i1)
      (fromIntegral i2) (fromIntegral i3)
      (fromIntegral i4) (fromIntegral i5)

ten6int16 fn t0 t1 t2 t3 t4 t5 i0 i1 i2 i3 i4 i5 i6 i7 i8 i9 i10 i11 i12 i13 i14 i15 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
   with3DynamicState t3 t4 t5 $ \s' t3' t4' t5' ->
    fn s' t0' t1' t2' t3' t4' t5'
      (fromIntegral i0) (fromIntegral i1)
      (fromIntegral i2) (fromIntegral i3)
      (fromIntegral i4) (fromIntegral i5)
      (fromIntegral i6) (fromIntegral i7)
      (fromIntegral i8) (fromIntegral i9)
      (fromIntegral i10) (fromIntegral i11)
      (fromIntegral i12) (fromIntegral i13)
      (fromIntegral i14) (fromIntegral i15)

ten6int15 fn t0 t1 t2 t3 t4 t5 i0 i1 i2 i3 i4 i5 i6 i7 i8 i9 i10 i11 i12 i13 i14 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
   with3DynamicState t3 t4 t5 $ \s' t3' t4' t5' ->
    fn s' t0' t1' t2' t3' t4' t5'
      (fromIntegral i0) (fromIntegral i1)
      (fromIntegral i2) (fromIntegral i3)
      (fromIntegral i4) (fromIntegral i5)
      (fromIntegral i6) (fromIntegral i7)
      (fromIntegral i8) (fromIntegral i9)
      (fromIntegral i10) (fromIntegral i11)
      (fromIntegral i12) (fromIntegral i13)
      (fromIntegral i14) 

ten6int15double1 fn t0 t1 t2 t3 t4 t5 i0 i1 i2 i3 i4 i5 i6 i7 i8 i9 i10 i11 i12 i13 i14 d0 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
   with3DynamicState t3 t4 t5 $ \s' t3' t4' t5' ->
    fn s' t0' t1' t2' t3' t4' t5'
      (fromIntegral i0) (fromIntegral i1)
      (fromIntegral i2) (fromIntegral i3)
      (fromIntegral i4) (fromIntegral i5)
      (fromIntegral i6) (fromIntegral i7)
      (fromIntegral i8) (fromIntegral i9)
      (fromIntegral i10) (fromIntegral i11)
      (fromIntegral i12) (fromIntegral i13)
      (fromIntegral i14) 
      (realToFrac d0)

ten4int2double1 fn t0 t1 t2 t3 i0 i1 d0 =
  with2DynamicState t0 t1 $ \s' t0' t1' ->
   with2DynamicState t2 t3 $ \ _ t2' t3' ->
    fn s' t0' t1' t2' t3'
      (fromIntegral i0) (fromIntegral i1)
      (realToFrac d0)

ten4int4double1 fn t0 t1 t2 t3 i0 i1 i2 i3 d0 =
  with2DynamicState t0 t1 $ \s' t0' t1' ->
   with2DynamicState t2 t3 $ \ _ t2' t3' ->
    fn s' t0' t1' t2' t3'
      (fromIntegral i0) (fromIntegral i1)
      (fromIntegral i2) (fromIntegral i3)
      (realToFrac d0)

ten6int4double1 fn t0 t1 t2 t3 t4 t5 i0 i1 i2 i3 d0 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
   with3DynamicState t3 t4 t5 $ \s' t3' t4' t5' ->
    fn s' t0' t1' t2' t3' t4' t5'
      (fromIntegral i0) (fromIntegral i1)
      (fromIntegral i2) (fromIntegral i3)
      (realToFrac d0)

ten6int6double1 fn t0 t1 t2 t3 t4 t5 i0 i1 i2 i3 i4 i5 d0 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
   with3DynamicState t3 t4 t5 $ \s' t3' t4' t5' ->
    fn s' t0' t1' t2' t3' t4' t5'
      (fromIntegral i0) (fromIntegral i1)
      (fromIntegral i2) (fromIntegral i3)
      (fromIntegral i4) (fromIntegral i5)
      (realToFrac d0)


ten6int10 fn t0 t1 t2 t3 t4 t5 i0 i1 i2 i3 i4 i5 i6 i7 i8 i9 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
   with3DynamicState t3 t4 t5 $ \s' t3' t4' t5' ->
    fn s' t0' t1' t2' t3' t4' t5'
      (fromIntegral i0) (fromIntegral i1)
      (fromIntegral i2) (fromIntegral i3)
      (fromIntegral i4) (fromIntegral i5)
      (fromIntegral i6) (fromIntegral i7)
      (fromIntegral i8) (fromIntegral i9)

ten6int6 fn t0 t1 t2 t3 t4 t5 i0 i1 i2 i3 i4 i5 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
   with3DynamicState t3 t4 t5 $ \s' t3' t4' t5' ->
    fn s' t0' t1' t2' t3' t4' t5'
      (fromIntegral i0) (fromIntegral i1)
      (fromIntegral i2) (fromIntegral i3)
      (fromIntegral i4) (fromIntegral i5)

ten6int6long4 fn t0 t1 t2 t3 t4 t5 i0 i1 i2 i3 i4 i5 l0 l1 l2 l3 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
   with3DynamicState t3 t4 t5 $ \s' t3' t4' t5' ->
    fn s' t0' t1' t2' t3' t4' t5'
      (fromIntegral i0) (fromIntegral i1)
      (fromIntegral i2) (fromIntegral i3)
      (fromIntegral i4) (fromIntegral i5)
      (fromIntegral l0) (fromIntegral l1)
      (fromIntegral l2) (fromIntegral l3)

ten6double2 fn t0 t1 t2 t3 t4 t5 d0 d1 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
   with3DynamicState t3 t4 t5 $ \s' t3' t4' t5' ->
    fn s' t0' t1' t2' t3' t4' t5'
      (realToFrac d0)
      (realToFrac d1)

ten6int6long4double1 fn t0 t1 t2 t3 t4 t5 i0 i1 i2 i3 i4 i5 l0 l1 l2 l3 d0 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
   with3DynamicState t3 t4 t5 $ \s' t3' t4' t5' ->
    fn s' t0' t1' t2' t3' t4' t5'
      (fromIntegral i0) (fromIntegral i1)
      (fromIntegral i2) (fromIntegral i3)
      (fromIntegral i4) (fromIntegral i5)
      (fromIntegral l0) (fromIntegral l1)
      (fromIntegral l2) (fromIntegral l3)
      (realToFrac d0)

ten6int6long6double1 fn t0 t1 t2 t3 t4 t5 i0 i1 i2 i3 i4 i5 l0 l1 l2 l3 l4 l5 d0 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
   with3DynamicState t3 t4 t5 $ \s' t3' t4' t5' ->
    fn s' t0' t1' t2' t3' t4' t5'
      (fromIntegral i0) (fromIntegral i1)
      (fromIntegral i2) (fromIntegral i3)
      (fromIntegral i4) (fromIntegral i5)
      (fromIntegral l0) (fromIntegral l1)
      (fromIntegral l2) (fromIntegral l3)
      (fromIntegral l4) (fromIntegral l5)
      (realToFrac d0)

ten6int8 fn t0 t1 t2 t3 t4 t5 i0 i1 i2 i3 i4 i5 i6 i7 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
   with3DynamicState t3 t4 t5 $ \s' t3' t4' t5' ->
    fn s' t0' t1' t2' t3' t4' t5'
      (fromIntegral i0) (fromIntegral i1)
      (fromIntegral i2) (fromIntegral i3)
      (fromIntegral i4) (fromIntegral i5)
      (fromIntegral i6) (fromIntegral i7)

ten6int8double1 fn t0 t1 t2 t3 t4 t5 i0 i1 i2 i3 i4 i5 i6 i7 d0 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
   with3DynamicState t3 t4 t5 $ \s' t3' t4' t5' ->
    fn s' t0' t1' t2' t3' t4' t5'
      (fromIntegral i0) (fromIntegral i1)
      (fromIntegral i2) (fromIntegral i3)
      (fromIntegral i4) (fromIntegral i5)
      (fromIntegral i6) (fromIntegral i7)
      (realToFrac d0)

ten6int10double1 fn t0 t1 t2 t3 t4 t5 i0 i1 i2 i3 i4 i5 i6 i7 i8 i9 d0 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
   with3DynamicState t3 t4 t5 $ \s' t3' t4' t5' ->
    fn s' t0' t1' t2' t3' t4' t5'
      (fromIntegral i0) (fromIntegral i1)
      (fromIntegral i2) (fromIntegral i3)
      (fromIntegral i4) (fromIntegral i5)
      (fromIntegral i6) (fromIntegral i7)
      (fromIntegral i8) (fromIntegral i9)
      (realToFrac d0)


ten6int12double1 fn t0 t1 t2 t3 t4 t5 i0 i1 i2 i3 i4 i5 i6 i7 i8 i9 i10 i11 d0 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
   with3DynamicState t3 t4 t5 $ \s' t3' t4' t5' ->
    fn s' t0' t1' t2' t3' t4' t5'
      (fromIntegral i0) (fromIntegral i1)
      (fromIntegral i2) (fromIntegral i3)
      (fromIntegral i4) (fromIntegral i5)
      (fromIntegral i6) (fromIntegral i7)
      (fromIntegral i8) (fromIntegral i9)
      (fromIntegral i10) (fromIntegral i11)
      (realToFrac d0)

ten5double1 fn t0 t1 t2 t3 t4 d0 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
   with2DynamicState t3 t4 $ \s' t3' t4' ->
    fn s' t0' t1' t2' t3' t4' (realToFrac d0)


ten5int1 fn t0 t1 t2 t3 t4 i0 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
   with2DynamicState t3 t4 $ \s' t3' t4' ->
    fn s' t0' t1' t2' t3' t4' (fromIntegral i0)

ten5int6 fn t0 t1 t2 t3 t4 i0 i1 i2 i3 i4 i5 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
   with2DynamicState t3 t4 $ \s' t3' t4' ->
    fn s' t0' t1' t2' t3' t4'
      (fromIntegral i0) (fromIntegral i1)
      (fromIntegral i2) (fromIntegral i3)
      (fromIntegral i4) (fromIntegral i5)

ten5int8 fn t0 t1 t2 t3 t4 i0 i1 i2 i3 i4 i5 i6 i7 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
   with2DynamicState t3 t4 $ \s' t3' t4' ->
    fn s' t0' t1' t2' t3' t4'
      (fromIntegral i0) (fromIntegral i1)
      (fromIntegral i2) (fromIntegral i3)
      (fromIntegral i4) (fromIntegral i5)
      (fromIntegral i6) (fromIntegral i7)


ten5int10 fn t0 t1 t2 t3 t4 i0 i1 i2 i3 i4 i5 i6 i7 i8 i9 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
   with2DynamicState t3 t4 $ \s' t3' t4' ->
    fn s' t0' t1' t2' t3' t4'
      (fromIntegral i0) (fromIntegral i1)
      (fromIntegral i2) (fromIntegral i3)
      (fromIntegral i4) (fromIntegral i5)
      (fromIntegral i6) (fromIntegral i7)
      (fromIntegral i8) (fromIntegral i9)

ten2int10 fn t0 t1 i0 i1 i2 i3 i4 i5 i6 i7 i8 i9 =
  with2DynamicState t0 t1 $ \s' t0' t1' ->
    fn s' t0' t1'
      (fromIntegral i0) (fromIntegral i1)
      (fromIntegral i2) (fromIntegral i3)
      (fromIntegral i4) (fromIntegral i5)
      (fromIntegral i6) (fromIntegral i7)
      (fromIntegral i8) (fromIntegral i9)



ten5int12 fn t0 t1 t2 t3 t4 i0 i1 i2 i3 i4 i5 i6 i7 i8 i9 i10 i11 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
   with2DynamicState t3 t4 $ \s' t3' t4' ->
    fn s' t0' t1' t2' t3' t4'
      (fromIntegral i0) (fromIntegral i1)
      (fromIntegral i2) (fromIntegral i3)
      (fromIntegral i4) (fromIntegral i5)
      (fromIntegral i6) (fromIntegral i7)
      (fromIntegral i8) (fromIntegral i9)
      (fromIntegral i10) (fromIntegral i11)


ten3double1 fn t0 t1 t2 d0 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
    fn s' t0' t1' t2' (realToFrac d0)

ten3double1bool1 fn t0 t1 t2 d0 b0 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
    fn s' t0' t1' t2' (realToFrac d0) (toEnum $ fromEnum b0)

ten3double2 fn t0 t1 t2 d0 d1 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
    fn s' t0' t1' t2' (realToFrac d0) (realToFrac d1)

ten4double2 fn t0 t1 t2 t3 d0 d1 =
  with2DynamicState t0 t1 $ \s' t0' t1' ->
  with2DynamicState t2 t3 $ \ _ t2' t3' ->
    fn s' t0' t1' t2' t3' (realToFrac d0) (realToFrac d1)



ten4double2bool2 fn t0 t1 t2 t3 d0 d1 b0 b1 =
  with2DynamicState t0 t1 $ \s' t0' t1' ->
  with2DynamicState t2 t3 $ \ _ t2' t3' ->
    fn s' t0' t1' t2' t3'
      (realToFrac d0)        (realToFrac d1)
      (toEnum $ fromEnum b0) (toEnum $ fromEnum b1)

ten4bool1ten1bool1 fn t0 t1 t2 t3 b0 t4 b1 =
  with2DynamicState t0 t1 $ \s' t0' t1' ->
  with3DynamicState t2 t3 t4 $ \ _ t2' t3' t4' ->
    fn s' t0' t1' t2' t3' (toEnum $ fromEnum b0) t4' (toEnum $ fromEnum b1)

ten2double1 fn t0 t1 d0 =
  with2DynamicState t0 t1 $ \s' t0' t1' ->
    fn s' t0' t1'
      (realToFrac d0)

ten2double2 fn t0 t1 d0 d1 =
  with2DynamicState t0 t1 $ \s' t0' t1' ->
    fn s' t0' t1'
      (realToFrac d0)
      (realToFrac d1)

ten2double1bool1 fn t0 t1 d0 b0 =
  with2DynamicState t0 t1 $ \s' t0' t1' ->
    fn s' t0' t1'
      (realToFrac d0)
      (toEnum $ fromEnum b0)

ten2double2bool1 fn t0 t1 d0 d1 b0 =
  with2DynamicState t0 t1 $ \s' t0' t1' ->
    fn s' t0' t1'
      (realToFrac d0)
      (realToFrac d1)
      (toEnum $ fromEnum b0)

ten3double2bool1 fn t0 t1 t2 d0 d1 b0 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
    fn s' t0' t1' t2'
      (realToFrac d0)
      (realToFrac d1)
      (toEnum $ fromEnum b0)

ten6int3bool1 fn t0 t1 t2 t3 t4 t5 i0 i1 i2 b0 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
   with3DynamicState t3 t4 t5 $ \_ t3' t4' t5' ->
    fn s' t0' t1' t2' t3' t4' t5'
      (fromIntegral i0)
      (fromIntegral i1)
      (fromIntegral i2)
      (toEnum $ fromEnum b0)

ten6int3bool1double1 fn t0 t1 t2 t3 t4 t5 i0 i1 i2 b0 d0 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
   with3DynamicState t3 t4 t5 $ \_ t3' t4' t5' ->
    fn s' t0' t1' t2' t3' t4' t5'
      (fromIntegral i0)
      (fromIntegral i1)
      (fromIntegral i2)
      (toEnum $ fromEnum b0)
      (realToFrac d0)

ten3bool1double1 fn t0 t1 t2 b0 d0 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
    fn s' t0' t1' t2' (toEnum $ fromEnum b0) (realToFrac d0)

ten10bool1double2 fn t0 t1 t2 t3 t4 t5 t6 t7 t8 t9 b0 d0 d1 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
   with2DynamicState t3 t4 $ \_ t3' t4' ->
    with3DynamicState t5 t6 t7 $ \_ t5' t6' t7' ->
     with2DynamicState t8 t9 $ \_ t8' t9' ->
      fn s' t0' t1' t2' t3' t4' t5' t6' t7' t8' t9'
        (toEnum $ fromEnum b0)
        (realToFrac d0)
        (realToFrac d1)

ten8bool1double2 fn t0 t1 t2 t3 t4 t5 t6 t7 b0 d0 d1 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
   with2DynamicState t3 t4 $ \_ t3' t4' ->
    with3DynamicState t5 t6 t7 $ \_ t5' t6' t7' ->
      fn s' t0' t1' t2' t3' t4' t5' t6' t7'
        (toEnum $ fromEnum b0)
        (realToFrac d0)
        (realToFrac d1)

ten7 fn t0 t1 t2 t3 t4 t5 t6 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
   with2DynamicState t3 t4 $ \_ t3' t4' ->
    with2DynamicState t5 t6 $ \_ t5' t6' ->
      fn s' t0' t1' t2' t3' t4' t5' t6'


ten6int12 fn t0 t1 t2 t3 t4 t5 i0 i1 i2 i3 i4 i5 i6 i7 i8 i9 i10 i11 =
  with3DynamicState t0 t1 t2 $ \s' t0' t1' t2' ->
   with3DynamicState t3 t4 t5 $ \_ t3' t4' t5' ->
    fn s' t0' t1' t2' t3' t4' t5'
      (fromIntegral i0) (fromIntegral i1)
      (fromIntegral i2) (fromIntegral i3)
      (fromIntegral i4) (fromIntegral i5)
      (fromIntegral i6) (fromIntegral i7)
      (fromIntegral i8) (fromIntegral i9)
      (fromIntegral i10) (fromIntegral i11)




-------------------------------------------------------------------------------

ten2int1 fn t0 t1 i0 =
  with2DynamicState t0 t1 $ \s' t0' t1' ->
    fn s' t0' t1' (fromIntegral i0)

ten2int2 fn t0 t1 i0 i1 =
  with2DynamicState t0 t1 $ \s' t0' t1' ->
    fn s' t0' t1' (fromIntegral i0) (fromIntegral i1)

ten2int3 fn t0 t1 i0 i1 i2 =
  with2DynamicState t0 t1 $ \s' t0' t1' ->
    fn s' t0' t1' (fromIntegral i0) (fromIntegral i1) (fromIntegral i2)
ten2int4 fn t0 t1 i0 i1 i2 i3 =
  with2DynamicState t0 t1 $ \s' t0' t1' ->
    fn s' t0' t1' (fromIntegral i0) (fromIntegral i1) (fromIntegral i2) (fromIntegral i3)


ten2int6 fn t0 t1 i0 i1 i2 i3 i4 i5 =
  with2DynamicState t0 t1 $ \s' t0' t1' ->
    fn s' t0' t1'
      (fromIntegral i0) (fromIntegral i1)
      (fromIntegral i2) (fromIntegral i3)
      (fromIntegral i4) (fromIntegral i5)

ten2int8 fn t0 t1 i0 i1 i2 i3 i4 i5 i6 i7 =
  with2DynamicState t0 t1 $ \s' t0' t1' ->
    fn s' t0' t1'
      (fromIntegral i0) (fromIntegral i1)
      (fromIntegral i2) (fromIntegral i3)
      (fromIntegral i4) (fromIntegral i5)
      (fromIntegral i6) (fromIntegral i7)

-- ========================================================================= --


-- CPU TENSORS ONLY
-- unfolded_acc  :: Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
-- unfolded_copy :: Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
-- volumetricConvolutionMM_updateOutput :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
-- volumetricConvolutionMM_updateGradInput :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
-- volumetricConvolutionMM_accGradParameters :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Double -> IO ()
-- temporalSubSampling_updateOutput :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Int -> Int -> Int -> IO ()
-- temporalSubSampling_updateGradInput :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Int -> Int -> IO ()
-- temporalSubSampling_accGradParameters :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Int -> Int -> Double -> IO ()
-- spatialFullConvolutionMap_updateOutput :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> IO ()
-- spatialFullConvolutionMap_updateGradInput :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> IO ()
-- spatialFullConvolutionMap_accGradParameters :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> Double -> IO ()
-- hardShrink_updateOutput      :: Dynamic -> Dynamic -> Double -> IO ()
-- hardShrink_updateGradInput   :: Dynamic -> Dynamic -> Dynamic -> Double -> IO ()
-- col2Im_updateGradInput       :: Dynamic -> Dynamic -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
-- linear_updateOutput      :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> IO ()
-- linear_updateGradInput   :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> IO ()
-- linear_accGradParameters :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Double -> IO ()
-- sparseLinear_legacyZeroGradParameters :: Dynamic -> Dynamic -> Dynamic -> IO ()
-- sparseLinear_legacyUpdateParameters   :: Dynamic -> Dynamic -> Dynamic -> Dynamic -> Dynamic -> Double -> IO ()
