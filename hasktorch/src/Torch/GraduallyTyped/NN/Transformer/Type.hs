{-# LANGUAGE DataKinds #-}
{-# LANGUAGE EmptyCase #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE StandaloneKindSignatures #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}

module Torch.GraduallyTyped.NN.Transformer.Type where

import Control.Monad.Reader (MonadIO, MonadReader, ask, liftIO)
import qualified Data.Map as Map
import Data.Singletons.TH (genSingletons)
import Foreign.ForeignPtr (ForeignPtr)
import Torch.GraduallyTyped.DType (KnownDataType)
import Torch.GraduallyTyped.Device (KnownDevice)
import Torch.GraduallyTyped.Layout (KnownLayout)
import Torch.GraduallyTyped.Shape.Type (KnownShape)
import Torch.GraduallyTyped.Tensor.Type (Tensor (..), checkedDataType, checkedDevice, checkedLayout, checkedShape)
import qualified Torch.Internal.Type as ATen (Tensor)

data TransformerStyle = T5 | BART | MBART | BERT | Pegasus
  deriving (Show, Eq)

genSingletons [''TransformerStyle]

type TensorDict = Map.Map String (ForeignPtr ATen.Tensor)

lookupTensor ::
  forall requiresGradient layout device dataType shape m.
  ( MonadReader TensorDict m,
    MonadIO m,
    MonadFail m,
    KnownLayout layout,
    KnownDevice device,
    KnownDataType dataType,
    KnownShape shape
  ) =>
  String ->
  m (Tensor requiresGradient layout device dataType shape)
lookupTensor s = do
  tensorDict <- ask
  liftIO
    ( maybe
        (fail $ "`" <> show s <> "` is not in the state dictionary.")
        (pure . UnsafeTensor)
        (Map.lookup s tensorDict)
    )
    >>= checkedLayout
    >>= checkedDevice
    >>= checkedDataType
    >>= checkedShape
