{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}

module Torch.Typed.SerializeSpec
  ( Torch.Typed.SerializeSpec.spec,
  )
where

import Control.Monad (foldM)
import Data.Kind
import Data.Maybe
import Data.Proxy
import GHC.Exts (toList)
import GHC.Generics
import GHC.TypeLits
import Test.Hspec (Spec, describe, it, shouldBe)
import Test.QuickCheck ()
import Torch (ATenTensor)
import Torch.Internal.Class (Castable)
import Torch.Internal.Managed.Type.Context (manual_seed_L)
import Torch.Typed

import qualified Torch.DType as D
import qualified Torch.Device as D

data
  MLP
    (inputFeatures :: Nat)
    (outputFeatures :: Nat)
    (hiddenFeatures :: Nat)
    (dtype :: D.DType)
    (device :: (D.DeviceType, Nat)) = MLP
  { layer0 :: Linear inputFeatures hiddenFeatures dtype device,
    layer1 :: Linear hiddenFeatures hiddenFeatures dtype device,
    layer2 :: Linear hiddenFeatures outputFeatures dtype device
  }
  deriving (Show, Generic, Parameterized)

saveMLP :: MLP 10 3 4 'D.Float '(D.CPU, 0) -> FilePath -> IO ()
saveMLP model filePath = saveParameters model filePath 

loadMLP :: MLP 10 3 4 'D.Float '(D.CPU, 0) -> FilePath -> IO (MLP 10 3 4 'D.Float '(D.CPU, 0))
loadMLP model filePath = loadParameters model filePath 

spec :: Spec
spec = pure ()

