{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE UndecidableInstances #-}

module Torch.Typed.RandomSpec
  ( Torch.Typed.RandomSpec.spec,
  )
where

import Test.Hspec (Spec, describe, it)
import Test.QuickCheck ()
import Torch.Typed
import Torch.Typed.AuxiliarySpec

spec :: Spec
spec = foldMap spec' availableDevices

spec' :: Device -> Spec
spec' device =
  describe ("for " <> show device) $ do
    it "multinomial" $ case device of
      Device {deviceType = CPU, deviceIndex = 0} -> do
        g <- mkPureGenerator @'( 'CPU, 0) 0
        let probs = ones :: Tensor '( 'CPU, 0) 'Float '[4]
            (t, _) = multinomial @3 True probs g
        checkDynamicTensorAttributes t
      Device {deviceType = CUDA, deviceIndex = 0} -> do
        g <- mkPureGenerator @'( 'CUDA, 0) 0
        let probs = ones :: Tensor '( 'CUDA, 0) 'Float '[4]
            (t, _) = multinomial @3 True probs g
        checkDynamicTensorAttributes t
      Device {deviceType = MPS, deviceIndex = 0} -> do
        g <- mkPureGenerator @'( 'MPS, 0) 0
        let probs = ones :: Tensor '( 'MPS, 0) 'Float '[4]
            (t, _) = multinomial @3 True probs g
        checkDynamicTensorAttributes t
