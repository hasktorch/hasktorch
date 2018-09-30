{-# LANGUAGE Strict #-}
module Torch.StorageSpec where

import Foreign hiding (with)
import Foreign.C.Types
import qualified Foreign as FM
import qualified Foreign.Marshal.Array as FM
import qualified Foreign.Marshal.Utils as FM
import Control.Monad
import Control.Monad.Managed
import GHC.Int
import Test.Hspec
import Torch.Types.TH.Structs
import Torch.Double.Storage
import Control.Monad.IO.Class
-- import Torch.Double
--
-- import qualified Torch.Double.Dynamic as Dynamic
-- import qualified Torch.Double.Storage as Storage
import qualified Data.List as List

main :: IO ()
main = hspec spec

spec :: Spec
spec =
  describe "fromList / storagedata" $ do
    let xs = [0..500]
    st <- runIO $ fromList xs

    it "is instantiated with an offset of 0" $
      size st >>= (`shouldBe` length xs)

    it "returns xs for storagedata" $
      storagedata st >>= (`shouldBe` xs)

    it ("can access the first 50 elements by `get`") $ do
      forM_ [0..50] $ \i ->
        get st i >>= (`shouldBe` fromIntegral i)

    it "should continue to be consistent after gets" $
      storagedata st >>= (`shouldBe` xs)

