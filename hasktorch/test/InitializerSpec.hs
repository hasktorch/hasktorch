module InitializerSpec where

import Test.Hspec
import Initializers

spec :: Spec
spec = do
    -- TODO - add validation conditions
    describe "Check initializers" $ do
        it "runs kaiming uniform" $ do
            x <- kaimingUniform' [4, 5]
            print x
        it "runs kaiming normal" $ do
            x <- kaimingNormal' [4, 5]
            print x
        it "runs Xavier Uniform" $ do
            x <- xavierUniform' [4, 5]
            print x
        it "runs Xavier Normal" $ do
            x <- xavierNormal' [4, 5]
            print x
