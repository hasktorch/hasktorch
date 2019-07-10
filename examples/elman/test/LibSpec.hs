
module LibSpec where

import Test.Hspec


main :: IO ()
main = hspec $ do
  describe "sample test" $ do
    it "sample test" $
      "hello world" `shouldBe` "hello world"
      