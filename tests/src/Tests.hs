module Main where

import TestTH (testTH)
import TestRandom (testsRandom)
import TestRawInterface (testRawInterface)

main = do
  testRawInterface
  testTH
  testsRandom
