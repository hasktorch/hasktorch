module Main where

import TestMath (testMath)
import TestRandom (testsRandom)
import TestRawInterface (testRawInterface)
import TestTH (testTH)

main = do
  testMath
  testRawInterface
  testsRandom
  testTH
