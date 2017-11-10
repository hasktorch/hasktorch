module Main where

import TestMath (testMath)
import TestRandom (testsRandom)
import TestTH (testTH)

main = do
  testMath
  testsRandom
  testTH
