module Main where

import TestTH (testTH)
import TestRawInterface (testRawInterface)

main = do
  testRawInterface
  testTH
