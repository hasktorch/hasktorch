{-# OPTIONS_GHC -fplugin=GHC.NotExport.Plugin #-}

module Main where

inline_c_ffi_foo a = a + a

main :: IO ()
main = putStrLn "☺"
