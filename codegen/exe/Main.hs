{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE FlexibleContexts #-}
module Main where

import Options.Applicative as OptParse

import CLIOptions
import FileMappings

-- ========================================================================= --

main :: IO ()
main = execParser opts >>= run
 where
  opts :: ParserInfo Options
  opts = info (cliOptions <**> helper) idm


run :: Options -> IO ()
run os = do
  putStrLn $ "Here we would run: "
    ++ show (codegenType os)
    ++ " on "
    ++ show (libraries os)


