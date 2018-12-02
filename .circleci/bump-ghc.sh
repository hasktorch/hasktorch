#!/usr/bin/env bash

CURRENT_GHC="8.4.3"
NEXT_GHC="8.4.4"

if ! [ -f "../.circleci/config.yml" ]; then
  echo "run the CI updater from _inside_ of the .circleci folder!"
  exit 1
elif ! [ -f "../cabal/project.freeze-${NEXT_GHC}" ]; then
  echo "freeze file does not exist for this version of ghc!"
  exit 1
else
  for config in "cabal.project.local" "config.yml"; do
    sed -i -e "s/${CURRENT_GHC}/${NEXT_GHC}/g" $config
  done
  cp -f ../cabal/project.freeze-${NEXT_GHC} ./cabal.project.freeze
fi
