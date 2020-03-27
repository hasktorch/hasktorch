#!/usr/bin/env bash

nix-shell --run "hie-8.8.3 -l hie.log -d --vomit $@ 2> hie.err"
