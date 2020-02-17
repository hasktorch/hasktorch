#!/usr/bin/env bash

nix-shell --run "hie-8.8.2 -l hie.log -d --vomit $@ 2> hie.err"
