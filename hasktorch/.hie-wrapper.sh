#!/usr/bin/env bash

nix-shell --run "hie-8.6.5 -l hie.log -d --vomit $@ 2> hie.err"
