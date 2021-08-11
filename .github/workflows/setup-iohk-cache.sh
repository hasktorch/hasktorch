#!/usr/bin/env bash

cachix use iohk

if ! grep hydra.iohk.io ~/.config/nix/nix.conf >& /dev/null; then
    sed -i -e 's/^\(substituters.*\)$/\1 https:\/\/hydra.iohk.io/g' ~/.config/nix/nix.conf
    sed -i -e 's/^\(trusted-public-keys.*\)$/\1 hydra.iohk.io:f\/Ea+s+dFdN+3Y\/G+FDgSq+a5NEWhJGzdjvKNGv0\/EQ=/g' ~/.config/nix/nix.conf
else
    echo hydra.iohk.io is already set.
fi
