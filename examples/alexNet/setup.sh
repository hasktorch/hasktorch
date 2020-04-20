#!/usr/bin/env bash

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1lH-GwVcNY0Jmf10zMRhGbbXaTe-LIH76' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1lH-GwVcNY0Jmf10zMRhGbbXaTe-LIH76" -O alexNet.pt && rm -rf /tmp/cookies.txt
wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
tar -xf images.tar.gz
rm images.tar.gz
