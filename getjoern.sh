#!/bin/bash
# Get Google Drive files here
# To view drive file, go to the link:
# https://drive.google.com/file/d/<file_id>


if [[ -n "${SINGSTORAGE}" ]]; then
    cd $SINGSTORAGE
fi

if [[ -d storage/external ]]; then
    echo "storage exists, starting download"
else
    mkdir --parents storage/external
fi

cd storage/external

if [[ ! -d joern-cli ]]; then
    wget https://github.com/joernio/joern/releases/download/v2.0.331/joern-cli.zip
    unzip joern-cli.zip
    rm joern-cli.zip
else
    echo "Already downloaded Joern"
fi