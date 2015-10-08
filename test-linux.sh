#!/bin/bash
cd "$cuSIFT_BUILD_DIR"
make test
# TODO: to be enabled later    
# - cat ./Testing/Temporary/LastTest.log | grep -B 15 -A 5 -n -i "failed"