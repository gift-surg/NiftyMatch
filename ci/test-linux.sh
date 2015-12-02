#!/bin/bash
cd "$cuSIFT_BUILD_DIR"
make test
ctestlog=Testing/Temporary/LastTest.log
cat $ctestlog
cat $ctestlog | grep -i failed
# last exit status 0 means, grep found failures!
if [ $? == 0 ]; then
    exit 1;
else
    echo 0;
fi