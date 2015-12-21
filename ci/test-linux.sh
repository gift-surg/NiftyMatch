#!/bin/bash
cd "$Test_BUILD_DIR"
make test
ctestlog=Testing/Temporary/LastTest.log
cat $ctestlog | grep -i fail -B 25 -A 3
cat $ctestlog | grep -i fail
# last exit status 0 means, grep found failures!
if [ $? == 0 ]; then
    exit 1;
else
    echo 0;
fi