#!/bin/bash

ABSOLUTE_SCRIPT=`readlink -m $0`
SCRIPTS_DIR=`dirname ${ABSOLUTE_SCRIPT}`
. ${SCRIPTS_DIR}/setup_java_env.sh 6G

runJavaCommandAndExit org.janelia.render.client.BoxClient $*