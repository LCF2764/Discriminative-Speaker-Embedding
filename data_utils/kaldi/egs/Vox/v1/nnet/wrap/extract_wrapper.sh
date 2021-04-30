#!/bin/bash

env=
gpuid=-1
min_chunk_size=25
chunk_size=10000
normalize=false
node="output"

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
  echo "Usage: $0 [options] <nnet-dir> <data> <embeddings-dir>"
  echo "Options:"
  echo "  --gpuid <-1>"
  echo "  --min-chunk-size <25>"
  echo "  --chunk-size <10000>"
  echo "  --normalize <false>"
  echo "  --node <output>"
  echo ""
  exit 100
fi

nnetdir=$1
feat=$2
dir=$3

if $normalize; then
  cmdopt_norm="--normalize"
fi

#export PYTHONPATH=`pwd`/../../:$PYTHONPATH

python nnet/lib/extract.py --gpu $gpuid --node $node --min-chunk-size $min_chunk_size --chunk-size $chunk_size $cmdopt_norm\
         "$nnetdir" "$feat" "$dir"
#deactivate