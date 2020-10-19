#!/bin/bash

mkdir -p converted-models/
cd converted-models

for MODEL in 'hybrid_finetuned_fc6+' 'hybrid_finetuned_all' 'vgg19_finetuned_fc6+' 'vgg19_finetuned_all'; do
  if [ ! -f "${MODEL}.pth" ]; then
      echo "Downloading: ${MODEL}.pth"
      wget https://github.com/fabiocarrara/visual-sentiment-analysis/releases/download/torch-models/${MODEL}.pth
  else
      echo "Skipping: ${MODEL}.pth already downloaded"
  fi
done

