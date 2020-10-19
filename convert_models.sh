#!/bin/bash

mkdir -p converted-models

for MODEL in 'hybrid_finetuned_fc6+' 'hybrid_finetuned_all' 'vgg19_finetuned_fc6+' 'vgg19_finetuned_all'; do
  pushd original-models/${MODEL}
  mmtoir -f caffe -w snapshot_iter_*.caffemodel -n deploy.prototxt -d ir
  mmtocode -f pytorch -n ir.pb  -w ir.npy -dw ${MODEL}.pth -d model.py
  popd
  mv original-models/${MODEL}/${MODEL}.pth converted-models/
done

mv original-models/hybrid_finetuned_all/model.py alexnet.py
mv original-models/vgg19_finetuned_all/model.py vgg19.py
