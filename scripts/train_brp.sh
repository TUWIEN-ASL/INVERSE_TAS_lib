#!/usr/bin/env bash
if [ -f $1 ]; then
  config=$1
else
  echo "need a config file"
  exit
fi

# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate INVERSE_TAS
which python

type=$(python -c "import yaml;print(yaml.safe_load(open('${config}'))['network']['type'])")
arch=$(python -c "import yaml;print(yaml.safe_load(open('${config}'))['network']['arch'])")
dataset=$(python -c "import yaml;print(yaml.safe_load(open('${config}'))['data']['dataset'])")
path=$(python -c "from configs.paths import project_path;print(f'{project_path}/models/BridgePrompt/train.py')")

now=$(date +"%Y%m%d_%H%M%S")
mkdir -p exp/${type}/${arch}/${dataset}/${now}
python -u ${path}  --config ${config} --log_time $now 2>&1|tee exp/${type}/${arch}/${dataset}/${now}/$now.log
