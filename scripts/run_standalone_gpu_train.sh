get_real_path() {
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

if [ $# != 2 ] && [ $# != 3 ]
then
    echo "=============================================================================================================="
    echo "Please run the script as: "
    echo "bash scripts/run_standalone_gpu_train.sh [DATASET] [CONFIG_PATH] [DEVICE_ID](option, default is 0)"
    echo "for example: bash run_standalone_gpu_train.sh /path/to/data/ /path/to/config/ 0"
    echo "=============================================================================================================="
    exit 1
fi
PROJECT_DIR=$(cd "$(dirname "$0")" || exit; pwd)
export DEVICE_ID=0
if [ $# != 2 ]
then
  export DEVICE_ID=$3
fi

mkdir "./output"

DATASET=$(get_real_path $1)
CONFIG_PATH=$(get_real_path $2)
echo "========== start run training ==========="
echo "please get log at train.log"
python ${PROJECT_DIR}/../train.py --data_path=$DATASET --config_path=$CONFIG_PATH --output_path './output' --run_distribute=False --device_target=GPU > train.log 2>&1 &
# python3 train.py --data_path=../dataset/dataset_for_dlinknet/train/ --config_path=dlinknet_config.yaml --output_path='./output' --run_distribute=False --device_target=GPU
