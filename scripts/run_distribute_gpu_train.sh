get_real_path() {
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

if [ $# != 4 ]
then
    echo "=============================================================================================================="
    echo "Usage: bash scripts/run_distribute_gpu_train.sh [DATASET] [CONFIG_PATH] [DEVICE_NUM] [CUDA_VISIBLE_DEVICES]"
    echo "Please run the script as: "
    echo "bash scripts/run_distribute_gpu_train.sh [DATASET] [CONFIG_PATH] [DEVICE_NUM] [CUDA_VISIBLE_DEVICES]"
    echo "for example: bash run_distribute_gpu_train.sh /absolute/path/to/data /absolute/path/to/config 8 0,1,2,3,4,5,6,7"
    echo "=============================================================================================================="
    exit 1
fi
PROJECT_DIR=$(cd "$(dirname "$0")" || exit; pwd)

export DEVICE_NUM=$3
export RANK_SIZE=$3
export CUDA_VISIBLE_DEVICES=$4

if [ -d "output_dis" ]; then
    rm -rf ./output_dis
fi
mkdir "./output_dis"

DATASET=$(get_real_path $1)
CONFIG_PATH=$(get_real_path $2)
echo "========== start run training ==========="
echo "please get log at train_dis.log"
mpirun -n $DEVICE_NUM --output-filename log_output --merge-stderr-to-stdout --allow-run-as-root\
 python ${PROJECT_DIR}/../train.py --data_path=$DATASET --config_path=$CONFIG_PATH --output_path './output_dis' --run_distribute=True --device_target=GPU > train_dis.log 2>&1 &

