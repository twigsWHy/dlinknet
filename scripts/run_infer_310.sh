if [[ $# -lt 3 || $# -gt 4 ]]; then
    echo "Usage: bash run_infer_310.sh [DATA_PATH] [LABEL_PATH] [MINDIR_PATH] [DEVICE_ID]
    DEVICE_ID is optional, it can be set by environment variable device_id, otherwise the value is zero.
    NEED_PREPROCESS means weather need preprocess or not, it's value is 'y' or 'n'."
exit 1
fi

get_real_path(){
    if [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        echo "$(realpath -m $PWD/$1)"
    fi
}

data_path=$(get_real_path $1)
label_path=$(get_real_path $2)
model=$(get_real_path $3)
device_id=0
if [ $# == 4 ]; then
    device_id=$4
fi

echo "mindir name: "$model
echo "dataset path: "$data_path
echo "device id: "$device_id

function compile_app()
{
    cd ../ascend310_infer/src || exit
    if [ -f "Makefile" ]; then
        make clean
    fi
    bash build.sh &> build.log
}

function infer()
{
    cd - || exit
    if [ -d result_Files ]; then
        rm -rf ./result_Files
    fi
     if [ -d time_Result ]; then
        rm -rf ./time_Result
    fi
    mkdir result_Files
    mkdir time_Result
    ../ascend310_infer/src/dlinknet --mindir_path=$model --dataset_path=$data_path --device_id=$device_id &> infer.log
}

function cal_acc()
{
    python ../postprocess.py result_Files $label_path &> acc.log &
}

compile_app
if [ $? -ne 0 ]; then
    echo "compile app code failed"
    exit 1
fi
infer
if [ $? -ne 0 ]; then
    echo "execute inference failed"
    exit 1
fi
cal_acc
if [ $? -ne 0 ]; then
    echo "calculate accuracy failed"
    exit 1
fi