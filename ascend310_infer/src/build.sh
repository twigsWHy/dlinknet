MINDSPORE_PATH="`pip show mindspore-ascend | grep Location | awk '{print $2"/mindspore"}' | xargs realpath`"
if [[ ! $MINDSPORE_PATH ]];then
    MINDSPORE_PATH="`pip show mindspore | grep Location | awk '{print $2"/mindspore"}' | xargs realpath`"
fi
cmake . -DMINDSPORE_PATH=$MINDSPORE_PATH
make