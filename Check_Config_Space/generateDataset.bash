# 进入批处理文件所在目录
cd "$(dirname "$0")"
# 如果有-r 参数并且data目录存在，则删除旧的输出目录
if [[ "$1" == "-r" && -d "data" ]]; then
    rm -rf data
fi
# 进入脚本文件夹
cd ./DatasetGenerator
# 执行 Python 脚本
echo Generating training dataset...
python run.py -s 1 1 -p 10 -r 0.001 -n 128 -d ../data/train -t 32
python run.py -s 1 1 -p 30 -r 0.001 -n 128 -d ../data/train -t 32
python run.py -s 1 1 -p 50 -r 0.001 -n 128 -d ../data/train -t 32
python run.py -s 1 1 -p 70 -r 0.001 -n 128 -d ../data/train -t 32
python run.py -s 1 1 -p 90 -r 0.001 -n 128 -d ../data/train -t 32
# 验证集
echo Generating validation dataset...
python run.py -s 1 1 -p 10 -r 0.01 -n 16 -d ../data/val -t 32
python run.py -s 1 1 -p 30 -r 0.01 -n 16 -d ../data/val -t 32
python run.py -s 1 1 -p 50 -r 0.01 -n 16 -d ../data/val -t 32
python run.py -s 1 1 -p 70 -r 0.01 -n 16 -d ../data/val -t 32
python run.py -s 1 1 -p 90 -r 0.01 -n 16 -d ../data/val -t 32
# 测试集
echo Generating test dataset...
python run.py -s 1 1 -p 20 -r 0.01 -n 16 -d ../data/test -t 32
python run.py -s 1 1 -p 40 -r 0.01 -n 16 -d ../data/test -t 32
python run.py -s 1 1 -p 60 -r 0.01 -n 16 -d ../data/test -t 32
python run.py -s 1 1 -p 80 -r 0.01 -n 16 -d ../data/test -t 32
python run.py -s 1 1 -p 100 -r 0.01 -n 16 -d ../data/test -t 32