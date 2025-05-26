@echo off
:: 进入批处理文件所在目录
cd /d "%~dp0"
:: 如果有-r 参数并且data目录存在，则删除旧的输出目录
if "%1"=="-r" (
    if exist "data" (
        rmdir /s /q "data"
    )
)
:: 进入脚本文件夹
cd ./DatasetGenerator
:: 执行 Python 脚本
:: 训练集
echo Generating training dataset...
echo "python run.py -s 1 1 -p 10 -r 0.001 -n 128 -d ../data/train -t 8"
python run.py -s 1 1 -p 10 -r 0.1 -n 32 -d ../data/train -t 8
:: 验证集
echo Generating validation dataset...
echo "python run.py -s 1 1 -p 10 -r 0.001 -n 32 -d ../data/val -t 8"
python run.py -s 1 1 -p 10 -r 0.1 -n 32 -d ../data/val -t 8
:: 测试集
echo Generating test dataset...
echo "python run.py -s 1 1 -p 10 -r 0.001 -n 32 -d ../data/test -t 8"
python run.py -s 1 1 -p 10 -r 0.1 -n 32 -d ../data/test -t 8