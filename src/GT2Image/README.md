# GT2Image

## environment

<span style="font-weight:bold;">You can use the environment that has already been installed on ai2 : <span style="color:#CC0000;">Instructg2i</span></span>


If you want to setup the environment by yourself , The necessary packages are listed below:

numpy = 1.23.5

huggingface_hub = 0.25.2

torchmetrics

pytorch==2.0.1

torchvision==0.15.2

torchaudio==2.0.2

pytorch-cuda=11.8

transformer = 4.37.2

diffusers==0.27.0

accelerate

datasets

wandb



## preprocess dataset

因为dgl包的问题，建议切换到已经配好的gfm环境运行下面的代码：（目前Movies数据集已经进行了预处理，可以跳过此步，直接进行测试）

```
python process.py --train_ratio=(你希望划分的训练集占总体的比例) \
				--read_data_path=(数据集.pt文件路径) \
				--csv_path=(数据集.csv文件路径) \
				--save_data_path=(处理后的数据集保存位置) \
```





## train

【注意】：为了解决路径问题，建议调试代码和运行代码时在<span style="font-weight:bold;"> ./UniMAG/src/GT2Image </span>文件夹下进行

训练：(目前已经有一个简版预训练好的模型，在文件夹/home/ai/dyz/UniMAG/src/GT2Image/pretrained, 请把整个pretrained文件夹复制到你的GT2Image文件夹下，可以跳过训练步骤)

step1: 定位到对应文件夹

```bash
cd src/GT2Image
```

step2: 打开config文件夹（在GT2Image下），建立对应预训练数据集的train.json , 参数的设置可以参考template.json和train_Movies.json , test_Movies.json



step3：开始训练

```bash
python Instructg2i/train.py
```





## test

step1: 定位到对应文件夹

```bash
cd src/GT2Image
```

step2: 打开config文件夹（在GT2Image下），建立对应预训练数据集的test.json , 参数的设置可以参考template.json和train_Movies.json , test_Movies.json



step3: 开始测试

```
python Instructg2i/test.py
```

