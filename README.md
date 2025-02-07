# 64M的迷你中文大模型 MyLlama64M

![](README.files/image.png)

训练自己的迷你中文大模型！
Train your own small language model on your own PC

### 对话展示 Dialogue example

> 问：什么是鹦鹉？
> 答：鹦鹉是一种很可爱的鸟。它们喜欢和人一起生活，喜欢吃水果和小虫子。鹦鹉的身体很长，有的像小房子，有的像一只小猴子。鹦鹉的声音很特别，听起来像在唱歌。
> 
> 鹦鹉的嘴巴很特别，它们喜欢吃小虫子和小动物。鹦鹉的嘴巴上有很多小点点，这些点可以帮助它们咬住食物。鹦鹉的嘴巴很灵敏，可以用来吃食物。
> 
> 鹦鹉可以在很多地方找到，比如公园、花园和农场。它们有的会在晚上出来，有的会在树上或者其他地方。鹦鹉喜欢和其他鹦鹉在一起，有时候它们会叫得很大声，像是在玩游戏。
> 
> 鹦鹉的宝宝叫小鸟。小鸟出生时很小，妈妈会给它们喂食。小鸟出生后，会跟着妈妈学习飞来飞去，怎么吃东西。鹦鹉的妈妈会照顾它们，给它们吃好吃的。
> 
> 鹦鹉也有一些特别的习惯。比如，它们会用嘴巴抓住食物，或者用嘴巴喝水。鹦鹉的声音非常特别，有时候它们会发出“啾啾”的声音，像是在说“咕咕”的声音。
> 
> 鹦鹉是一种可爱的动物。它们让我们的生活更加有趣。我们可以和鹦鹉一起玩，听它们发出的声音。鹦鹉是我们生活中很重要的一部分。

### 如何开始 How to 
####  安装依赖项 Install the requirements 
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install zhconv
pip install wandb
pip install transformers
pip install jsonlines
pip install matplotlib
pip install torchao --extra-index-url https://download.pytorch.org/whl/cu124
```
#### 下载训练数据 Download the dataset 
```bash
git clone https://huggingface.co/datasets/fzmnm/TinyHelen-zh data/train_data
mkdir data/train_data/tinyHelen-zh
mv data/train_data/*.jsonl data/train_data/tinyHelen-zh/
```
#### 预处理训练数据 Tokenize the dataset 
```bash
python pretokenize.py data/train_data/tinyHelen-zh data/tokenized_train_data/tinyHelen-zh
cp ./weights.yaml data/tokenized_train_data
it takes a few minutes
```
需要几分钟
it takes a few minutes

可以再自己准备一些测试数据，用同样的方法预处理好了之后放在 `data/tokenized_train_data/eval` 文件夹中
you can also tokenize something and put them in `data/tokenized_train_data/eval` folder

#### （可选）使用wandb来在线查看训练进度 (Optional) Set up wandb logging 
在wandb.ai创建账号，然后
go to wandb.ai and create an account, then
```bash
wandb login
```
#### 开始炼丹 Start Training Your Model
修改 `train.py` 中的 `wandb_log=True`
调整 `batch_size` 以适应你的 GPU 内存
modify `wandb_log=True` in train.py
adjust `batch_size` to fit your gpu memory

```bash
python train.py
```

在我的4090m（相当于4070，16GB显存）上，1次iter需要6秒。通常需要5000次iter就能得到一个好的结果。所以只需一个晚上，你就可以制作自己的LLM了！

On my 4090m (equivalent to 4070, 16GB vram), 1 iteration takes 6 sec. Usually, it takes 5k iterations to get a good result. So just after one night, you can make your own LLM!

#### 试玩 Play with your LLM
```bash
python inference.py data/checkpoints/my_llama_v1_64M_fp8/latest.pt -c
```
加-c 是接龙模式。不加-c是问答模式（可以通过sft进一步调优）
-c if for completion mode. without -c is assistant mode, but need sft to make it more robust