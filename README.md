# Visual Prompt Analysis

## Set Up

```bash
conda create -n sam3 python=3.11 -y
conda activate sam3
pip install -r requirements.txt

```
### Dataset
```bash
mkdir data
# coco
mkdir coco
cd coco
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip val2017.zip
unzip annotations_trainval2017.zip
cd ..

# coco-o
gdown 1aBfIJN0zo_i80Hv4p7Ch7M8pRzO37qbq
unzip ood_coco

```
#### coco-c Dataset
```bash
cd ../SAM3/coco-c
python dataset.py

```
