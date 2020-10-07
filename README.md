# *Imagefy*

## *What is It?*
---
- A AI Driven Module that will help you choose the best pictures

## *How Does It do that?*
---
- Imagefy Uses KMeans Clustering & InceptionResNetV2 CNN in order to Group  Images by Similarity and then choose the best image out of every group

## *Recomended setup*
---
- *Make Sure you have the appropriate Drivers for Tensorflow*
- *List of drivers & installtion instructions can be found [here](https://www.tensorflow.org/install/gpu)* 

```
1. git clone https://github.com/yuval7451/Imagefy.git
2. cd Imagefy
3. virtualenv imagefyenv
4. imagefyenv\Scripts\activate
5. pip install -r requirments.txt
```
## Usage & Help
---
- Help
```
python3 imagefy\main.py --help
```
- Usage
```
python3 imagefy\main.py -d "YOUR\IMAGE\DIR\*" -o "YOU\OUTPUT\DIR" -v <!log at INFO or DEBUG> -s <!image size> --tensorboard <!log tensorboard output> -e NUM_EPOCHS -b BATCH_SIZE -c NUM_CLUSTERS --top <!the amount of images to consider top from each cluster> 
```