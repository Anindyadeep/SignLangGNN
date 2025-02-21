# **SignLangGNN**

<p align="center">
  <img src="Images/bg.png" />
</p>


When **GNNs** 💜 **MediaPipe**. This is a starter project where I tried to implement some traditional image classification problem i.e. the ASL sign language classification problem. The twist here is we used the **graph** generated from the hand images using mediapipe. And the graph I got, I extrated the `{x, y, z}` co-ordinates of the nodes and also the edge index for the connecteion and translated this image classification problem to a graph classiciation problem. 

**Project Structure**
```
--------- Data
            |___ CSVs # containing the co-ordinates of per images
            |___ raw
                   |___ train.csv
                   |___ valid.csv
                   |___ test.csv 
            |___ ImageData
                   |___ asl_alphabet_test
                            |___ A/
                            |___ B/ 
                            ....
                            |___ space

                   |___ asl_alphabet_train
            |
            |___ Models # the GNN models
            |___ src
                   |__ dataset.py # pyg custom data
                   |__ train.py   # train loop
                   |__ utils.py   # different utility functions
            |
            |___ main.py # from data to train
            |___ run.py  # real time video visualization
```
I used `PyTorch geometric` and `PyTorch` for the project. To view the results in details head over to the `IPYNB` folder and see the first IPYNB file. To run this project first clone this repo using this command:

```
git clone https://github.com/Anindyadeep/SignLangGNN
```

After that run the `main.py` using this command. Other things will be managed automatically, provided al,l the essential libraries are installed.

```
python3 main.py
```
---

## **Initial Results**

The traning and validation process went smooth as with a very simple base model it gave an `train acc` of `0.85` and `validation acc` of `0.86`. It also provided an `test acc` of `0.84`. The model was run for 8 epochs. The model also gets confused with some sort of examples and we can say that it currently suffers from adverserial attacks.

## **Improvements**

These are the improvements we can do with this project:

1. Improved GNN models. We can make more robust and complex models and improve the performance.

2. Adding edge features. Some of the edge features like `distance between two nodes` and the `angle between two nodes` could produce some potential improvements to the performance of our model.


## **Future Works and Citation**
Using **Temporal Graph Neural Nets** could make more robust and accurate model for this kind of problem. But for that we need temporal data like videos instaed of images, so that we could generate `static temporal graphs` and compute on them as a dynamic graph sequence problem. Feel free to cite my work, Thank you.

```
@misc{Sannigrahi2022SignLangGNN,
  author = {Anindyadeep Sannigrahi},
  title = {SignLangGNN: ASL Sign Language Classification using Graph Neural Networks and MediaPipe},
  year = {2022},
  url = {https://github.com/Anindyadeep/SignLangGNN}
}
```
