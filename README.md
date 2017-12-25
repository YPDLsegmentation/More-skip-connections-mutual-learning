# More-skip-connections-mutual-learning

### More skip connections experiment code

model_0.py   basic model, based on vgg16 and U-net

model_1.py   basic model + conv1_2 skip connection

model_2.py   basic model + conv1_1 & conv1_2 skip connection

train.py          train independent model, please import correct model before running 


### Mutual learning experiment code

model_mutual1.py  student network1, based on vgg16 and U-net

model_mutual2.py  student network2, based on vgg19 and U-net

mutual_train.py       train two student networks above
