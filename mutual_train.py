from __future__ import print_function
import numpy as np
import cPickle
import tensorflow as tf
import cv2
import random

from utilities import label_img_to_color

from model_mutual1 import UNet_model
from model_mutual2 import UNet_model2

project_dir = "/scratch/lxx/segmentation-master/"

data_dir = project_dir + "data/"
display_num=100

# change this to not overwrite all log data when you train the model:
model_id = "5-1"
model2_id = "5-2"

batch_size = 4
img_height = 512
img_width = 1024

model = UNet_model(model_id, img_height=img_height, img_width=img_width,
            batch_size=batch_size)

model2 = UNet_model2(model2_id, img_height=img_height, img_width=img_width,
            batch_size=batch_size)

no_of_classes = model.no_of_classes

# load the mean color channels of the train imgs:
train_mean_channels = cPickle.load(open("data/mean_channels.pkl"))

# load the training data from disk:
train_img_paths = cPickle.load(open(data_dir + "train_img_paths.pkl"))
train_trainId_label_paths = cPickle.load(open(data_dir + "train_trainId_label_paths.pkl"))
train_data = zip(train_img_paths, train_trainId_label_paths)

# compute the number of batches needed to iterate through the training data:
no_of_train_imgs = len(train_img_paths)
no_of_batches = int(no_of_train_imgs/batch_size)

# load the validation data from disk:
val_img_paths = cPickle.load(open(data_dir + "val_img_paths.pkl"))
val_trainId_label_paths = cPickle.load(open(data_dir + "val_trainId_label_paths.pkl"))
val_data = zip(val_img_paths, val_trainId_label_paths)

# compute the number of batches needed to iterate through the val data:
no_of_val_imgs = len(val_img_paths)
no_of_val_batches = int(no_of_val_imgs/batch_size)

# define params needed for label to onehot label conversion:
layer_idx = np.arange(img_height).reshape(img_height, 1)
component_idx = np.tile(np.arange(img_width), (img_height, 1))

# evaluate model1
def evaluate_on_val():
    random.shuffle(val_data)
    val_img_paths, val_trainId_label_paths = zip(*val_data)

    val_batch_losses = []
    raw_losses=[]
    mutual_losses=[]
    batch_pointer = 0
    for step in range(no_of_val_batches):
        batch_imgs = np.zeros((batch_size, img_height, img_width, 3), dtype=np.float32)
        batch_onehot_labels = np.zeros((batch_size, img_height, img_width,
                    no_of_classes), dtype=np.float32)

        for i in range(batch_size):
            # read the next img:
            img = cv2.imread(val_img_paths[batch_pointer + i], -1)
            img = img - train_mean_channels
            batch_imgs[i] = img

            # read the next label:
            trainId_label = cv2.imread(val_trainId_label_paths[batch_pointer + i], -1)

            # convert the label to onehot:
            onehot_label = np.zeros((img_height, img_width, no_of_classes), dtype=np.float32)
            onehot_label[layer_idx, component_idx, trainId_label] = 1
            batch_onehot_labels[i] = onehot_label

        batch_pointer += batch_size

        batch_feed_dict = model.create_feed_dict(imgs_batch=batch_imgs,
                    onehot_labels_batch=batch_onehot_labels)

        # run a forward pass, get the batch loss and the logits:
        raw_loss,logits = sess.run([model.loss_raw,model.logits],
                    feed_dict=batch_feed_dict)

        # val_batch_losses.append(batch_loss)
        raw_losses.append(raw_loss)
        # mutual_losses.append(mutual_loss)
        if (step+1)%30==0:
            print ("epoch: %d/%d, val step: %d/%d,loss_raw: %g" % (epoch+1,
                    no_of_epochs, step+1, no_of_val_batches,raw_loss))

        if step < 4:
            # save the predicted label images to disk for debugging and
            # qualitative evaluation:
            predictions = np.argmax(logits, axis=3)
            for i in range(batch_size):
                pred_img = predictions[i]
                label_img_color = label_img_to_color(pred_img)
                cv2.imwrite((model.debug_imgs_dir + "val_" + str(epoch) + "_" +
                            str(step) + "_" + str(i) + ".png"), label_img_color)

    # val_loss = np.mean(val_batch_losses)
    val_loss_raw = np.mean(raw_losses)
    # val_loss_mutual=np.mean(mutual_losses)
    print ('model1 val loss_raw:%g'%(val_loss_raw))
    return val_loss_raw

# evaluate model2
def evaluate_on_val2():
    random.shuffle(val_data)
    val_img_paths, val_trainId_label_paths = zip(*val_data)

    val_batch_losses = []
    raw_losses = []
    mutual_loss = [] 
    batch_pointer = 0
    for step in range(no_of_val_batches):
        batch_imgs = np.zeros((batch_size, img_height, img_width, 3), dtype=np.float32)
        batch_onehot_labels = np.zeros((batch_size, img_height, img_width,
                    no_of_classes), dtype=np.float32)

        for i in range(batch_size):
            # read the next img:
            img = cv2.imread(val_img_paths[batch_pointer + i], -1)
            img = img - train_mean_channels
            batch_imgs[i] = img

            # read the next label:
            trainId_label = cv2.imread(val_trainId_label_paths[batch_pointer + i], -1)

            # convert the label to onehot:
            onehot_label = np.zeros((img_height, img_width, no_of_classes), dtype=np.float32)
            onehot_label[layer_idx, component_idx, trainId_label] = 1
            batch_onehot_labels[i] = onehot_label

        batch_pointer += batch_size

        batch_feed_dict = model2.create_feed_dict(imgs_batch=batch_imgs,
                    onehot_labels_batch=batch_onehot_labels)

        # run a forward pass, get the batch loss and the logits:
        raw_loss,logits = sess.run([model2.loss_raw,model2.logits],
                    feed_dict=batch_feed_dict)

        # val_batch_losses.append(batch_loss)
        raw_losses.append(raw_loss)
        # mutual_losses.append(mutual_loss)
        if (step+1)%30==0:
            print ("epoch: %d/%d, val step: %d/%d,loss_raw: %g" % (epoch+1,
                    no_of_epochs, step+1, no_of_val_batches,raw_loss))

        if step < 4:
            # save the predicted label images to disk for debugging and
            # qualitative evaluation:
            predictions = np.argmax(logits, axis=3)
            for i in range(batch_size):
                pred_img = predictions[i]
                label_img_color = label_img_to_color(pred_img)
                cv2.imwrite((model2.debug_imgs_dir + "val_" + str(epoch) + "_" +
                            str(step) + "_" + str(i) + ".png"), label_img_color)

    # val_loss = np.mean(val_batch_losses)
    val_loss_raw = np.mean(raw_losses)
    # val_loss_mutual=np.mean(mutual_losses)
    print ('model2 val loss_raw:%g'%(val_loss_raw))
    return val_loss_raw
    
def train_data_iterator():
    random.shuffle(train_data)
    train_img_paths, train_trainId_label_paths = zip(*train_data)

    batch_pointer = 0
    for step in range(no_of_batches):
        # get and yield the next batch_size imgs and onehot labels from the train data:
        batch_imgs = np.zeros((batch_size, img_height, img_width, 3), dtype=np.float32)
        batch_onehot_labels = np.zeros((batch_size, img_height, img_width,
                    no_of_classes), dtype=np.float32)

        for i in range(batch_size):
            # read the next img:
            img = cv2.imread(train_img_paths[batch_pointer + i], -1)
            img = img - train_mean_channels
            batch_imgs[i] = img

            # read the next label:
            trainId_label = cv2.imread(train_trainId_label_paths[batch_pointer + i], -1)

            # convert the label to onehot:
            onehot_label = np.zeros((img_height, img_width, no_of_classes), dtype=np.float32)
            onehot_label[layer_idx, component_idx, trainId_label] = 1
            batch_onehot_labels[i] = onehot_label

        batch_pointer += batch_size

        yield (batch_imgs, batch_onehot_labels)

no_of_epochs = 30

# create a saver for saving all model variables/parameters:
saver = tf.train.Saver(tf.trainable_variables(), write_version=tf.train.SaverDef.V2)

# initialize all log data containers:
train_loss_per_epoch = []
val_loss_per_epoch = []
train_loss_per_epoch_raw = []
val_loss_per_epoch_raw = []
train_loss_per_epoch_mutual = []
val_loss_per_epoch_mutual = []

train_loss2_per_epoch = []
val_loss2_per_epoch = []
train_loss2_per_epoch_raw = []
val_loss2_per_epoch_raw = []
train_loss2_per_epoch_mutual = []
val_loss2_per_epoch_mutual = []

# initialize a list containing the 5 best val losses (is used to tell when to
# save a model checkpoint):
best_epoch_losses = [1000, 1000, 1000, 1000, 1000]
best_epoch_losses2 = [1000, 1000, 1000, 1000, 1000]

loss_log = open("%strain_loss_log.txt"% model.model_dir, "w")
loss2_log = open("%strain_loss2_log.txt"% model2.model_dir, "w")

with tf.Session() as sess:
    # initialize all variables/parameters:
    init = tf.global_variables_initializer()
    sess.run(init)

    model.load_initial_weights(sess)

    model1_prob_batch = np.zeros((batch_size, img_height, img_width,
                    no_of_classes), dtype=np.float32)
    model2_prob_batch = np.zeros((batch_size, img_height, img_width,
                    no_of_classes), dtype=np.float32)


    for epoch in range(no_of_epochs):
        print ("###########################")
        print ("######## NEW EPOCH ########")
        print ("###########################")
        print ("epoch: %d/%d" % (epoch+1, no_of_epochs))

        # run an epoch and get all batch losses:
        batch_losses = []
        batch_losses_raw = [] 
        batch_losses_mutual = []

        batch_losses2 = []
        batch_losses2_raw = [] 
        batch_losses2_mutual = []

        for step, (imgs, onehot_labels) in enumerate(train_data_iterator()):
            # create a feed dict containing the batch data:

            batch_feed_dict = model.create_feed_dict(imgs_batch=imgs,
                        onehot_labels_batch=onehot_labels,mate_prob_batch=model2_prob_batch)

            batch_feed_dict2 = model2.create_feed_dict(imgs_batch=imgs,
                        onehot_labels_batch=onehot_labels,mate_prob_batch=model1_prob_batch)

            batch_loss, batch_loss_raw, batch_loss_mutual,model2_prob_batch, _ = sess.run([model2.loss, model2.loss_raw, model2.loss_mutual, model2.logits, model2.train_op],
                        feed_dict=batch_feed_dict2)
            batch_losses2.append(batch_loss)
            batch_losses2_raw.append(batch_loss_raw)
            batch_losses2_mutual.append(batch_loss_mutual)

            if((step+1)%display_num==0):
                print ("step:%d/%d,model2 loss:%g,raw:%g,mutual:%g" % (step+1, no_of_batches, batch_loss,batch_loss_raw,batch_loss_mutual),end='')

            batch_loss, batch_loss_raw, batch_loss_mutual,model1_prob_batch, _ = sess.run([model.loss, model.loss_raw, model.loss_mutual, model.logits, model.train_op],
                        feed_dict=batch_feed_dict)
            batch_losses.append(batch_loss)
            batch_losses_raw.append(batch_loss_raw)
            batch_losses_mutual.append(batch_loss_mutual)

            if((step+1)%display_num==0):
                print ("model1 loss:%g,raw:%g,mutual:%g" % (batch_loss,batch_loss_raw,batch_loss_mutual))

        # compute the train epoch loss:
        train_epoch_loss = np.mean(batch_losses)
        # save the train epoch loss:
        train_loss_per_epoch.append(train_epoch_loss)
        # save the train epoch losses to disk:
        cPickle.dump(train_loss_per_epoch, open("%strain_loss_per_epoch.pkl"
                    % model.model_dir, "w"))

        train_epoch_loss_raw = np.mean(batch_losses_raw)
        # save the train epoch loss:
        train_loss_per_epoch_raw.append(train_epoch_loss_raw)
        # save the train epoch losses to disk:
        cPickle.dump(train_loss_per_epoch_raw, open("%strain_loss_per_epoch.pkl"
                    % model.model_dir, "w"))

        train_epoch_loss_mutual = np.mean(batch_losses_mutual)
        # save the train epoch loss:
        train_loss_per_epoch_mutual.append(train_epoch_loss_mutual)
        # save the train epoch losses to disk:
        cPickle.dump(train_loss_per_epoch_mutual, open("%strain_loss_per_epoch.pkl"
                    % model.model_dir, "w"))

        print ("model 1 training loss: %g raw: %g mutual: %g" % (train_epoch_loss,train_epoch_loss_raw,train_epoch_loss_mutual))

        # run the model on the validation data:
        val_loss = evaluate_on_val()

        # save the val epoch loss:
        val_loss_per_epoch.append(val_loss)
        # save the val epoch losses to disk:
        cPickle.dump(val_loss_per_epoch, open("%sval_loss_per_epoch.pkl"\
                    % model.model_dir, "w"))
        # print >>loss2_log, "validation loss: %g" % val_loss
        # print ("model1 validation loss: %g" % val_loss)

        if val_loss < max(best_epoch_losses): # (if top 5 performance on val:)
            # save the model weights to disk:
            checkpoint_path = (model.checkpoints_dir + "model_" +
                        model.model_id + "_epoch_" + str(epoch + 1) + ".ckpt")
            saver.save(sess, checkpoint_path)
            print ("checkpoint saved in file: %s" % checkpoint_path)

            # update the top 5 val losses:
            index = best_epoch_losses.index(max(best_epoch_losses))
            best_epoch_losses[index] = val_loss


        # compute the train epoch loss:
        train_epoch_loss2 = np.mean(batch_losses2)
        # save the train epoch loss:
        train_loss2_per_epoch.append(train_epoch_loss2)
        # save the train epoch losses to disk:
        cPickle.dump(train_loss2_per_epoch, open("%strain_loss_per_epoch.pkl"
                    % model2.model_dir, "w"))

        train_epoch_loss2_raw = np.mean(batch_losses2_raw)
        # save the train epoch loss:
        train_loss2_per_epoch_raw.append(train_epoch_loss2_raw)
        # save the train epoch losses to disk:
        cPickle.dump(train_loss2_per_epoch_raw, open("%strain_loss_per_epoch.pkl"
                    % model.model_dir, "w"))

        train_epoch_loss2_mutual = np.mean(batch_losses2_mutual)
        # save the train epoch loss:
        train_loss2_per_epoch_mutual.append(train_epoch_loss2_mutual)
        # save the train epoch losses to disk:
        cPickle.dump(train_loss2_per_epoch_mutual, open("%strain_loss_per_epoch.pkl"
                    % model.model_dir, "w"))

        print ("model 2 training loss: %g raw: %g mutual: %g" % (train_epoch_loss2,train_epoch_loss2_raw,train_epoch_loss2_mutual))

        # run the model on the validation data:
        val_loss2 = evaluate_on_val2()

        # save the val epoch loss:
        val_loss2_per_epoch.append(val_loss2)
        # save the val epoch losses to disk:
        cPickle.dump(val_loss2_per_epoch, open("%sval_loss_per_epoch.pkl"\
                    % model2.model_dir, "w"))
        # print >>loss2_log, "validation loss: %g" % val_loss
        # print("model 2 validation loss: %g" % val_loss2)

        if val_loss2 < max(best_epoch_losses2): # (if top 5 performance on val:)
            # save the model weights to disk:
            checkpoint_path = (model2.checkpoints_dir + "model_" +
                        model2.model_id + "_epoch_" + str(epoch + 1) + ".ckpt")
            saver.save(sess, checkpoint_path)
            print ("checkpoint saved in file: %s" % checkpoint_path)

            # update the top 5 val losses:
            index = best_epoch_losses2.index(max(best_epoch_losses2))
            best_epoch_losses2[index] = val_loss2


