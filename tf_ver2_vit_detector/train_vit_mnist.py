import time
import train_utils
import numpy as np
import pandas as pd
import pickle as pkl
from utils import convert_to_xywh

import tensorflow as tf
import tf_ver2_vit_detector as tf_obj_detector

# Custom function to parse the data. #
def _parse_image(
    filename, img_rows, img_cols):
    image_string  = tf.io.read_file(filename)
    image_decoded = \
        tf.image.decode_jpeg(image_string, channels=3)
    image_decoded = \
        tf.image.convert_image_dtype(image_decoded, tf.float32)
    image_resized = tf.image.resize(
        image_decoded, [img_rows, img_cols])
    image_resized = tf.ensure_shape(
        image_resized, shape=(img_rows, img_cols, 3))
    return image_resized

def image_augment(img_in, img_bbox, p=0.5):
    if np.random.uniform() >= p:
        p_tmp = np.random.uniform()
        if p_tmp <= 0.50:
            p_uni = np.random.uniform()
            if p_uni <= 0.50:
                tmp_in = tf.image.random_brightness(img_in, 0.25)
            else:
                tmp_in = tf.image.random_contrast(img_in, 0.75, 1.25)
            
            tmp_max = tf.reduce_max(tmp_in).numpy()
            if tmp_max > 1.0:
                tmp_in = tmp_in / tmp_max
            return (tf.constant(tmp_in), img_bbox)
        else:
            # Flip left-right. #
            img_in = img_in[:, ::-1, :]
            
            tmp_bbox = img_bbox
            img_bbox = tmp_bbox[:, ::-1, :]
            
            img_bbox[:, :, 1] = 1.0 - img_bbox[:, :, 1]
            return (img_in, tf.constant(img_bbox))
    else:
        return (img_in, img_bbox)

def train(
    model, n_classes, sub_batch_sz, batch_size, 
    train_data, training_loss, st_step, max_steps, optimizer, 
    ckpt, ck_manager, label_dict, init_lr=1.0e-3, min_lr=1e-6, 
    downsample=32, decay=0.75, display_step=100, step_cool=50, 
    base_rows=320, base_cols=320, disp_rows=320, disp_cols=320, 
    thresh=0.50, save_flag=False, train_loss_log="train_losses.csv"):
    n_data = len(train_data)
    base_dims = min(base_rows, base_cols)
    
    start_time = time.time()
    tot_reg_loss  = 0.0
    tot_cls_loss  = 0.0
    for step in range(st_step, max_steps):
        batch_sample  = np.random.choice(
            n_data, size=batch_size, replace=False)
        
        rnd_scale = np.random.uniform(low=0.6, high=1.3)
        raw_dims  = int(rnd_scale * base_dims)
        if raw_dims % 32 == 0:
            img_dims = int(rnd_scale * 320 / 32) * 32
        else:
            img_dims = (int(rnd_scale * 320 / 32) + 1) * 32
        pad_dims = int((img_dims - raw_dims) / 2.0)
        
        img_boxes = []
        for tmp_idx in batch_sample:
            img_bbox  = np.zeros([
                int(img_dims/downsample), 
                int(img_dims/downsample), n_classes+5])
            tmp_bbox  = np.array(train_data[
                tmp_idx]["objects"]["bbox"])
            tmp_bbox  = convert_to_xywh(tmp_bbox).numpy()
            tmp_class = np.array(train_data[
                tmp_idx]["objects"]["label"])
            
            # Sort by area in descending order. #
            tmp_box_areas = \
                tmp_bbox[:, 2] * tmp_bbox[:, 3] * 100
            
            obj_class  = \
                tmp_class[np.argsort(tmp_box_areas)]
            tmp_sorted = \
                tmp_bbox[np.argsort(tmp_box_areas)]
            for n_box in range(len(tmp_sorted)):
                tmp_object = tmp_sorted[n_box]
                
                tmp_x_cen  = pad_dims + tmp_object[0] * raw_dims
                tmp_y_cen  = pad_dims + tmp_object[1] * raw_dims
                tmp_width  = tmp_object[2] * raw_dims
                tmp_height = tmp_object[3] * raw_dims
                
                if tmp_width < 0 or tmp_height < 0:
                    continue
                
                tmp_w_reg = tmp_width / img_dims
                tmp_h_reg = tmp_height / img_dims
                tmp_w_cen = int(tmp_x_cen / downsample)
                tmp_h_cen = int(tmp_y_cen / downsample)
                tmp_w_off = \
                    (tmp_x_cen - tmp_w_cen*downsample) / downsample
                tmp_h_off = \
                    (tmp_y_cen - tmp_h_cen*downsample) / downsample
                cls_label = int(obj_class[n_box]) + 5
                
                img_bbox[tmp_h_cen, tmp_w_cen, :5] = [
                    tmp_h_off, tmp_w_off, tmp_h_reg, tmp_w_reg, 1.0]
                img_bbox[tmp_h_cen, tmp_w_cen, cls_label] = 1.0
            
            img_boxes.append(img_bbox)
            del img_bbox
            
            if tmp_idx == batch_sample[-1]:
                disp_box = np.zeros([
                    int(disp_rows/downsample), 
                    int(disp_cols/downsample), n_classes+5])
                disp_scale = min(disp_rows, disp_cols)
                
                for n_box in range(len(tmp_sorted)):
                    tmp_object = tmp_sorted[n_box]
                    
                    tmp_x_cen  = tmp_object[0] * disp_rows
                    tmp_y_cen  = tmp_object[1] * disp_cols
                    tmp_width  = tmp_object[2] * disp_rows
                    tmp_height = tmp_object[3] * disp_cols
                    
                    if tmp_width < 0 or tmp_height < 0:
                        continue
                    
                    tmp_w_reg = tmp_width / disp_scale
                    tmp_h_reg = tmp_height / disp_scale
                    tmp_w_cen = int(tmp_x_cen / downsample)
                    tmp_h_cen = int(tmp_y_cen / downsample)
                    tmp_w_off = \
                        (tmp_x_cen - tmp_w_cen*downsample) / downsample
                    tmp_h_off = \
                        (tmp_y_cen - tmp_h_cen*downsample) / downsample
                    cls_label = int(obj_class[n_box]) + 5
                    
                    disp_box[tmp_h_cen, tmp_w_cen, :5] = [
                        tmp_h_off, tmp_w_off, tmp_h_reg, tmp_w_reg, 1.0]
                    disp_box[tmp_h_cen, tmp_w_cen, cls_label] = 1.0
        
        img_files = [
            train_data[x]["image"] for x in batch_sample]
        img_array = [_parse_image(
            x, img_rows=raw_dims, 
            img_cols=raw_dims) for x in img_files]
        img_batch = [tf.image.pad_to_bounding_box(
            x, pad_dims, pad_dims, img_dims, img_dims) for x in img_array]
        
        img_tuple = [image_augment(
            img_batch[x], img_boxes[x]) \
                for x in range(batch_size)]
        img_batch = [tf.expand_dims(
            x, axis=0) for x, y in img_tuple]
        img_bbox  = [tf.expand_dims(
            y, axis=0) for x, y in img_tuple]
        
        # Note that TF parses the image transposed, so the  #
        # bounding boxes coordinates are already transposed #
        # during the formatting of the data.                #
        img_batch = tf.concat(img_batch, axis=0)
        img_bbox  = tf.cast(tf.concat(
            img_bbox, axis=0), tf.float32)
        img_mask  = img_bbox[:, :, :, 4]
        
        epoch = int(step * batch_size / n_data)
        lrate = max(decay**epoch * init_lr, min_lr)
        
        tmp_losses = train_utils.train_step(
            model, sub_batch_sz, img_batch, img_bbox, img_mask, 
            optimizer, downsample=downsample, learning_rate=lrate)
        
        ckpt.step.assign_add(1)
        tot_cls_loss += tmp_losses[0]
        tot_reg_loss += tmp_losses[1]
        
        if (step+1) % display_step == 0:
            avg_reg_loss = tot_reg_loss.numpy() / display_step
            avg_cls_loss = tot_cls_loss.numpy() / display_step
            training_loss.append((step+1, avg_cls_loss, avg_reg_loss))
            
            tot_reg_loss = 0.0
            tot_cls_loss = 0.0
            
            print("Step", str(step+1), "Summary:")
            print("Learning Rate:", str(optimizer.lr.numpy()))
            print("Average Epoch Cls. Loss:", str(avg_cls_loss) + ".")
            print("Average Epoch Reg. Loss:", str(avg_reg_loss) + ".")
            
            elapsed_time = (time.time() - start_time) / 60.0
            print("Elapsed time:", str(elapsed_time), "mins.")
            
            start_time = time.time()
            if (step+1) % step_cool != 0:
                img_title = "ViT Object Detection Result "
                img_title += "at Step " + str(step+1)
                
                train_utils.show_object_boxes(
                    img_batch[-1], img_bbox[-1], 
                    img_dims, downsample=downsample)
                
                disp_box = tf.constant(disp_box)
                train_utils.obj_detect_results(
                    img_files[-1], model, 
                    label_dict, heatmap=True, 
                    thresh=thresh, downsample=downsample, 
                    img_rows=disp_rows, img_cols=disp_cols, 
                    img_box=disp_box, img_title=img_title)
                print("-" * 50)
        
        if (step+1) % step_cool == 0:
            if save_flag:
                # Save the training losses. #
                train_cols_df = ["step", "cls_loss", "reg_loss"]
                train_loss_df = pd.DataFrame(
                    training_loss, columns=train_cols_df)
                train_loss_df.to_csv(train_loss_log, index=False)
                
                # Save the model. #
                save_path = ck_manager.save()
                print("Saved model to {}".format(save_path))
            print("-" * 50)
            
            save_img_file = "C:/Users/admin/Desktop/Data/"
            save_img_file += "Results/MNIST_Object_Detection/"
            save_img_file += "mnist_vit_" + str(step+1) + ".jpg"
            
            img_title = "ViT Object Detection Result "
            img_title += "at Step " + str(step+1)
            
            train_utils.show_object_boxes(
                img_batch[-1], img_bbox[-1], 
                img_dims, downsample=downsample)
            
            disp_box = tf.constant(disp_box)
            train_utils.obj_detect_results(
                img_files[-1], model, label_dict, 
                heatmap=True, downsample=downsample, 
                img_box=disp_box, thresh=thresh, 
                img_rows=disp_rows, img_cols=disp_cols, 
                img_title=img_title, save_img_file=save_img_file)
            time.sleep(120)

# Load the VOC 2012 dataset. #
tmp_path = "C:/Users/admin/Desktop/GitHub_Cloned_Repo/yymnist/"
load_pkl_file = tmp_path + "mnist_objects.pkl"
with open(load_pkl_file, "rb") as tmp_load:
    mnist_data = pkl.load(tmp_load)

# Generate the label dictionary. #
id_2_label = dict([(x, str(x)) for x in range(10)])

# Transformer Parameters. #
n_layers  = 3
num_heads = 4
hidden_sz = 256
ffwd_size = 4*hidden_sz

# Training Parameters. #
restore_flag = False

downsample = 16
base_rows  = 320
base_cols  = 320
disp_rows  = 320
disp_cols  = 320
max_rows   = 1.3 * base_rows
max_cols   = 1.3 * base_cols
row_length = int(max_rows / downsample)
col_length = int(max_cols / downsample)
step_cool  = 500
init_lr    = 0.001
min_lr     = 1.0e-5
decay_rate = 1.00
max_steps  = 5000
batch_size = 48
sub_batch  = 4
n_classes  = len(id_2_label)
display_step = 50

# Define the checkpoint callback function. #
mnist_path = "C:/Users/admin/Desktop/TF_Models/mnist_model/"
train_loss = mnist_path + "mnist_losses_vit.csv"
ckpt_model = mnist_path + "mnist_vit"

# Load the weights if continuing from a previous checkpoint. #
mnist_model = tf_obj_detector.ViT_Detector(
    n_classes, n_layers, num_heads, 
    row_length, col_length, hidden_sz, ffwd_size, 
    p_keep=0.9, ker_size=downsample, var_type="norm_add")
model_optim = tf.keras.optimizers.Adam()

checkpoint = tf.train.Checkpoint(
    step=tf.Variable(0), 
    mnist_model=mnist_model, 
    model_optim=model_optim)
ck_manager = tf.train.CheckpointManager(
    checkpoint, directory=ckpt_model, max_to_keep=1)

if restore_flag:
    train_loss_df = pd.read_csv(train_loss)
    training_loss = [tuple(
        train_loss_df.iloc[x].values) \
        for x in range(len(train_loss_df))]
    checkpoint.restore(ck_manager.latest_checkpoint)
    if ck_manager.latest_checkpoint:
        print("Model restored from {}".format(
            ck_manager.latest_checkpoint))
    else:
        print("Error: No latest checkpoint found.")
else:
    training_loss = []
st_step = checkpoint.step.numpy().astype(np.int32)

# Print out the model summary. #
#print(mnist_model.summary())
print("-" * 50)

print("Fit model on training data (" +\
      str(len(mnist_data)) + " training samples).")

train(mnist_model, n_classes, 
      sub_batch, batch_size, 
      mnist_data, training_loss, 
      st_step, max_steps, model_optim, checkpoint, 
      ck_manager, id_2_label, downsample=downsample, 
      base_rows=base_rows, base_cols=base_cols, 
      disp_rows=disp_rows, disp_cols=disp_cols, 
      display_step=display_step, step_cool=step_cool, 
      init_lr=init_lr, min_lr=min_lr, decay=decay_rate, 
      thresh=0.50, save_flag=True, train_loss_log=train_loss)
print("Model fitted.")
