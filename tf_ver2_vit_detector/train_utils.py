import numpy as np
import tensorflow as tf

from PIL import Image
import matplotlib.pyplot as plt

def _parse_image(
    filename, img_rows=448, img_cols=448):
    image_string  = tf.io.read_file(filename)
    image_decoded = \
        tf.image.decode_jpeg(image_string, channels=3)
    image_decoded = \
        tf.image.convert_image_dtype(image_decoded, tf.float32)
    image_resized = tf.image.resize(
        image_decoded, [img_rows, img_cols])
    image_resized = tf.ensure_shape(
        image_resized ,shape=(img_rows, img_cols, 3))
    return image_resized

def bbox_flip90(img_bbox):
    img_bbox = tf.transpose(img_bbox, [1, 0, 2, 3])
    tmp_bbox = img_bbox.numpy()
    img_bbox = tmp_bbox
    
    img_bbox[:, :, :, 0] = tmp_bbox[:, :, :, 1]
    img_bbox[:, :, :, 1] = tmp_bbox[:, :, :, 0]
    img_bbox[:, :, :, 2] = tmp_bbox[:, :, :, 3]
    img_bbox[:, :, :, 3] = tmp_bbox[:, :, :, 2]
    img_bbox = tf.constant(img_bbox)
    return img_bbox

def sigmoid_loss(labels, logits):
    return tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.cast(labels, tf.float32), logits=logits)

def focal_loss(
    labels, logits, alpha=0.25, gamma=2.0):
    labels = tf.cast(labels, tf.float32)
    tmp_log_logits  = tf.math.log(1.0 + tf.exp(-1.0 * tf.abs(logits)))
    
    tmp_abs_term = tf.math.add(
        tf.multiply(labels * alpha * tmp_log_logits, 
                    tf.pow(1.0 - tf.nn.sigmoid(logits), gamma)), 
        tf.multiply(tf.pow(tf.nn.sigmoid(logits), gamma), 
                    (1.0 - labels) * (1.0 - alpha) * tmp_log_logits))
    
    tmp_x_neg = tf.multiply(
        labels * alpha * tf.minimum(logits, 0), 
        tf.pow(1.0 - tf.nn.sigmoid(logits), gamma))
    tmp_x_pos = tf.multiply(
        (1.0 - labels) * (1.0 - alpha), 
        tf.maximum(logits, 0) * tf.pow(tf.nn.sigmoid(logits), gamma))
    
    foc_loss_stable = tmp_abs_term + tmp_x_pos - tmp_x_neg
    return tf.reduce_sum(foc_loss_stable, axis=[1, 2, 3])

def model_loss(
    bboxes, masks, outputs, img_size=448, 
    reg_lambda=0.10, loss_type="sigmoid", eps=1.0e-6):
    reg_weight = tf.expand_dims(masks, axis=3)
    reg_output = outputs[:, :, :, :4]
    cls_output = outputs[:, :, :, 4:]
    cls_labels = tf.cast(bboxes[:, :, :, 4:], tf.int32)
    
    if loss_type == "sigmoid":
        total_cls_loss  = tf.reduce_sum(
            sigmoid_loss(cls_labels, cls_output))
    else:
        total_cls_loss  = tf.reduce_sum(
            focal_loss(cls_labels, cls_output))
    total_reg_loss  = tf.reduce_sum(tf.multiply(
        tf.abs(bboxes[:, :, :, :4] - reg_output), reg_weight))
    return total_cls_loss, total_reg_loss

def train_step(
    model, sub_batch_sz, images, bboxes, masks, 
    optimizer, loss_type="focal", downsample=32, 
    cls_lambda=5.0, learning_rate=1.0e-3, grad_clip=1.0):
    optimizer.lr.assign(learning_rate)
    
    batch_size  = images.shape[0]
    input_dims  = images.shape[1:]
    
    if batch_size <= sub_batch_sz:
        n_sub_batch = 1
    elif batch_size % sub_batch_sz == 0:
        n_sub_batch = int(batch_size / sub_batch_sz)
    else:
        n_sub_batch = int(batch_size / sub_batch_sz) + 1
    
    model_params  = model.trainable_variables
    acc_gradients = [tf.zeros_like(var) for var in model_params]
    
    tmp_reg_loss = 0.0
    tmp_cls_loss = 0.0
    for n_sub in range(n_sub_batch):
        id_st = n_sub*sub_batch_sz
        if n_sub != (n_sub_batch-1):
            id_en = (n_sub+1)*sub_batch_sz
        else:
            id_en = batch_size
        
        tmp_images = images[id_st:id_en, :, :, :]
        tmp_bboxes = bboxes[id_st:id_en, :, :, :]
        tmp_masks  = masks[id_st:id_en, :, :]
        
        output_dims = [
            (id_en-id_st), 
            int(input_dims[0]/downsample), 
            int(input_dims[1]/downsample), -1]
        with tf.GradientTape() as grad_tape:
            tmp_output = tf.reshape(
                model(tmp_images, training=True), output_dims)
            
            tmp_losses = model_loss(
                tmp_bboxes, tmp_masks, 
                tmp_output, loss_type=loss_type)
            
            tmp_cls_loss += tmp_losses[0]
            tmp_reg_loss += tmp_losses[1]
            total_losses = \
                cls_lambda*tmp_losses[0] + tmp_losses[1]
        
        # Accumulate the gradients. #
        tmp_gradients = \
            grad_tape.gradient(total_losses, model_params)
        acc_gradients = [
            (acc_grad+grad) for \
            acc_grad, grad in zip(acc_gradients, tmp_gradients)]
    
    # Update using the optimizer. #
    avg_reg_loss  = tmp_reg_loss / batch_size
    avg_cls_loss  = tmp_cls_loss / batch_size
    acc_gradients = [tf.math.divide_no_nan(
        acc_grad, batch_size) for acc_grad in acc_gradients]
    
    clipped_gradients, _ = \
        tf.clip_by_global_norm(acc_gradients, grad_clip)
    optimizer.apply_gradients(
        zip(clipped_gradients, model_params))
    return avg_cls_loss, avg_reg_loss

def obj_detect_results(
    img_in_file, model, labels, heatmap=True, 
    downsample=32, thresh=0.50, img_box=None, 
    img_title=None, img_rows=448, img_cols=448, 
    save_img_file="object_detection_result.jpg"):
    img_scale = min(img_rows, img_cols)
    
    # Read the image. #
    image_resized = tf.expand_dims(_parse_image(
        img_in_file, img_rows=img_rows, img_cols=img_cols), axis=0)
    
    output_dims = [
        1, int(img_rows/downsample), 
        int(img_cols/downsample), -1]
    
    tmp_output = model.predict(image_resized)
    tmp_output = tf.reshape(
        tmp_output, output_dims).numpy()
    reg_output = tmp_output[0, :, :, :4]
    cls_output = tmp_output[0, :, :, 4:]
    cls_probs  = tf.nn.sigmoid(cls_output)
    n_classes  = cls_output.shape[2]
    
    # Plot the bounding boxes on the image. #
    fig, ax = plt.subplots(1)
    tmp_img = np.array(
        Image.open(img_in_file), dtype=np.uint8)
    ax.imshow(tmp_img)
    
    img_width   = tmp_img.shape[0]
    img_height  = tmp_img.shape[1]
    tmp_w_ratio = img_width / img_rows
    tmp_h_ratio = img_height / img_cols
    
    if heatmap:
        if n_classes > 1:
            obj_probs = tf.reduce_max(
                cls_probs[:, :, 1:], axis=2)
        else:
            obj_probs = cls_probs[:, :, 0]
        
        obj_probs = tf.image.resize(tf.expand_dims(
            obj_probs, axis=2), [img_width, img_height])
        tmp = ax.imshow(tf.squeeze(
            obj_probs, axis=2), "jet", alpha=0.50)
        fig.colorbar(tmp, ax=ax)
    
    n_obj_detected = 0
    if n_classes > 1:
        prob_max = tf.reduce_max(
            cls_probs[:, :, 1:], axis=2)
        pred_label = tf.math.argmax(
            cls_probs[:, :, 1:], axis=2)
    else:
        prob_max = cls_probs[:, :, 0]
    tmp_thresh = \
        np.where(prob_max >= thresh, 1, 0)
    tmp_coords = np.nonzero(tmp_thresh)
    
    for n_box in range(len(tmp_coords[0])):
        x_coord = tmp_coords[0][n_box]
        y_coord = tmp_coords[1][n_box]
        
        tmp_boxes = reg_output[x_coord, y_coord, :]
        tmp_probs = int(
            prob_max[x_coord, y_coord].numpy()*100)
        if n_classes > 1:
            tmp_label = str(labels[
                pred_label[x_coord, y_coord].numpy()])
        else:
            tmp_label = str(labels[0])
        
        x_centroid = \
            tmp_w_ratio * (x_coord + tmp_boxes[0])*downsample
        y_centroid = \
            tmp_h_ratio * (y_coord + tmp_boxes[1])*downsample
        box_width  = tmp_w_ratio * img_scale * tmp_boxes[2]
        box_height = tmp_h_ratio * img_scale * tmp_boxes[3]
        
        if box_width > img_width:
            box_width = img_width
        if box_height > img_height:
            box_height = img_height
        
        # Output prediction is transposed. #
        x_lower = x_centroid - box_width/2
        y_lower = y_centroid - box_height/2
        if x_lower < 0:
            x_lower = 0
        if y_lower < 0:
            y_lower = 0
        
        box_patch = plt.Rectangle(
            (y_lower, x_lower), box_height, box_width, 
            linewidth=1, edgecolor="red", fill=None)
        
        n_obj_detected += 1
        tmp_text = \
            tmp_label + ": " + str(tmp_probs) + "%"
        ax.add_patch(box_patch)
        ax.text(y_lower, x_lower, tmp_text, 
                fontsize=10, color="red")
    print(str(n_obj_detected), "objects detected.")
    
    # True image is not transposed. #
    if img_box is not None:
        tmp_true_box = np.nonzero(img_box[:, :, 4])
        for n_box in range(len(tmp_true_box[0])):
            x_coord = tmp_true_box[0][n_box]
            y_coord = tmp_true_box[1][n_box]
            tmp_boxes = img_box[x_coord, y_coord, :4]
            
            x_centroid = \
                tmp_w_ratio * (x_coord + tmp_boxes[0])*downsample
            y_centroid = \
                tmp_h_ratio * (y_coord + tmp_boxes[1])*downsample
            box_width  = tmp_w_ratio * img_scale * tmp_boxes[2]
            box_height = tmp_h_ratio * img_scale * tmp_boxes[3]
            
            x_lower = x_centroid - box_width/2
            y_lower = y_centroid - box_height/2
            box_patch = plt.Rectangle(
                (y_lower.numpy(), x_lower.numpy()), 
                box_height.numpy(), box_width.numpy(), 
                linewidth=1, edgecolor="black", fill=None)
            ax.add_patch(box_patch)
    
    if img_title is not None:
        fig.suptitle(img_title)
    fig.savefig(save_img_file, dpi=199)
    plt.close()
    del fig, ax
    return None

def show_object_boxes(
    img_array, img_box, 
    img_dims, downsample=32, 
    save_img_file="ground_truth.jpg"):
    n_classes = int(img_box.shape[2]) - 4
    
    # Plot the bounding boxes on the image. #
    fig, ax = plt.subplots(1)
    #tmp_img = np.array(
    #    Image.open(img_in_file), dtype=np.uint8)
    ax.imshow(img_array)
    
#    img_width   = tmp_img.shape[0]
#    img_height  = tmp_img.shape[1]
#    tmp_w_ratio = img_width / img_dims
#    tmp_h_ratio = img_height / img_dims
    tmp_w_ratio = 1.0
    tmp_h_ratio = 1.0
    
    if n_classes > 1:
        obj_probs = tf.reduce_max(
            img_box[:, :, 5:], axis=2)
    else:
        obj_probs = img_box[:, :, 4]
    
    obj_probs = tf.image.resize(tf.expand_dims(
        obj_probs, axis=2), [img_dims, img_dims])
    tmp = ax.imshow(tf.squeeze(
        obj_probs, axis=2), "jet", alpha=0.50)
    fig.colorbar(tmp, ax=ax)
    
    # True image is not transposed. #
    if img_box is not None:
        tmp_true_box = np.nonzero(img_box[:, :, 4])
        for n_box in range(len(tmp_true_box[0])):
            x_coord = tmp_true_box[0][n_box]
            y_coord = tmp_true_box[1][n_box]
            tmp_boxes = img_box[x_coord, y_coord, :4]
            
            x_centroid = \
                tmp_w_ratio * (x_coord + tmp_boxes[0])*downsample
            y_centroid = \
                tmp_h_ratio * (y_coord + tmp_boxes[1])*downsample
            box_width  = tmp_w_ratio * img_dims * tmp_boxes[2]
            box_height = tmp_h_ratio * img_dims * tmp_boxes[3]
            
            x_lower = x_centroid - box_width/2
            y_lower = y_centroid - box_height/2
            box_patch = plt.Rectangle(
                (y_lower.numpy(), x_lower.numpy()), 
                box_height.numpy(), box_width.numpy(), 
                linewidth=1, edgecolor="black", fill=None)
            ax.add_patch(box_patch)
    
    fig.suptitle("Ground Truth")
    fig.savefig(save_img_file, dpi=199)
    plt.close()
    del fig, ax
    return None