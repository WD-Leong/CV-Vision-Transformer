import tensorflow as tf
from tf_bias_layer import BiasLayer

def split_heads(full_inputs, num_heads):
    batch_size = tf.shape(full_inputs)[0]
    input_len  = tf.shape(full_inputs)[1]
    depth_size = tf.shape(full_inputs)[2] // num_heads
    
    split_outputs = tf.reshape(
        full_inputs, [batch_size, input_len, num_heads, depth_size])
    return tf.transpose(split_outputs, [0, 2, 1, 3])

def combine_heads(head_inputs):
    batch_size = tf.shape(head_inputs)[0]
    seq_length = tf.shape(head_inputs)[2]
    num_heads  = tf.shape(head_inputs)[1]
    depth_size = tf.shape(head_inputs)[3]
    hidden_size = num_heads*depth_size
    
    combined_outputs = tf.reshape(tf.transpose(
        head_inputs, [0, 2, 1, 3]), [batch_size, seq_length, hidden_size])
    return combined_outputs

def layer_normalisation(layer_input, bias, scale, eps=1.0e-6):
    eps = tf.constant(eps)
    layer_mean = tf.reduce_mean(
        layer_input, axis=[-1], keepdims=True)
    layer_var  = tf.reduce_mean(
        tf.square(layer_input-layer_mean), axis=[-1], keepdims=True)
    
    layer_std = (layer_input-layer_mean) * tf.math.rsqrt(layer_var + eps)
    return (layer_std*scale) + bias

class ViT_Detector(tf.keras.Model):
    def __init__(
        self, n_classes, n_layers, n_heads, 
        row_length, col_length, hidden_size, 
        ffwd_size, p_pi=0.99, p_keep=0.9, 
        ker_size=8, var_type="norm_add", **kwargs):
        super(ViT_Detector, self).__init__(**kwargs)
        bias_focal = tf.math.log((1.0-p_pi)/p_pi)
        
        self.p_keep  = p_keep
        self.n_heads = n_heads
        self.ker_size = ker_size
        self.var_type = var_type
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.ffwd_size = ffwd_size
        self.head_size = int(hidden_size / n_heads)
        
        self.row_length  = row_length
        self.col_length  = col_length
        self.hidden_size = hidden_size
        self.in_channels = ker_size*ker_size*3
        
        self.b_focal = BiasLayer(
            bias_init=bias_focal, 
            trainable=True, name="b_focal")
        
        # Projection layers. #
        input_shape  = [self.in_channels, self.hidden_size]
        output_shape = [self.hidden_size, self.n_classes+5]
        
        self.input_proj  = tf.Variable(tf.random.normal(
            input_shape, stddev=0.1), name="input_proj")
        self.output_proj = tf.Variable(tf.random.normal(
            output_shape, stddev=0.1), name="output_proj")
        
        # Transformer Encoder Variables. #
        pos_emb_shape  = [
            self.n_layers, self.row_length, 
            self.col_length, self.hidden_size]
        attn_wgt_shape = [
            self.n_layers, self.hidden_size, self.hidden_size]
        ffwd_wgt_shape = [
            self.n_layers, self.hidden_size, self.ffwd_size]
        out_wgt_shape  = [
            self.n_layers, self.ffwd_size, self.hidden_size]
        
        self.p_e_q = tf.Variable(
            tf.random.normal(attn_wgt_shape, stddev=0.1), name="p_e_q")
        self.p_e_k = tf.Variable(
            tf.random.normal(attn_wgt_shape, stddev=0.1), name="p_e_k")
        self.p_e_v = tf.Variable(
            tf.random.normal(attn_wgt_shape, stddev=0.1), name="p_e_v")
        self.p_e_c = tf.Variable(
            tf.random.normal(attn_wgt_shape, stddev=0.1), name="p_e_c")
        
        self.p_e_ff1 = tf.Variable(tf.random.normal(
            ffwd_wgt_shape, stddev=0.1), name="p_e_ff1")
        self.p_e_ff2 = tf.Variable(tf.random.normal(
            out_wgt_shape, stddev=0.1), name="p_e_ff2")
        self.b_e_ff1 = tf.Variable(tf.zeros(
            [self.n_layers, self.ffwd_size]), name="b_e_ff1")
        self.b_e_ff2 = tf.Variable(tf.zeros(
            [self.n_layers, self.hidden_size]), name="b_e_ff2")
        
        self.e_o_bias  = tf.Variable(tf.zeros(
            [self.hidden_size]), name="e_o_bias")
        self.e_o_scale = tf.Variable(tf.ones(
            [self.hidden_size]), name="e_o_scale")
        
        self.b_e_bias_1  = tf.Variable(tf.zeros(
            [self.n_layers, self.hidden_size]), name="b_e_bias_1")
        self.b_e_bias_2  = tf.Variable(tf.zeros(
            [self.n_layers, self.hidden_size]), name="b_e_bias_2")
        self.b_e_scale_1 = tf.Variable(tf.ones(
            [self.n_layers, self.hidden_size]), name="b_e_scale_1")
        self.b_e_scale_2 = tf.Variable(tf.ones(
            [self.n_layers, self.hidden_size]), name="b_e_scale_2")
        
        # Position Embeddings. #
        self.e_pos_emb = tf.Variable(tf.random.normal(
            pos_emb_shape, stddev=0.1), name="e_pos_embed")
    
    def transformer_encode(
        self, e_pos_embed, enc_inputs, p_reg=1.0, p_keep=1.0):
        n_heads = self.n_heads
        head_size = tf.cast(self.head_size, tf.float32)
        
        layer_input = enc_inputs
        for m in range(self.n_layers):
            layer_in = e_pos_embed[m] + layer_input
            
            # Self Attention Layer. #
            x_e_q = split_heads(tf.tensordot(
                layer_in, self.p_e_q[m], [[2], [0]]), n_heads)
            x_e_q = x_e_q * tf.math.rsqrt(head_size)
            
            x_e_k = split_heads(tf.tensordot(
                layer_in, self.p_e_k[m], [[2], [0]]), n_heads)
            x_e_v = split_heads(tf.tensordot(
                layer_in, self.p_e_v[m], [[2], [0]]), n_heads)
            
            x_scores = tf.matmul(x_e_q, x_e_k, transpose_b=True)
            x_alphas = tf.nn.softmax(x_scores)
            
            x_head_conc  = combine_heads(tf.matmul(x_alphas, x_e_v))
            x_multi_head = tf.nn.dropout(tf.tensordot(
                x_head_conc, self.p_e_c[m], [[2], [0]]), rate=1.0-p_reg)
            
            # Residual Connection Layer. #
            if self.var_type == "norm_add":
                x_heads_norm = tf.add(layer_in, layer_normalisation(
                    x_multi_head, self.b_e_bias_1[m], self.b_e_scale_1[m]))
            elif self.var_type == "add_norm":
                x_heads_norm = layer_normalisation(
                    tf.add(layer_in,  x_multi_head), 
                    self.b_e_bias_1[m], self.b_e_scale_1[m])
            
            # Feed forward layer. #
            x_ffw1 = tf.nn.relu(tf.add(tf.tensordot(
                x_heads_norm, self.p_e_ff1[m], [[2], [0]]), self.b_e_ff1[m]))
            x_ffw2 = tf.add(tf.tensordot(
                x_ffw1, self.p_e_ff2[m], [[2], [0]]), self.b_e_ff2[m])
            x_ffwd = tf.nn.dropout(x_ffw2, rate=1.0-p_keep)
            
            # Residual Connection Layer. #
            if self.var_type == "norm_add":
                x_ffw_norm = tf.add(
                    x_heads_norm,  layer_normalisation(
                        x_ffwd, self.b_e_bias_2[m], self.b_e_scale_2[m]))
            elif self.var_type == "add_norm":
                x_ffw_norm = layer_normalisation(
                    tf.add(x_heads_norm,  x_ffwd), 
                    self.b_e_bias_2[m], self.b_e_scale_2[m])
            
            # Append the output of the current layer. #
            layer_input = x_ffw_norm
        
        # Residual Connection. #
        if self.var_type == "norm_add":
            enc_outputs = tf.add(
                enc_inputs, layer_normalisation(
                    x_ffw_norm, self.e_o_bias, self.e_o_scale))
        elif self.var_type == "add_norm":
            enc_outputs = layer_normalisation(tf.add(
                enc_inputs, x_ffw_norm), self.e_o_bias, self.e_o_scale)
        return enc_outputs
    
    def call(self, x_image, p_reg=1.0, training=False):
        input_dims = tf.shape(x_image)
        batch_size = input_dims[0]
        enc_shape  = [batch_size, -1, self.hidden_size]
        
        # Image Patches. #
        ker_sizes = [1, self.ker_size, self.ker_size, 1]
        x_strides = [1, self.ker_size, self.ker_size, 1]
        x_patches = tf.image.extract_patches(
            x_image, sizes=ker_sizes, 
            strides=x_strides, rates=[1, 1, 1, 1], padding="VALID")
        
        # Image Embedding. #
        x_enc_raw = tf.tensordot(
            x_patches, self.input_proj, [[3], [0]])
        raw_enc_shape = tf.shape(x_enc_raw)
        
        # Initialise decoder inputs to zeros. #
        x_encode = tf.reshape(x_enc_raw, enc_shape)
        
        # Encoder Model. #
        pos_embed_reshape = [
            self.n_layers, -1, self.hidden_size]
        
        e_pos_embed = self.e_pos_emb[
            :, :raw_enc_shape[1], :raw_enc_shape[2], :]
        e_pos_embed = tf.reshape(
            e_pos_embed, pos_embed_reshape)
        enc_outputs = self.transformer_encode(
            e_pos_embed, x_encode, p_reg=p_reg)
        
        # Output Projection. #
        enc_project = tf.tensordot(
            enc_outputs, self.output_proj, [[2], [0]])
        
        # Set the outputs accordingly. #
        reg_output = tf.nn.sigmoid(enc_project[:, :, :4])
        cls_output = self.b_focal(enc_project[:, :, 4:])
        
        encoder_output = tf.concat(
            [reg_output, cls_output], axis=2)
        return encoder_output

