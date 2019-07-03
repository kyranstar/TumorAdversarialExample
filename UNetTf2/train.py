import argparse
from parse_config import parse_config
from niftynet.layer.loss_segmentation import LossFunction
from tensorflow.contrib.layers.python.layers import regularizers
from model import MSNet

@tf.function
def step(model, x, y, optimizer):
    with tf.GradientTape() as tape:
      prediction = model(x)
      loss = loss_fn(prediction, y)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def train(config):
    # 1, load configuration parameters
    config_data  = config['data']
    config_net   = config['network']
    config_train = config['training']

    random.seed(config_train.get('random_seed', 1))
    assert(config_data['with_ground_truth'])

    net_name    = config_net['net_name']
    class_num   = config_net['class_num']
    batch_size  = config_data.get('batch_size', 5)

    # 2, construct graph
    full_data_shape  = [batch_size] + config_data['data_shape']
    full_label_shape = [batch_size] + config_data['label_shape']
    x = tf.placeholder(tf.float32, shape = full_data_shape)
    w = tf.placeholder(tf.float32, shape = full_label_shape)
    y = tf.placeholder(tf.int64,   shape = full_label_shape)

    w_regularizer = regularizers.l2_regularizer(config_train.get('decay', 1e-7))
    b_regularizer = regularizers.l2_regularizer(config_train.get('decay', 1e-7))
    net = MSNet(num_classes = class_num,
                    w_regularizer = w_regularizer,
                    b_regularizer = b_regularizer,
                    name = net_name)
    net.set_params(config_net)
    proby    = tf.nn.softmax(predicty)

    loss_func = LossFunction(n_class=class_num)
    print('size of predicty:', predicty)

    # 3, initialize session and saver
    lr = config_train.get('learning_rate', 1e-3)
    opt_step = tf.train.AdamOptimizer(lr).minimize(loss)
    saver = tf.train.Saver()

    dataloader = DataLoader(config_data)
    dataloader.load_data()

    # 4, start to train
    loss_file = config_train['model_save_prefix'] + "_loss.txt"
    start_it  = config_train.get('start_iteration', 0)
    if( start_it > 0):
        saver.restore(sess, config_train['model_pre_trained'])
    loss_list, temp_loss_list = [], []
    for n in range(start_it, config_train['maximal_iteration']):
        train_pair = dataloader.get_subimage_batch()
        tempx = train_pair['images']
        tempw = train_pair['weights']
        tempy = train_pair['labels']
        with tf.GradientTape() as tape:
            predicty = net(tempx, is_training = True)
            loss = loss_func(predicty, tempy, weight_map = tempw)
        gradients = tape.gradient(loss, net.trainable_variables)
        optimizer.apply_gradients(zip(gradients, net.trainable_variables))
            #opt_step.run(session = sess, feed_dict={x:tempx, w: tempw, y:tempy})

        if(n % config_train['test_iteration'] == 0):
            batch_dice_list = []
            for step in range(config_train['test_step']):
                train_pair = dataloader.get_subimage_batch()
                tempx = train_pair['images']
                tempw = train_pair['weights']
                tempy = train_pair['labels']
                dice = loss.eval(feed_dict ={x:tempx, w:tempw, y:tempy})
                batch_dice_list.append(dice)
            batch_dice = np.asarray(batch_dice_list, np.float32).mean()
            t = time.strftime('%X %x %Z')
            print(t, 'n', n,'loss', batch_dice)
            loss_list.append(batch_dice)
            np.savetxt(loss_file, np.asarray(loss_list))

        if((n + 1) % config_train['snapshot_iteration']  == 0):
            saver.save(sess, config_train['model_save_prefix']+"_{0:}.ckpt".format(n + 1))
    sess.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="training config file. example: python train.py config17/train_wt_ax.txt")
    args = parser.parse_args()

    config = parse_config(args.config_file)
    train(config)
