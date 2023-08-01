import tensorflow as tf
from glob import glob
from sklearn.model_selection import train_test_split
from batch_generator import batch_generator
from model import make_model
from scheduler import scheduler
import datetime
import keras
from metrics import dice_cosh_loss, dice_coef_complete, dice_coef_enhancing, dice_coef_necrotic, dice_coef


def steps(m, batchsize):
    return (m+batchsize-1)//batchsize


def train(images, masks):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = dice_cosh_loss(masks, predictions)
    gradients = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))

    return loss


def test(images, masks):
    predictions = model(images, training=False)


def training_loop(traingen, testgen, callbacks, train_steps, test_steps):
    num_epochs = 50
    train_loss = 0

    losses = []
    for epoch in range(num_epochs):
        for step in range(train_steps):
            sample, masks = next(traingen)

            train_loss = train(sample, masks)
            losses.append(train_loss)

            if (step % 10 == 0):
                print("Loss so far per batch %.4f" % (float(train_loss)))

        train_loss = tf.keras.metrics.Mean(losses)
        print("Training Loss after epoch %d - %.4f" % (epoch, train_loss))

        for step in test_steps:
            sample, masks = next(testgen)

            test(sample, masks)


batch_size = 2
samples_paths = sorted(glob("./Data/Train/samples_processed/*.npy"))
masks_paths = sorted(glob("./Data/Train/masks_reformatted/*.npy"))

train_sample_paths, test_sample_paths = train_test_split(
    samples_paths, test_size=0.25)

traingen = batch_generator(train_sample_paths, batch_size)
testgen = batch_generator(test_sample_paths, batch_size)

train_steps = steps(len(train_sample_paths), batch_size)
test_steps = steps(len(test_sample_paths), batch_size)

model = make_model()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)


log_dir = "/kaggle/working/logs/fit/" + \
    datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir, histogram_freq=1)
schedule = tf.keras.callbacks.LearningRateScheduler(scheduler)
plateau_handler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', patience=3, factor=0.1)
nanterminate = tf.keras.callbacks.TerminateOnNaN()
earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss')
checkpoints = tf.keras.callbacks.ModelCheckpoint(filepath="/kaggle/working/models/2pathcnn_bilstm",
                                                 save_weights_only=False, save_freq='epoch', save_best_only=True, monitor='dice_coef', mode='max', verbose=1)

callbacks = [tensorboard_callback, schedule,
             plateau_handler, nanterminate, earlystop, checkpoints]
