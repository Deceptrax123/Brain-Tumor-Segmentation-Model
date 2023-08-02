import tensorflow as tf
from glob import glob
from sklearn.model_selection import train_test_split
from batch_generator import batch_generator
from model import make_model
from scheduler import scheduler
import datetime
from losses import Complete_Dice_Loss
from metrics import Complete_Dice_Coef, Enhancing_Dice_Coef, Necrotic_Dice_Coef, Edema_Dice_Coef


@tf.function
def steps(m, batchsize):
    return (m+batchsize-1)//batchsize


@tf.function
def train(images, masks):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = Complete_Dice_loss(masks, predictions)
    gradients = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))

    train_complete.update_state(masks, predictions)
    train_edema.update_state(masks, predictions)
    train_enhancing.update_state(masks, predictions)
    train_necrotic.update_state(masks, predictions)

    return loss


@tf.function
def test(images, masks):
    predictions = model(images, training=False)

    test_complete.update_state(masks, predictions)
    test_edema.update_state(masks, predictions)
    test_enhancing.update_state(masks, predictions)
    test_necrotic.update_state(masks, predictions)


@tf.function
def training_loop(traingen, testgen, callbacks, train_steps, test_steps):
    num_epochs = 50
    train_loss = 0

    losses = []
    train_losses = []
    # train metrics
    tcomplete = []
    tedema = []
    tnecrotic = []
    tenhancing = []

    # test metrics
    tecomplete = []
    teedema = []
    tenecrotic = []
    teenhancing = []

    for epoch in range(num_epochs):
        for step in range(train_steps):
            sample, masks = next(traingen)

            train_loss = train(sample, masks)
            losses.append(train_loss)

            if (step % 10 == 0):
                print("Loss so far per batch %.4f" % (float(train_loss)))

        train_loss = tf.keras.metrics.Mean(losses)
        print("Training Loss after epoch %d - %.4f" % (epoch, train_loss))
        train_losses.append(train_loss)

        for step in test_steps:
            sample, masks = next(testgen)

            test(sample, masks)

        # metrics after training
        print("Training Metrics after epoch %d- Dice coef complete- %.4f,Enhancing- %0.4f, Necrotic- %0.4f, Edema- %0.4f" %
              (epoch, train_complete.result(), train_enhancing.result(), train_necrotic.result(), train_edema.result()))

        # metrics after testing
        print("Test metrics after epoch %d- Dice Coef complete- %0.4f, Enhancing- %0.4f, Necrotic- %0.4f, Edema- %0.4f" %
              (epoch, test_complete.result(), test_enhancing.result(), test_necrotic.result(), test_edema.result()))

        # Append to list to visualize later
        tcomplete.append(train_complete.result())
        tenhancing.append(train_enhancing.result())
        tedema.append(train_edema.result())
        tnecrotic.append(train_necrotic.result())
        tecomplete.append(test_complete.result())
        teedema.append(test_edema.result())
        teenhancing.append(test_enhancing.result())
        tenecrotic.append(test_necrotic.result())

        # reset states after each epoch
        train_complete.reset_states()
        train_enhancing.reset_state()
        train_edema.reset_state()
        train_necrotic.reset_state()

        test_complete.reset_states()
        test_enhancing.reset_state()
        test_edema.reset_state()
        test_necrotic.reset_state()

    return train_losses, tcomplete, tedema, tenhancing, tecomplete, teedema, teenhancing, tenecrotic


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

# Call metric classes
train_complete = Complete_Dice_Coef()
test_complete = Complete_Dice_Coef()

train_necrotic = Necrotic_Dice_Coef()
test_necrotic = Necrotic_Dice_Coef()

train_enhancing = Enhancing_Dice_Coef()
test_enhancing = Enhancing_Dice_Coef()

train_edema = Edema_Dice_Coef()
test_edema = Edema_Dice_Coef()

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
