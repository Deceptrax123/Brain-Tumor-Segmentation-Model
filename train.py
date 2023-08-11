import tensorflow as tf
from glob import glob
from sklearn.model_selection import train_test_split
from batch_generator import train_batch_generator, test_batch_generator
from tried_models.model import make_model
from scheduler import scheduler
import datetime
from losses import Complete_Dice_Loss, CrossEntropyDiceLoss
from metrics import Complete_Dice_Coef, Enhancing_Dice_Coef, Necrotic_Dice_Coef, Edema_Dice_Coef
from models import UnetLSTM


def steps(m, batchsize):
    return (m+batchsize-1)//batchsize


@tf.function
def train(images, masks):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = CrossEntropyDiceLoss.call(masks, predictions)
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
    test_loss = CrossEntropyDiceLoss().call(masks, predictions)

    test_complete.update_state(masks, predictions)
    test_edema.update_state(masks, predictions)
    test_enhancing.update_state(masks, predictions)
    test_necrotic.update_state(masks, predictions)

    return test_loss


def training_loop(traingen, testgen, callbacks, train_steps, test_steps):
    num_epochs = 50

    train_losses = []
    test_losses = []
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
        # get learning rate as per scheduler function
        lr = scheduler(epoch+1, optimizer.learning_rate)
        optimizer.learning_rate = lr

        traingen = train_batch_generator(train_sample_paths, batch_size)
        testgen = test_batch_generator(test_sample_paths, batch_size)

        print(
            "--------------------------Start of Epoch %d---------------------------------" % (epoch+1))
        losses = []
        tlosses = []
        for step in range(train_steps):

            sample, masks = next(traingen)

            train_loss = train(sample, masks)
            losses.append(train_loss)

            if (step % 50 == 0):
                print("Loss so far per 50 batches - %.4f" %
                      (float(train_loss)))

        m.update_state(losses)
        print("Training Loss after epoch % d - %.4f" %
              (epoch+1, float(m.result())))

        train_losses.append(m.result())
        m.reset_state()

        for step in range(test_steps):
            sample, masks = next(testgen)

            test_loss = test(sample, masks)
            tlosses.append(test_loss)

        test_mean.update_state(tlosses)
        test_losses.append(test_mean.result())

        print("Test Loss after epoch %d - %.4f" %
              (epoch+1, float(test_mean.result())))

        test_mean.reset_state()

        # metrics after training
        print("Training Metrics after epoch %d- Dice coef complete- %.4f,Enhancing- %0.4f, Necrotic- %0.4f, Edema- %0.4f" %
              (epoch+1, train_complete.result()/train_steps, train_enhancing.result()/train_steps, train_necrotic.result()/train_steps, train_edema.result()/train_steps))

        # metrics after testing
        print("Test metrics after epoch %d- Dice Coef complete- %0.4f, Enhancing- %0.4f, Necrotic- %0.4f, Edema- %0.4f" %
              (epoch+1, test_complete.result()/test_steps, test_enhancing.result()/test_steps, test_necrotic.result()/test_steps, test_edema.result()/test_steps))

        # Append to list to visualize later
        tcomplete.append(train_complete.result()/train_steps)
        tenhancing.append(train_enhancing.result()/train_steps)
        tedema.append(train_edema.result()/train_steps)
        tnecrotic.append(train_necrotic.result()/train_steps)
        tecomplete.append(test_complete.result()/test_steps)
        teedema.append(test_edema.result()/test_steps)
        teenhancing.append(test_enhancing.result()/test_steps)
        tenecrotic.append(test_necrotic.result()/test_steps)

        # reset states after each epoch
        train_complete.reset_state()
        train_enhancing.reset_state()
        train_edema.reset_state()
        train_necrotic.reset_state()

        test_complete.reset_state()
        test_enhancing.reset_state()
        test_edema.reset_state()
        test_necrotic.reset_state()

        print("-------------------------End of Epoch %d --------------------------------" % (epoch+1))

    return train_losses, tcomplete, tedema, tenhancing, tecomplete, teedema, teenhancing, tenecrotic


batch_size = 2
samples_paths = sorted(glob("./Data/Train/samples_processed/*.npy"))
masks_paths = sorted(glob("./Data/Train/masks_reformatted/*.npy"))

train_sample_paths, test_sample_paths = train_test_split(
    samples_paths, test_size=0.25)

train_steps = steps(len(train_sample_paths), batch_size)
test_steps = steps(len(test_sample_paths), batch_size)

# model = make_model()
model = UnetLSTM()

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

m = tf.keras.metrics.Mean()
test_mean = tf.keras.metrics.Mean()

model.summary()
