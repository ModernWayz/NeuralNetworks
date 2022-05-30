import os
import tensorflow as tf
import numpy as np
import IPython.display as display
import PIL.Image

IMAGE_URL = 'https://images.unsplash.com/photo-1456081101716-74e616ab23d8?ixlib=rb-1.2.1&raw_url=true&q=80&fm=jpg&crop=entropy&cs=tinysrgb&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=2676'
IMAGES_DIR = 'Labo_Week_11/images/'
OUTPUT_DIR = 'Labo_Week_11/'
IMAGES = 5
OCTAVE_SCALE = 0.90

def download(url, max_dim=None):
    name = url.split('/')[-1]
    image_path = tf.keras.utils.get_file(name, origin=url)
    img = PIL.Image.open(image_path)
    if max_dim:
        img.thumbnail((max_dim, max_dim))
    return np.array(img)

def deprocess(img):
    img = 255*(img + 1.0)/2.0
    return tf.cast(img, tf.uint8)

def show(img):
    display.display(PIL.Image.fromarray(np.array(img)))

original_img = download(IMAGE_URL, max_dim=500)
base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')

names = ['mixed3', 'mixed5']
layers = [base_model.get_layer(name).output for name in names]

dream_model = tf.keras.Model(inputs=base_model.input, outputs=layers)

def calc_loss(img, model):
  img_batch = tf.expand_dims(img, axis=0)
  layer_activations = model(img_batch)
  if len(layer_activations) == 1:
    layer_activations = [layer_activations]

  losses = []
  for act in layer_activations:
    loss = tf.math.reduce_mean(act)
    losses.append(loss)

  return  tf.reduce_sum(losses)

class DeepDream(tf.Module):
    def __init__(self, model):
        self.model = model
    @tf.function(
        input_signature=(
        tf.TensorSpec(shape=[None,None,3], dtype=tf.float32),
        tf.TensorSpec(shape=[], dtype=tf.int32),
        tf.TensorSpec(shape=[], dtype=tf.float32),)
    )
    def __call__(self, img, steps, step_size):
        print("Tracing")
        loss = tf.constant(0.0)
        for n in tf.range(steps):
            with tf.GradientTape() as tape:
                tape.watch(img)
                loss = calc_loss(img, self.model)
            gradients = tape.gradient(loss, img)
            gradients /= tf.math.reduce_std(gradients) + 1e-8 
            img = img + gradients*step_size
            img = tf.clip_by_value(img, -1, 1)
        return loss, img

deepdream = DeepDream(dream_model)

def run_deep_dream_simple(img, steps=100, step_size=0.01):
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    img = tf.convert_to_tensor(img)
    step_size = tf.convert_to_tensor(step_size)
    steps_remaining = steps
    step = 0
    while steps_remaining:
        if steps_remaining>100:
            run_steps = tf.constant(100)
        else:
            run_steps = tf.constant(steps_remaining)
            steps_remaining -= run_steps
            step += run_steps
            loss, img = deepdream(img, run_steps, tf.constant(step_size))
            display.clear_output(wait=True)
            show(deprocess(img))
            print ("Step {}, loss {}".format(step, loss))
    result = deprocess(img)
    display.clear_output(wait = True)
    return result

for i in range(IMAGES):
    OCTAVE_SCALE += 0.10
    img = tf.constant(np.array(original_img))
    base_shape = tf.shape(img)[:-1]
    float_base_shape = tf.cast(base_shape, tf.float32)

    for n in range(-2, 3):
        new_shape = tf.cast(float_base_shape*(OCTAVE_SCALE**n), tf.int32)
        img = tf.image.resize(img, new_shape).numpy()
        img = run_deep_dream_simple(img=img, steps=100, step_size=0.01)

    display.clear_output(wait=True)
    img = tf.image.resize(img, base_shape)
    img = tf.image.convert_image_dtype(img/255.0, dtype=tf.uint8)
    img_result = np.clip(img, 0.0, 255.0)
    img_result = img_result.astype(np.uint8)
    PIL.Image.fromarray(img_result, mode='RGB').save(IMAGES_DIR + str(i) + '.jpg')

images = []
for filename in os.listdir(IMAGES_DIR):
    if filename == '.DS_Store':
        continue
    image = PIL.Image.open(IMAGES_DIR + filename)
    images.append(image)

for i in reversed(range(len(os.listdir(IMAGES_DIR)) -2)):
    if(i == 0):
        break
    image = PIL.Image.open(IMAGES_DIR + str(i) + '.jpg')
    images.append(image)

image.save(OUTPUT_DIR + 'output.gif', save_all = True, append_images = images, optimize = True, loop = 0, duration = 10)