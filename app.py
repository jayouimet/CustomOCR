from PIL import Image, ImageDraw, ImageFont
import numpy as np
import random
import tensorflow as tf
from src.models.crnn_64_512 import Crnn_64_512

alphabet_64_256 = " ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?-"
alphabet_64_512 = " ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?-'"

alphabet = alphabet_64_512
num_to_char = {i: c for i, c in enumerate(alphabet, start=1)}

keys   = tf.constant(list(alphabet), dtype=tf.string)
vals   = tf.constant(list(range(1, len(alphabet)+1)), dtype=tf.int32)
initializer = tf.lookup.KeyValueTensorInitializer(
    keys, vals,
    key_dtype=tf.string,
    value_dtype=tf.int32
)
char_to_num_table = tf.lookup.StaticHashTable(initializer, default_value=0)

def make_synthetic_image(text, width=512, height=64):
    img = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(img)

    font = ImageFont.truetype("arial.ttf", size=random.randint(24, 32))

    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    w, h = right - left, bottom - top

    x = (width - w) // 2
    y = (height - h) // 2
    draw.text((x, y), text, fill=0, font=font)

    return np.array(img) / 255.0, text

def gen():
    while True:
        text = "".join(random.choices(alphabet, k=random.randint(15, 30)))
        img, lbl = make_synthetic_image(text)
        yield img.astype(np.float32), lbl

dataset = tf.data.Dataset.from_generator(
    gen,
    output_signature=(
        tf.TensorSpec(shape=(64,512), dtype=tf.float32),
        tf.TensorSpec(shape=(),       dtype=tf.string),
    )
)

def encode_sample(img, label):
    img = img[..., tf.newaxis]
    chars = tf.strings.unicode_split(label, 'UTF-8')
    label_seq = char_to_num_table.lookup(chars)
    return img, label_seq

dataset = (
    dataset
    .map(encode_sample, num_parallel_calls=tf.data.AUTOTUNE)
    .padded_batch(
        batch_size=32,
        padded_shapes=([64,512,1], [None]),
        padding_values=(0.0, 0)
    )
    .prefetch(tf.data.AUTOTUNE)
)

def decode_prediction(logits):
    pred = tf.argmax(logits, axis=-1).numpy()
    chars = []
    prev = 0
    for p in pred:
        if p != prev and p != 0:
            chars.append(num_to_char[p])
        prev = p
    return "".join(chars)

model = Crnn_64_512(alphabet, True)

model.train(dataset)

img_np, txt = make_synthetic_image("Hey, what's new today with you?")

im = Image.fromarray((img_np * 255).astype(np.uint8))
im.show()

logits = model.predict(img_np[None,...,None])
print("Decoded:", decode_prediction(logits[0]))