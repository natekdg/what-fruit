import tensorflow as tf

def load_model():
    model = tf.saved_model.load("yolov4")
    return model

def get_predictions(model, image):
    image_array = tf.convert_to_tensor(image)
    image_array = tf.expand_dims(image_array, 0)
    predictions = model(image_array)
    return predictions
