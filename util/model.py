# import wget
import tensorflow as tf

def movenet(input_image):
    
    interpreter = tf.lite.Interpreter(model_path="weights/lite-model_movenet_singlepose_thunder_tflite_float16_4.tflite")
    interpreter.allocate_tensors()
    # TF Lite format expects tensor type of uint8.
    input_image = tf.cast(input_image, dtype=tf.uint8)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
    # Invoke inference.
    interpreter.invoke() 
    # Get the model prediction.
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
    return keypoints_with_scores
