from function import *
from char_detector import *
from char_classifier import *
from tensorflow.keras.models import load_model
import functools
import numpy as np

def get_msg(enc_chr):
    output_labels = [' '] + [chr(idx) for idx in range(65, 65+26)] + ['PAD']
    output_map = {c:idx for idx, c in enumerate(output_labels)}
    for key, value in output_map.items():
         if enc_chr == value:
             return key

class Decoder:
    def __init__(self):
        self.input_labels = [str(idx) for idx in range(10)] + [chr(idx) for idx in range(65, 65+26)] + ['PAD']
        self.input_map = {c:idx for idx, c in enumerate(self.input_labels)}
        self.output_labels = [' '] + [chr(idx) for idx in range(65, 65+26)] + ['PAD']
        self.output_map = {c:idx for idx, c in enumerate(self.output_labels)}
        self.char_detector = CharDetector()
        self.char_classifier = CharClassifier()
        self.decoder = None

    def load_models(self):
        self.char_detector.load_model()
        self.char_classifier.load_model()
        self.decoder = load_model('models/decrypter.h5')

    def read_encrypted_message(self, image):
        list_char = self.char_detector.predict(image)
        list_char = sorted(list_char, key=functools.cmp_to_key(compare_bbox))
        msg = ''
        for char_bbox in list_char:
          char_width = char_bbox[2] - char_bbox[0]
          char_height = char_bbox[3] - char_bbox[1]
          char_image = image.crop([
              char_bbox[0],
              char_bbox[1],
              char_bbox[2],
              char_bbox[3]                      
          ])
          x = char_image.height - char_image.width
          y = - char_image.height + char_image.width
          if char_image.width < char_image.height:
            char_image = ImageOps.expand(char_image, (x//2, x//4, x//2, x//4))
          else:
            char_image = ImageOps.expand(char_image, (y//4, y//2, y//4, y//2))
          # display(char_image)
          char_msg = self.char_classifier.predict(char_image)
          msg += char_msg
        return msg

    def decrypt_message(self, encrypted_message):
        """
        :param encrypted_message: a string of the encrypted message
        :return: a string of the decrypted message
        """
        enc_msg = [self.input_map[c] for c in encrypted_message]
        np_enc_msg = self.input_map['PAD'] * np.ones(16)
        np_enc_msg[:len(enc_msg)] = enc_msg
        np_enc_msg = np.array([np_enc_msg])
        my_predict = self.decoder.predict(np_enc_msg)
        decrypt_msg = ''
        for predict in my_predict[0]:
          if get_msg(np.argmax(predict)) != 'PAD':
            decrypt_msg += get_msg(np.argmax(predict))
        return decrypt_msg 

