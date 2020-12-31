import struct
import numpy as np
from PIL import Image

data_format_ETL8G = {
    'record_size': 8199,
    'resolution': [128, 127],
    'record_format': '>2H8sI4B4H2B30x8128s11x',
    'char_code_index': 1,
    'image_data_index': 14,
    'bit_depth': 4,
    'char_set': 'JIS_X_0208',
}

def read_record(f, data_format):
    s = f.read(data_format['record_size'])
    r = struct.unpack(data_format['record_format'], s)

    img_data_idx = data_format['image_data_index']
    reso      = data_format['resolution']
    bit_depth = data_format['bit_depth']

    img = Image.frombytes('F', reso, r[img_data_idx], 'bit', bit_depth)
    imgL = img.convert('L')
    return r + (imgL,)
  
def read_char():
    # Type of characters = 71 + 4, person = 160, y = 127, x = 128
    hiragana_array = np.zeros([71, 160, 127, 128], dtype=np.uint8)
    #foo = open('hiragana_list.txt', 'wb+')
    for j in range(1, 33):
        filename = './ETL8G/ETL8G_{:02d}'.format(j)
        with open(filename, 'rb') as f:
            for id_dataset in range(5):
                idx = 0
                for _ in range(956):
                    r = read_record(f, data_format_ETL8G)
                    # YO.SHIRA, YU.SHIRA, YA.SHIRA, TSU.SHIR
                    if b'.HIRA' in r[2] or b'.WO.' in r[2]:# or b'YO.SHIRA' in r[2] or b'YU.SHIRA' in r[2] or b'YA.SHIRA' in r[2] or b'TSU.SHIR' in r[2]:
                        if not b'KAI' in r[2] and not b'HEI' in r[2]:
                            #if j == 1:
                                #foo.write(r[2] + b'\n')
                            hiragana_array[idx, (j - 1) * 5 + id_dataset] = np.array(r[-1])
                            idx += 1
    np.savez_compressed("hiragana.npz", hiragana_array)
    #foo.close()
read_char()