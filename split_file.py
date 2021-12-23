your_file_name = 'D:/HCMUE/DS_2021/ds101f21project-group03/models/char_detector.h5'
CHUNK_SIZE = 40*1024*1024
file_number = 1
with open(your_file_name, 'rb') as f:
    chunk = f.read(CHUNK_SIZE)
    while chunk:
        with open(your_file_name + '_part_' + str(file_number), 'wb') as chunk_file:
            chunk_file.write(chunk)
        file_number += 1
        chunk = f.read(CHUNK_SIZE)
