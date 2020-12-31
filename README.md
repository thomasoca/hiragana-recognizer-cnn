# hiragana-recognizer-cnn

Python script to train handwritten hiragana recognition using ETL hiragana database as the dataset

## How to use
- run [extract_file.py](extract_file.py) to unzip the training data
- run [hiragana_cnn.py](hiragana_cnn.py) to run the model and create the Tensorflow model
- run [convert_tflite.py](convert_tflite.py) to convert Tensorflow model into tflite model

## Additional information
- The hiragana dataset can be downloaded from [here](http://etlcdb.db.aist.go.jp/)

## References
- https://github.com/Nippon2019/Handwritten-Japanese-Recognition
- https://github.com/choo/etlcdb-image-extractor
- https://www.freecodecamp.org/news/build-a-handwriting-recognizer-ship-it-to-app-store-fcce24205b4b/
