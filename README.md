# Hyrule Warriors: Human Humming to Musical Note Transcription
## Introduction
Our team aims to create a deep learning model that can take human humming as an input and transcribe it into notes. We were inspired by the new <i>The Legend of Zelda: Ocarina of Time</i> game which includes a feature allowing the player to hum the notes of a song to activate a certain mechanic.

We want to make a model that can be used in a similar way so that a user can hum into their microphone and have a transcribed series of notes instantaneously.

## Required dependencies
### Install with pip
Use pip to install the following packages

#### Data Collection
* huggingface_hub (for HumTrans dataset)
```
pip install huggingface_hub
```
* remotezip
```
pip install remotezip
```
* pretty_midi
```
pip install pretty_midi
```
* librosa (for spectrogram creation)
```
pip install librosa
```
#### Math
* numpy
```
pip install numpy
```
* scipy
```
pip install scipy
```


## Data Collection
### HumTrans
In `./raw_data/HumTrans/test/` is `extraction.ipynb`. This is a sample notebook that shows the process of downloading humming `.wav` files and their associated MIDI files. There are also two example files `one_hum.wav` and `one_midi.wav`.