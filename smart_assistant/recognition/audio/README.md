# This folder is for work looking into speaker recognition
[data.csv](recognition/audio/Models/data.csv) file is for use in training while [data.json](recognition/audio/Models/data.json) is a better formatted version to help see what each value is.

## Recognize Voices
This has been done by [Apple on their Siri](https://machinelearning.apple.com/research/personalized-hey-siri), but it is only for one single persons voice.
They have the user say a set of phrases that the os then learns to recognize that persons 
sound. The phrases are listed below in order:
1. "Hey Siri"
2. "Hey Siri"
3. "Hey Siri"
4. "Hey Siri, how is the weather today?"
5. "Hey Siri, it's me."

## TODO:
- Create Neural Network instead of binary
- Set up __init__.py file
- clean up the directory as a whole
- pylint everything