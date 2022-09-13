""" Setup file for easy install and running """

from setuptools import setup, find_packages

setup(
    name='UWG Smart Assistant',
    version='1.1.0',
    description='This package is the main research package for a UWG centered smart assistant',
    author='Steven Kight and Ana Stanescu',
    url='https://github.com/StevenKight/UWG-Smart-Assistant',
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'wolfie = smart_assistant:main'
        ]
    },
    packages=find_packages(),
    install_requires=['tensorflow==2.7.0',
                        'scikit-learn==1.1.1',
                        'numpy==1.22.0',
                        'pandas==1.4.3',
                        'opencv-python==4.5.5.62',
                        'keras==2.7.0',
                        'cmake==3.22.1',
                        'nltk==3.6.7',
                        'dlib==19.24.0',
                        'beautifulsoup4',
                        'pillow==9.1.1',
                        'pylint==2.14.4',
                        'playsound==1.2.2',
                        'gTTS==2.2.4',
                        'SpeechRecognition==3.8.1',
                        'PyAudio==0.2.12'],
    package_data={'smart_assistant.conversation.models': ['chatbot_model.h5',
                                                            '*.pkl',
                                                            'intents.json'],
                    'smart_assistant.conversation.uwg': ['Events.json',
                                                        'students.txt'],
                    'smart_assistant.recognition.face.models': ['data.csv',
                                                                'Encodings',
                                                                'Encodings_Names'],
                    'smart_assistant.recognition.face.models.Dlib': ['*.dat'],
                    'smart_assistant.recognition.audio.Models': ['audio_model.h5',
                                                                    '*.csv']}
)
