# Exploration - Natural Language Processing

## [NLTK - Natural Language Toolkit](https://github.com/Graphight/Learning#DS-NLP/NaturalLanguageToolkit/)

This was done following the [sentdex tutorial series](https://youtube.com/playlist?list=PLQVvvaa0QuDf2JswnfiGkliBInZnIC4HL) on NLTK, a python library aimed at text-mining.

It includes a twitter bot that streams certain keywords or tags, applying sentiment analysis to them.

The idea was to build a financial sentiment corpus which would allow me to trawl certain stock tickers on twitter or reddit.
I would then use this as a feature for my Quant machine learning algorithm. 
The hope here was to capture the public sentiment of "meme stocks" and see if it is actually indicative of market movement.

This would require building some interface tool to retrain / analyse certain phrases for sentiment. 
As I am already familiar with Streamlit I think this is a good place to start.

## [MLM - Masked Language Model](https://github.com/Graphight/Learning#DS-NLP/MaskedLanguageModel/)

This was done following a medium article [I made an AI that can study for me](https://medium.com/geekculture/i-made-an-ai-that-can-study-for-me-7c9329c54dae)

The idea is to use a pretrained roBERTa model to analyse a given text and answer queries about it. 
The medium article over-hypes what the model can do, but it is still a good example of how to use a pretrained model. 

I could also make some easy adjustments to make it much more useful