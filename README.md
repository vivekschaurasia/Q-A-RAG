# Q&amp;A-RAG
 

# News Q&A Bot with Retrieval-Augmented Generation (RAG)

This project demonstrates the implementation of a News Q&A bot using Retrieval-Augmented Generation (RAG). The bot leverages a dataset of FAQs and integrates fallback support with OpenAI's GPT model to provide relevant responses to user queries.

---

## Author
 
**Vivek Santosh Chaurasia **  
[Medium Blog](https://medium.com/@vivekschaurasia/building-a-news-q-a-bot-a-step-by-step-guide-to-real-time-retrieval-augmented-generation-rag-783630c580ab)  
---

## Features

- **FAQ Matching:** Uses fuzzy matching to identify the closest question from the FAQ dataset and provide its corresponding answer.
- **GPT Integration:** If no close match is found in the FAQ dataset, the bot uses OpenAI's GPT model to generate a helpful response.
- **Chat Interface:** Interactive command-line chat interface for user queries.

---

## Prerequisites

### Tools and Libraries
- Python 3.7+
- [pandas](https://pandas.pydata.org/)
- [fuzzywuzzy](https://github.com/seatgeek/fuzzywuzzy)
- [openai](https://platform.openai.com/docs)

### API Key
- An OpenAI API key is required to access GPT functionality. Replace the placeholder in the code with your own API key.

---




