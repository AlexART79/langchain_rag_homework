### 1. Prerequisites
1. Ollama installed, **gemma3:latest** and **nomic-embed-text** models are pulled
2. Project dependencies installed 
```commandline
uv sync
```

### 2. Create vector store index
```commandline
python index_data.py
```

### 3. Query the DB
```commandline
python main.py
```
Query the vectorstore:
```
>> what is UFC?
...
>> when the 1st UFC event took place?
...
>> what do you know about upcoming UFC Whitehouse event?
...

```