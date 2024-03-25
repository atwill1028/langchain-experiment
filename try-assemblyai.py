# https://zenn.dev/articles/54c3d14ab5eaf1

# YouTube動画から音声トラックを抽出し、AssemblyAIを使って文字起こしを行う

# 使用ライブラリのインポート
import yt_dlp 
import assemblyai as aai

# 文字起こし対象のYouTube動画URL
URL = 'https://www.youtube.com/watch?v=yjbFXmSBh5E'  

# yt_dlpを使って音声トラックを取得
with yt_dlp.YoutubeDL() as ydl:
    info = ydl.extract_info(URL, download=False) 
    for format in info["formats"][::-1]:
        if format["resolution"] == "audio only" and format["ext"] == "m4a":
            url = format["url"]
            break

# AssemblyAIの設定と文字起こし実行()
aai.settings.api_key = "YOUR API KEY" 
config = aai.TranscriptionConfig(speaker_labels=True, speakers_expected=2)
transcript = aai.Transcriber().transcribe(url, config)

# 文字起こし結果をファイルに保存
with open('transcript.txt', 'w') as f:
    for utterance in transcript.utterances: 
        speaker = utterance.speaker
        text = utterance.text 
        f.write(f"Speaker{speaker}: {text}\n")
```	
## 2.文字起こししたファイルの内容について質問し、回答を得ることができるようにする。
```
# LLAMAで文書の内容を理解し、質問に回答するシステム構築

# ライブラリのインポート  
import os  
from llama_index.llms import Replicate   
from llama_index import VectorStoreIndex, ServiceContext, SimpleDirectoryReader

# LLAMAの設定とラップ
os.environ["REPLICATE_API_TOKEN"] = "YOUR API KEY" # APIトークン 
llama2_7b_chat = "meta/llama-2-7b-chat:8e6975e5ed6174911a6ff3d60540dfd4844201974602551e10e9e87ab143d81e"  
llm = Replicate(model=llama2_7b_chat, temperature=0.9, additional_kwargs={"top_p": 1, "max_new_tokens": 200})  

# 文書の読み込み
loader = SimpleDirectoryReader(input_files=["./transcript.txt"])  
documents = loader.load_data()  

# インデックスとクエリエンジンの構築
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(service_context=service_context)  

# 質問と回答
response = query_engine.query("What is the relationships between SpeakerA and SpeakerB?") 
print(response)
