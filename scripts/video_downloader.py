from yt_dlp import YoutubeDL

URLS = ["https://www.youtube.com/watch?v=41kgSdbq3bI"]
with YoutubeDL() as ydl:
    ydl.download(URLS)