import sys

from yt_dlp import YoutubeDL

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/video_downloader.py /path/to/video")
        sys.exit(1)

    print("got here")

    URLS = [sys.argv[1]]
    with YoutubeDL() as ydl:
        ydl.download(URLS)

if __name__ == "__main__":
    main()