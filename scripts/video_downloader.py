import sys

from yt_dlp import YoutubeDL

def main():
    if len(sys.argv) < 3:
        print(
            "Usage: python scripts/video_downloader.py /path/to/video " +
            "<browser profile>"
        )
        sys.exit(1)

    print("got here")

    URLS = [sys.argv[1]]
    ydl_opts = {
        "cookiesfrombrowser": ("chrome", sys.argv[2])
    }
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download(URLS)

if __name__ == "__main__":
    main()