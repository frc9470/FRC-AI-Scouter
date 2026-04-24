import sys

from yt_dlp import YoutubeDL

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/video_downloader.py /path/to/video")
        sys.exit(1)

    print("got here")

    URLS = [sys.argv[1]]
    
    ydl_opts = {
        # Force H.264 (avc1) video codec and m4a audio, which is required for QuickTime Player compatibility
        # Fallback to best single file mp4
        'format': 'bestvideo[height<=1080][ext=mp4][vcodec^=avc1]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]/best',
        'merge_output_format': 'mp4',
    }
    
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download(URLS)

if __name__ == "__main__":
    main()