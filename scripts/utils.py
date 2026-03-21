import tkinter as tk

def get_screen_size():
    try:
        print("[screen_utils.py] creating dummy window...")
        root = tk.Tk()
        root.withdraw()
        
        print("[screen_utils.py] fetching screen size...")
        w = int(root.winfo_screenwidth())
        h = int(root.winfo_screenheight())
        root.destroy()
        
        print(f"[screen_utils.py] screen w: {w} || screen h: {h}")
        return w, h
    except Exception as e:
        print(f"[screen_utils.py] Hit exception when fetching screen size: {e}")
        # Defaults to 1080p