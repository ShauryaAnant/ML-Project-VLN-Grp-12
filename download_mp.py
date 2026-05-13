import os
import tempfile
import urllib.request
import sys  # ✅ NEW: Required to dynamically write to the terminal

BASE_URL = 'http://kaldir.vc.cit.tum.de/matterport/'
RELEASE_TASKS = 'v1/tasks/'

# ✅ NEW: The Progress Bar Function
def show_progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        percent = downloaded * 100 / total_size
        
        # Convert bytes to Gigabytes (GB) for readability (since it's a 15GB file)
        downloaded_gb = downloaded / (1024**3)
        total_gb = total_size / (1024**3)
        
        # Create the visual bar: [████████------------]
        bar_length = 40
        filled_length = int(bar_length * downloaded / total_size)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        
        # '\r' forces the terminal to overwrite the current line
        sys.stdout.write(f'\r\t[{bar}] {percent:.1f}% ({downloaded_gb:.2f} GB / {total_gb:.2f} GB)')
        sys.stdout.flush()
    else:
        # Fallback if the server doesn't report the total file size
        downloaded_mb = downloaded / (1024**2)
        sys.stdout.write(f'\r\tDownloading... {downloaded_mb:.1f} MB downloaded')
        sys.stdout.flush()

def download_file(url, out_file):
    out_dir = os.path.dirname(out_file)
    if not os.path.isfile(out_file):
        print(f'\tFetching: {url}')
        
        # Create a temporary file safely
        fh, out_file_tmp = tempfile.mkstemp(dir=out_dir)
        os.close(fh) 
        
        # ✅ NEW: Attach our progress bar to the download command!
        urllib.request.urlretrieve(url, out_file_tmp, reporthook=show_progress)
        
        # Print a newline so the next text doesn't overwrite our 100% progress bar
        print() 
        
        os.rename(out_file_tmp, out_file)
    else:
        print('WARNING: skipping download of existing file ' + out_file)

if __name__ == "__main__":
    print("🚀 Starting Habitat Matterport3D Download...")
    out_dir = "data/scene_datasets/mp3d"
    
    # Ensure the folder exists
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # We only want the specific Habitat 3D meshes
    target_file = 'mp3d_habitat.zip'
    url = BASE_URL + RELEASE_TASKS + target_file
    out_file = os.path.join(out_dir, target_file)

    download_file(url, out_file)
    print("✅ Download Complete! Now you just need to unzip it.")