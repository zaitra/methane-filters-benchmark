import subprocess

def run_mag1c(path_to_img, path_to_hdr, mag1c_path):
    try:
        result = subprocess.run(["python", mag1c_path, path_to_img,"-o"], capture_output=True, text=True, check=True)
        
        print("MAG1C Output:")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error running MAG1C:")
        print(e.stderr)

if __name__ == "__main__":
    path_to_img = "./ang20191021t171902_rdn_v2x1/ang20191021t171902_rdn_v2x1_img"
    path_to_hdr = "./ang20191021t171902_rdn_v2x1/ang20191021t171902_rdn_v2x1_img.hdr"
    mag1c_path = "./mag1c_zaitra/mag1c/mag1c.py"
    
    run_mag1c(path_to_img, path_to_hdr, mag1c_path)