import os

def save_ptx(compiled_kernel, file_name):
    ptx_code = compiled_kernel.asm['ptx']
    output_path = os.path.join('ptx_analysis', file_name)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(ptx_code)
    print(f"PTX saved to {output_path}")