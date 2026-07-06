def save_ptx(compiled_kernel, output_path):
    ptx_code = compiled_kernel.asm['ptx']
    with open(output_path, 'w') as f:
        f.write(ptx_code)
    print(f"PTX saved to {output_path}")