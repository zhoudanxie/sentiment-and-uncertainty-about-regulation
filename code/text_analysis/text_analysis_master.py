import os

# Set directory
dir=os.path.dirname(os.path.realpath(__file__))
scripts_folder=f'{dir}/scripts'

for filename in os.listdir(scripts_folder):
    if filename.endswith(".py"):
        filepath = os.path.join(scripts_folder, filename)
        print(f"Running: {filename}...")
        with open(filepath) as f:
            code = f.read()
            exec(code)
    
print('End of execution!')