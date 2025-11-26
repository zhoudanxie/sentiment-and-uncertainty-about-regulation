#-----------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------Appendix Figures----------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
import sys, subprocess
import os

# Set directory
dir=os.path.dirname(os.path.realpath(__file__))
scripts_folder=f'{dir}/appendix_figures'
output_folder=f'{dir}/../../../figures/appendix_figures'

for filename in os.listdir(scripts_folder):
    if filename.endswith(".py"):
        filepath = os.path.join(scripts_folder, filename)
        print(f"Running {filename}...")
        with open(filepath) as f:
            code = f.read()
            exec(code)

#%% End
print(f"All appendix figures are saved in the {output_folder} folder.")


