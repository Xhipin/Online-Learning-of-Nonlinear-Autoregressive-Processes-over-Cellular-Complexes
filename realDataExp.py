from core.experimentation import RFHORSORealExp
import os
current_file = os.path.abspath(__file__)
# Get the directory that contains this file
script_directory = os.path.dirname(current_file)
# Change the working directory to that directory
os.chdir(script_directory)

rfhorsoExp = RFHORSORealExp()
rfhorsoExp.exp_run()