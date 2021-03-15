# Go to the directory from where the job was submitted (initial directory is $HOME)
cd /home/projects/ku_00017/people/simpol/scripts/currents/the_currents_of_conflict

# Load all required modules for the job
module load tools
module load anaconda3/2020.07

# This is where the work is done
# Make sure that this script is not bigger than 64kb ~ 150 lines, otherwise put in seperat script and execute from here
python tt_Rat_Mat32_sb.py 
 