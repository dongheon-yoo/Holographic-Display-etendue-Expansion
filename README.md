# Follow up codes for "High Resolution Etendue Expansion for Holographic Displays" using MATLAB &amp; L-BFGS method
Link: [Github repository](https://github.com/dongheon-yoo/Holographic-Display-etendue-Expansion)

# Dependencies
* [High Resolution Etendue Expansion for Holographic Displays](https://research.fb.com/publications/high-resolution-etendue-expansion-for-holographic-displays/)
* [Wirtinger Holography for Near-Eye Displays](https://www.cs.unc.edu/~cpk/wirtinger-holography.html)
* [FMINLBFGS: Fast Limited Memory Optimizer by Dirk-Jan Kroon](https://kr.mathworks.com/matlabcentral/fileexchange/23245-fminlbfgs-fast-limited-memory-optimizer)

# Start
1. Please put the 'fminlbfgs_version2c' directory in the main directory.
2. Put the image and write the image name 'targetImName' in the 'main_gpu.m'
3. Run the 'main_gpu.m' code

# Acknowledgements
* I referred the code from Wirtinger holography article for computing gradients.
* Custom library of MATLAB for L-BFGS method by Dirk-Jan Kroon is used.
